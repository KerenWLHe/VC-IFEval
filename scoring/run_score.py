"""
Scoring entry point for VLM Constraint Eval.

Reads a predictions JSONL file produced by inference/run_inference*.py,
calls a GPT judge for each item, and writes scored results + a summary.

Usage:
    # By model/bench name (looks up processed/ dir automatically):
    python scoring/run_score.py \\
        --model_name Qwen2.5-VL-7B-Instruct \\
        --bench_name v3 \\
        --project_dir $(pwd)

    # By explicit file path:
    python scoring/run_score.py \\
        --inference_file eval_results/Qwen2.5-VL-7B-Instruct/v3/.../predictions.jsonl \\
        --project_dir $(pwd)
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import yaml
from tqdm import tqdm

from data_gen.utils.log import get_logger
from data_gen.utils.tools import close_proxy
from scoring.extractors import (
    extract_cmp_gpt,
    extract_direct_gpt,
    extract_image_influence,
)
from scoring.prompts import (
    prompt_cmp_gpt,
    prompt_direct_gpt,
    prompt_image_influence,
)
from vlm_eval.api import OpenAIWrapper
from vlm_eval.smp import dump, load
from vlm_eval.utils.mp_util import track_progress_rich

logger = get_logger(__name__)
logger.propagate = False


# ── Config ────────────────────────────────────────────────────────────────────

def load_judge_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("judge", {})


# ── GPT call helpers ──────────────────────────────────────────────────────────

def call_gpt_with_image(gpt: OpenAIWrapper, pt: str, image_path: str, retry: int = 5):
    """Call judge model with text prompt + image. Returns (ret_code, ans, response)."""
    message = [{"type": "text", "value": pt}, {"type": "image", "value": image_path}]
    for _ in range(retry):
        ret_code, ans, response = gpt.generate_inner(message)
        if ret_code == 0:
            return ret_code, ans, response
    return ret_code, ans, response


def call_gpt_text_only(gpt: OpenAIWrapper, pt: str, retry: int = 5):
    """Call judge model with text-only prompt. Returns (ret_code, ans, response)."""
    message = [{"type": "text", "value": pt}]
    for attempt in range(retry):
        ret_code, ans, response = gpt.generate_inner(message)
        if ret_code == 0:
            return ret_code, ans, response
        logger.info(f"GPT retry {attempt + 1}/{retry}")
    return ret_code, ans, response


# ── Item judge ────────────────────────────────────────────────────────────────

# Module-level globals set in main() before parallel dispatch
_gpt: OpenAIWrapper = None
_img_root: str = ""
_real_image_key: str = "image"
_aux_noimg_dict: dict = {}


def judge_one_item(item_json: str) -> tuple[int, str, dict]:
    """
    Judge a single scored item.

    Returns (ret_code, message, score_dict) where:
        ret_code == 0  → success
        ret_code == 1  → failed (will be retried by the outer loop)

    Scoring logic:
      direct_gpt (with image) for all direct_gpt constraints,
      then image_influence (text-only, once per item) if aux_no_image exists.
    """
    item = json.loads(item_json)
    score_dict: dict = {}
    constraints = item.get("constraints", [])

    direct_constraints = [c for c in constraints if c["judge"]["method"] == "direct_gpt"]

    # 1. direct_gpt: evaluate all constraints in one GPT call (with image)
    if direct_constraints:
        pt         = prompt_direct_gpt(direct_constraints, item["prediction"])
        image_path = os.path.join(_img_root, item[_real_image_key])
        ret, ans, full = call_gpt_with_image(_gpt, pt, image_path)
        if ret != 0:
            logger.error(f"[direct_gpt GPT fail] id={item.get('id')}\n{ans}")
            return 1, "direct_gpt: GPT call failed", {}
        try:
            parsed = extract_direct_gpt(ans)
            score_dict["gpt_resp_direct_gpt"] = ans
            for i, c in enumerate(direct_constraints):
                score_dict[c["key"]] = parsed.get(f"constraint_{i + 1}", 0.0)
        except ValueError as e:
            logger.error(f"[direct_gpt extract fail] id={item.get('id')}: {e}")
            return 1, "direct_gpt: score extraction failed", {}

    # 2. image_influence: one call per item (not per-constraint)
    #    Runs whenever aux_no_image prediction is available.
    pred_with    = item["prediction"]
    pred_without = _aux_noimg_dict.get(item["id"])

    if pred_without is not None:
        ins      = item.get("instruction", "")
        cons     = item.get("constraints", [])
        cons_str = "\n".join(
            f"- {c['value'] if isinstance(c, dict) and 'value' in c else str(c)}"
            for c in cons
        )
        q_text = f"Instruction:\n{ins}\nConstraints:\n{cons_str}"

        pt = prompt_image_influence(q_text, pred_with, pred_without)
        ret, ans, full = call_gpt_text_only(_gpt, pt)
        if ret != 0:
            logger.error(f"[image_influence GPT fail] id={item.get('id')}\n{ans}")
            return 1, "image_influence: GPT call failed", {}
        try:
            score_dict["image_influence"]          = extract_image_influence(ans)
            score_dict["gpt_resp_image_influence"] = ans
        except ValueError as e:
            logger.error(f"[image_influence extract fail] id={item.get('id')}: {e}")
            return 1, "image_influence: score extraction failed", {}
    else:
        logger.warning(f"No aux_no_image prediction for id={item['id']}; skipping image_influence.")

    # Compute total_score (skip gpt_resp_* keys)
    numeric_scores = {k: v for k, v in score_dict.items() if not k.startswith("gpt_resp_")}
    if not numeric_scores:
        logger.warning(f"No numeric scores for id={item.get('id')}")
        score_dict["total_score"] = 0.0
    else:
        score_dict["total_score"] = sum(numeric_scores.values()) / len(numeric_scores)

    logger.info(f"Scored id={item.get('id')}: {score_dict}")
    return 0, "success", score_dict


# ── Single scoring pass ───────────────────────────────────────────────────────

def score_once(
    main_data: list,
    params_all: list,
    indices_all: list,
    output_file: str,
    tmp_file: str,
    nproc: int,
):
    """Run one parallel scoring pass. Safe to call multiple times (resume-aware)."""
    ans = {}
    if os.path.exists(tmp_file):
        ans = load(tmp_file)
        logger.info(f"Loaded {len(ans)} cached results from {tmp_file}")

    # Only dispatch items not yet in cache
    pending_params   = [p for p, i in zip(params_all, indices_all) if i not in ans]
    pending_indices  = [i for i in indices_all if i not in ans]

    logger.info(f"Total: {len(indices_all)} | Cached: {len(ans)} | Pending: {len(pending_indices)}")

    if pending_indices:
        track_progress_rich(
            judge_one_item,
            pending_params,
            nproc=nproc,
            chunksize=nproc,
            keys=pending_indices,
            save=tmp_file,
        )

    ans = load(tmp_file)

    # Write results not yet in output_file
    done_ids: set = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    data_left = [item for item in main_data if item["id"] not in done_ids]
    logger.info(f"Writing {len(data_left)} new records to {output_file}")

    with open(output_file, "a") as fout:
        for item in tqdm(data_left, desc="Writing scored results"):
            result = ans.get(item["id"])
            if result is None:
                continue
            ret_code, msg, score_dict = result
            if ret_code != 0:
                logger.error(f"Skipping failed item id={item['id']}: {msg}")
                # Remove from cache so it will be retried next round
                del ans[item["id"]]
                continue
            item["score"] = score_dict
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    dump(ans, tmp_file)


# ── Summary generation ────────────────────────────────────────────────────────

def generate_summary(output_file: str, model_name: str, bench_name: str):
    df = pd.read_json(output_file, lines=True)
    score_sum = sum(row["score"]["total_score"] for _, row in df.iterrows())
    accuracy  = score_sum / len(df)

    print(f"\n{'='*50}")
    print(f"  Model     : {model_name}")
    print(f"  Benchmark : {bench_name}")
    print(f"  Samples   : {len(df)}")
    print(f"  Score sum : {score_sum:.4f}")
    print(f"  Accuracy  : {accuracy * 100:.2f}%")
    print(f"{'='*50}\n")

    summary = pd.DataFrame([{
        "model":     model_name,
        "bench":     bench_name,
        "score_sum": score_sum,
        "len":       len(df),
        "accuracy":  accuracy * 100.0,
    }])

    summary_path = output_file.replace(".jsonl", "_summary.jsonl")
    with open(summary_path, "w") as f:
        f.write(summary.to_json(orient="records"))
    logger.info(f"Summary saved to {summary_path}")
    return accuracy


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Score VLM predictions with a GPT judge")
    parser.add_argument("--config",         default="configs/eval_config.yaml")
    parser.add_argument("--model_name",     default=None)
    parser.add_argument("--bench_name",     default=None)
    parser.add_argument("--inference_file", default=None,
                        help="Direct path to predictions JSONL (overrides model/bench lookup)")
    parser.add_argument("--project_dir",    default=None)
    return parser.parse_args()


def main():
    global _gpt, _img_root, _real_image_key, _aux_noimg_dict

    args = parse_args()

    # Load judge config
    judge_cfg = load_judge_config(args.config)
    nproc     = judge_cfg.get("nproc", 8)
    max_retries = judge_cfg.get("max_retries", 10) if not hasattr(args, "max_retries") else 10

    # Resolve project dir
    project_dir = args.project_dir or "."

    # Resolve input file
    if args.inference_file:
        input_file = args.inference_file
        model_name = args.model_name or "unknown_model"
        bench_name = args.bench_name or "unknown_bench"
    elif args.model_name and args.bench_name:
        input_file = os.path.join(
            project_dir, "eval_results",
            args.model_name, args.bench_name, "processed",
            f"{args.model_name}_{args.bench_name}.jsonl"
        )
        model_name = args.model_name
        bench_name = args.bench_name
    else:
        raise ValueError("Provide --inference_file OR both --model_name and --bench_name")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Predictions file not found: {input_file}")

    # Derived paths
    _img_root   = os.path.join(project_dir, "data")
    output_file = input_file.replace(".jsonl", f"_{judge_cfg.get('model_name', 'gpt-4o-mini')}.jsonl")
    tmp_file    = output_file.replace(".jsonl", "_temp.pkl")

    close_proxy()

    # Read predictions
    with open(input_file) as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    # Split into main (with-image) and aux (without-image)
    NOIMG_TYPES = {"aux_no_image", "aux_noimg", "no_image", "noimg", "aux_no_vision", "no_vision"}
    main_data = [r for r in all_records if r.get("infer_type") == "main"]
    aux_data  = [r for r in all_records if str(r.get("infer_type", "")).lower() in NOIMG_TYPES]

    if not main_data:
        raise ValueError("No 'main' records found in predictions file.")

    _aux_noimg_dict = {r["id"]: r["prediction"] for r in aux_data}
    _real_image_key = _find_image_key(main_data[0])

    logger.info(f"main records: {len(main_data)} | aux_no_image records: {len(aux_data)}")

    # Init judge model
    judge_model = judge_cfg.get("model_name", "gpt-4o-mini")
    _gpt = OpenAIWrapper(
        judge_model,
        temperature=judge_cfg.get("temperature", 0),
        max_tokens=judge_cfg.get("max_tokens", 4096),
        img_detail=judge_cfg.get("img_detail", "high"),
        img_size=-1,
        timeout=judge_cfg.get("timeout", 300),
    )

    params_all  = [json.dumps(item) for item in main_data]
    indices_all = [item["id"] for item in main_data]

    # Scoring loop with retries
    for attempt in range(1, max_retries + 1):
        if os.path.exists(output_file):
            with open(output_file) as f:
                done_count = sum(1 for line in f if line.strip())
            if done_count == len(main_data):
                logger.info("All items scored.")
                break
        logger.info(f"Scoring attempt {attempt}/{max_retries}")
        score_once(main_data, params_all, indices_all, output_file, tmp_file, nproc)

    generate_summary(output_file, model_name, bench_name)


def _find_image_key(item: dict) -> str:
    for key in ("image", "image_name", "filename", "img_name",
                "image_path", "img_path", "img"):
        if item.get(key):
            return key
    return "image"


if __name__ == "__main__":
    main()
