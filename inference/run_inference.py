"""
Single-GPU inference for VLM Constraint Eval.

For each task in the benchmark, runs two inference passes:
  - main        : with image
  - aux_no_image: without image (used for image-influence scoring)

Usage:
    python inference/run_inference.py [--config configs/eval_config.yaml] [overrides...]

    # Override individual fields without editing the yaml:
    python inference/run_inference.py --cuda_device 2 --max_new_tokens 256
"""

import argparse
import json
import logging
import os
from datetime import datetime

import torch
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI overrides (CLI wins)."""
    cfg = load_config(args.config)
    inf = cfg.get("inference", {})
    pth = cfg.get("paths", {})

    # CLI overrides
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    inf.update(overrides)
    cfg["inference"] = inf
    cfg["paths"] = pth
    return cfg


# ── Model loading ─────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model_and_processor(model_path: str, processor_path: str, torch_dtype_str: str):
    dtype = DTYPE_MAP.get(torch_dtype_str, torch.float32)
    logger.info(f"Loading model from {model_path} (dtype={torch_dtype_str}) ...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # avoid flash-attn SIGFPE issues
    )
    processor = AutoProcessor.from_pretrained(
        processor_path or model_path,
        trust_remote_code=True,
    )
    logger.info("Model and processor loaded.")
    return model, processor


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    """
    Construct the instruction prompt from a task item.

    For P-Level (perception) tasks: use the question directly.
    For C-Level (constraint) tasks: append each constraint value to the instruction.
    """
    if item.get("tag") == "P-Level":
        return item.get("question", "")

    instruction = item.get("instruction") or item.get("question", "")
    constraints = item.get("constraints", [])
    for c in constraints:
        if isinstance(c, dict):
            instruction += " " + c.get("value", "")
        elif isinstance(c, str):
            instruction += " " + c
    return instruction.strip()


# ── Single-sample inference ───────────────────────────────────────────────────

@torch.inference_mode()
def infer_with_image(
    model,
    processor,
    image_path: str,
    prompt: str,
    gen_kwargs: dict,
) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as e:
        raise RuntimeError(f"Cannot open image: {image_path}") from e

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


@torch.inference_mode()
def infer_without_image(
    model,
    processor,
    prompt: str,
    gen_kwargs: dict,
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ── Record assembly ───────────────────────────────────────────────────────────

def _image_key(item: dict) -> str:
    """Return the field name that holds the image filename."""
    for key in ("image", "image_name", "filename", "img_name",
                "image_path", "img_path", "img"):
        if item.get(key):
            return key
    return "image_name"  # fallback


def build_base_record(item: dict, image_dir: str) -> dict:
    """Build the output record skeleton shared by both inference passes."""
    img_key = _image_key(item)
    image_name = item.get(img_key, "")

    question = item.get("question") or item.get("instruction", "")

    # Normalize constraints to list-of-dicts
    raw_constraints = item.get("constraints", [])
    constraints = []
    for i, c in enumerate(raw_constraints, 1):
        if isinstance(c, dict):
            constraints.append(c)
        else:
            constraints.append({
                "key": f"c{i}",
                "value": str(c),
                "judge": {"method": "direct_gpt"},
            })

    answer = item.get("answer") or item.get("answer_32B")
    relative_image = os.path.join(os.path.basename(image_dir), image_name)

    return {
        "id":          image_name,
        "image_name":  image_name,
        "image":       relative_image,
        "question":    question,
        "instruction": question,
        "constraints": constraints,
        "answer":      answer,
        "tag":         item.get("tag", "C-Level"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU VLM inference")
    parser.add_argument("--config", default="configs/eval_config.yaml")

    # Optional CLI overrides (match keys in configs/eval_config.yaml > inference)
    parser.add_argument("--model_path",        default=None)
    parser.add_argument("--model_name",        default=None)
    parser.add_argument("--bench_name",        default=None)
    parser.add_argument("--cuda_device",       default=None)
    parser.add_argument("--torch_dtype",       default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max_new_tokens",    default=None, type=int)
    parser.add_argument("--do_sample",         default=None, action="store_true")
    parser.add_argument("--top_k",             default=None, type=int)
    parser.add_argument("--top_p",             default=None, type=float)
    parser.add_argument("--run_with_image",    default=None, action="store_true")
    parser.add_argument("--run_without_image", default=None, action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = build_config(args)
    inf  = cfg["inference"]
    pth  = cfg["paths"]

    # ── Device ───────────────────────────────────────────────────────────────
    cuda_device = str(inf.get("cuda_device", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    logger.info(f"CUDA_VISIBLE_DEVICES={cuda_device}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    project_dir  = pth.get("project_dir", ".")
    image_dir    = os.path.join(project_dir, pth.get("image_dir", "data/sampled_images_50"))
    input_tasks  = os.path.join(project_dir, pth.get("input_tasks", "data/Eval/v3.jsonl"))
    output_root  = os.path.join(project_dir, pth.get("output_dir", "eval_results"))

    model_name   = inf.get("model_name", "model")
    bench_name   = inf.get("bench_name", "bench")
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir   = os.path.join(output_root, model_name, bench_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    output_file  = os.path.join(output_dir, "predictions.jsonl")

    # ── Load data ─────────────────────────────────────────────────────────────
    ext = os.path.splitext(input_tasks)[-1].lower()
    with open(input_tasks, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    logger.info(f"Loaded {len(data)} tasks from {input_tasks}")

    # Ensure every item has an index field
    for i, item in enumerate(data):
        item.setdefault("index", i)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path = inf.get("model_path", model_name)
    processor_path = inf.get("processor_path") or model_path
    torch_dtype = inf.get("torch_dtype", "float32")
    model, processor = load_model_and_processor(model_path, processor_path, torch_dtype)

    # Generation kwargs
    gen_kwargs = {
        "max_new_tokens": inf.get("max_new_tokens", 512),
        "do_sample":      inf.get("do_sample", True),
        "top_k":          inf.get("top_k", 50),
        "top_p":          inf.get("top_p", 0.9),
        "eos_token_id":   processor.tokenizer.eos_token_id,
        "pad_token_id":   processor.tokenizer.eos_token_id,
        "use_cache":      True,
    }
    # When do_sample=False, top_k / top_p must not be set (causes warnings)
    if not gen_kwargs["do_sample"]:
        gen_kwargs.pop("top_k", None)
        gen_kwargs.pop("top_p", None)

    run_with_image    = inf.get("run_with_image",    True)
    run_without_image = inf.get("run_without_image", True)

    # ── Resume: collect already-written ids ──────────────────────────────────
    done_ids: set = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add((rec["id"], rec.get("infer_type", "")))
                except Exception:
                    pass
        logger.info(f"Resuming: {len(done_ids)} records already written.")

    # ── Inference loop ────────────────────────────────────────────────────────
    with open(output_file, "a", encoding="utf-8") as fout:
        for item in tqdm(data, desc="Inference"):
            try:
                base = build_base_record(item, image_dir)
            except Exception as e:
                logger.error(f"Failed to build record for item {item.get('index')}: {e}")
                continue

            if not base["id"] or not base["question"]:
                logger.warning(f"Skipping item with missing id/question: index={item.get('index')}")
                continue

            image_path = os.path.join(image_dir, base["image_name"])
            prompt     = build_prompt(item)

            # ── Pass 1: with image ────────────────────────────────────────────
            if run_with_image and (base["id"], "main") not in done_ids:
                try:
                    pred = infer_with_image(model, processor, image_path, prompt, gen_kwargs)
                except Exception as e:
                    logger.error(f"[main] {base['id']}: {e}")
                    pred = ""
                rec = {**base, "infer_type": "main", "prediction": pred}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

            # ── Pass 2: without image ─────────────────────────────────────────
            if run_without_image and (base["id"], "aux_no_image") not in done_ids:
                try:
                    pred = infer_without_image(model, processor, prompt, gen_kwargs)
                except Exception as e:
                    logger.error(f"[aux_no_image] {base['id']}: {e}")
                    pred = ""
                rec = {**base, "infer_type": "aux_no_image", "prediction": pred}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

    logger.info(f"Done. Output: {output_file}")


if __name__ == "__main__":
    main()
