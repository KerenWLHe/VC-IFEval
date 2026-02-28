"""
Multi-GPU inference for VLM Constraint Eval (torchrun).

Uses the vlmeval VLM registry so any supported model can be swapped in.
Each rank writes its own shard; rank 0 merges and copies to processed/.

Usage:
    torchrun --nproc_per_node=4 inference/run_inference_mp.py \\
        --model_name Qwen2.5-VL-7B-Instruct \\
        --bench_name v3 \\
        --project_dir $(pwd)

    # Single-GPU (equivalent to run_inference.py but via vlmeval registry):
    torchrun --nproc_per_node=1 inference/run_inference_mp.py ...
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import torch
from tqdm import tqdm

from data_gen.utils.log import get_logger
from data_gen.utils.tools import get_real_image_key, make_prompt_v2
from vlm_eval.config import supported_VLM

logger = get_logger(__name__)


# ── Distributed helpers ───────────────────────────────────────────────────────

def get_rank_and_world_size():
    return (
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def setup_distributed(rank: int, world_size: int):
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)


def barrier(world_size: int):
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    if item.get("tag") == "P-Level":
        return item.get("question", "")
    return make_prompt_v2(
        item.get("instruction") or item.get("question", ""),
        item.get("constraints", []),
    )


# ── Resume helpers ────────────────────────────────────────────────────────────

def collect_done_indices(output_dir: str, world_size: int) -> set:
    """Read all existing rank shards and return the set of done item indices."""
    done = set()
    for i in range(world_size):
        shard = os.path.join(output_dir, f"output_rank_{i}.jsonl")
        if os.path.exists(shard):
            logger.info(f"Found existing shard {shard}, resuming...")
            with open(shard, "r") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["index"])
                    except Exception:
                        pass
    return done


# ── Merging ───────────────────────────────────────────────────────────────────

def merge_shards(output_dir: str, model_name: str, bench_name: str, world_size: int) -> str:
    """Merge per-rank shards into a single predictions.jsonl and copy to processed/."""
    merged_path = os.path.join(output_dir, "predictions.jsonl")
    with open(merged_path, "w") as fout:
        for rank in range(world_size):
            shard = os.path.join(output_dir, f"output_rank_{rank}.jsonl")
            if os.path.exists(shard):
                with open(shard, "r") as f:
                    shutil.copyfileobj(f, fout)

    # Copy to processed/ for scorer consumption
    processed_dir = os.path.join(output_dir, "..", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    dest = os.path.join(processed_dir, f"{model_name}_{bench_name}.jsonl")
    shutil.copy(merged_path, dest)
    logger.info(f"Merged output copied to {dest}")
    return merged_path


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU VLM inference")
    parser.add_argument("--model_name",   default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--bench_name",   default="v3")
    parser.add_argument("--project_dir",  default=".")
    parser.add_argument("--current_time", default=None,
                        help="Timestamp for output dir (auto-generated if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank, world_size = get_rank_and_world_size()
    setup_distributed(local_rank, world_size)
    logger.info(f"Rank {local_rank}/{world_size}")

    # ── Paths ──────────────────────────────────────────────────────────────────
    project_dir  = args.project_dir
    model_name   = args.model_name
    bench_name   = args.bench_name
    timestamp    = args.current_time or datetime.now().strftime("%Y%m%d_%H%M%S")

    image_dir    = os.path.join(project_dir, "data", "sampled_images_50")
    input_file   = os.path.join(project_dir, "data", "Eval", f"{bench_name}.jsonl")
    output_dir   = os.path.join(project_dir, "eval_results", model_name, bench_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    shard_file   = os.path.join(output_dir, f"output_rank_{local_rank}.jsonl")

    # ── Load data ──────────────────────────────────────────────────────────────
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    for i, item in enumerate(data):
        item.setdefault("index", i)

    real_image_key = get_real_image_key(data[0])
    logger.info(f"real_image_key={real_image_key}, total={len(data)}")

    # ── Resume: skip already-processed items ──────────────────────────────────
    done_indices = collect_done_indices(output_dir, world_size)
    data = [item for item in data if item["index"] not in done_indices]
    logger.info(f"Items remaining after resume filter: {len(data)}")

    barrier(world_size)

    # ── Distribute among ranks ─────────────────────────────────────────────────
    data = [item for i, item in enumerate(data) if i % world_size == local_rank]
    logger.info(f"Rank {local_rank} handling {len(data)} items")

    # ── Load model ─────────────────────────────────────────────────────────────
    if model_name not in supported_VLM:
        raise ValueError(
            f"Model '{model_name}' not found in vlm_eval/config.py supported_VLM registry. "
            f"Available: {list(supported_VLM.keys())}"
        )
    model = supported_VLM[model_name]()

    # ── Inference ──────────────────────────────────────────────────────────────
    with open(shard_file, "a") as fout:
        for item in tqdm(data, desc=f"Rank {local_rank}"):
            image_name = item.get(real_image_key, "")
            img_path   = os.path.join(image_dir, image_name)

            if not os.path.exists(img_path):
                logger.error(f"Image not found, skipping: {img_path}")
                continue

            prompt = build_prompt(item)
            logger.debug(f"prompt: {prompt}")

            try:
                prediction = model.generate([
                    {"type": "image", "value": img_path},
                    {"type": "text",  "value": prompt},
                ])
            except Exception as e:
                logger.error(f"[Rank {local_rank}] index={item['index']}: {e}")
                prediction = ""

            item["prediction"] = prediction
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    barrier(world_size)

    # ── Rank 0: merge shards ───────────────────────────────────────────────────
    if local_rank == 0:
        merged = merge_shards(output_dir, model_name, bench_name, world_size)
        logger.info(f"All done. Final output: {merged}")


if __name__ == "__main__":
    main()
