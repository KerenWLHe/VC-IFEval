#!/bin/bash
# ============================================================
# Full evaluation pipeline: inference → scoring
# Usage:
#   bash scripts/run_eval.sh \
#       --model_name Qwen2.5-VL-7B-Instruct \
#       --bench_name v3 \
#       --project_dir /path/to/vlm-constraint-eval
# ============================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
BENCH_NAME="v3"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# ── Arg parsing ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)  MODEL_NAME="$2";  shift 2 ;;
    --bench_name)  BENCH_NAME="$2";  shift 2 ;;
    --project_dir) PROJECT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

INFERENCE_OUTPUT="${PROJECT_DIR}/eval_results/${MODEL_NAME}/${BENCH_NAME}/${CURRENT_TIME}/predictions.jsonl"

echo "========================================================"
echo "  Model     : ${MODEL_NAME}"
echo "  Benchmark : ${BENCH_NAME}"
echo "  Project   : ${PROJECT_DIR}"
echo "  Output    : ${INFERENCE_OUTPUT}"
echo "========================================================"

# ── Step 1: Inference ────────────────────────────────────────
echo "[1/2] Running inference..."
python inference/run_inference_mp.py \
  --model_name   "${MODEL_NAME}" \
  --bench_name   "${BENCH_NAME}" \
  --current_time "${CURRENT_TIME}" \
  --project_dir  "${PROJECT_DIR}"

# ── Step 2: Scoring ──────────────────────────────────────────
echo "[2/2] Running scoring..."
python scoring/run_score.py \
  --model_name    "${MODEL_NAME}" \
  --bench_name    "${BENCH_NAME}" \
  --project_dir   "${PROJECT_DIR}"

echo "Done. Results in: ${PROJECT_DIR}/eval_results/${MODEL_NAME}/${BENCH_NAME}/"
