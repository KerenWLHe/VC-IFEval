#!/bin/bash
# ============================================================
# Score only (skip inference, use an existing predictions file)
# Usage:
#   bash scripts/run_score_only.sh \
#       --inference_file /path/to/predictions.jsonl \
#       --project_dir    /path/to/vlm-constraint-eval
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INFERENCE_FILE=""
MODEL_NAME=""
BENCH_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inference_file) INFERENCE_FILE="$2"; shift 2 ;;
    --model_name)     MODEL_NAME="$2";     shift 2 ;;
    --bench_name)     BENCH_NAME="$2";     shift 2 ;;
    --project_dir)    PROJECT_DIR="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${INFERENCE_FILE}" && (-z "${MODEL_NAME}" || -z "${BENCH_NAME}") ]]; then
  echo "ERROR: provide --inference_file OR both --model_name and --bench_name"
  exit 1
fi

python scoring/run_score.py \
  ${INFERENCE_FILE:+--inference_file "${INFERENCE_FILE}"} \
  ${MODEL_NAME:+--model_name "${MODEL_NAME}"} \
  ${BENCH_NAME:+--bench_name "${BENCH_NAME}"} \
  --project_dir "${PROJECT_DIR}"
