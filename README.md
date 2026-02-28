# VC-IFEval

A lightweight evaluation framework for assessing **visual constraint-following ability** of Vision-Language Models (VLMs).

Given a benchmark of image-question pairs annotated with explicit constraints, the framework:
1. **Runs inference** on the target VLM (with and without image, to measure visual grounding)
2. **Scores responses** using GPT as a judge, evaluating per-constraint satisfaction and image influence

## Features

- Supports both **local HuggingFace models** (single-GPU and multi-GPU via `torchrun`) and **API models** (GPT, Claude, Gemini, etc.)
- Two-pass inference: *main* (with image) + *auxiliary* (without image) for image influence analysis
- Resume-safe: intermediate `.pkl` caches allow scoring to restart from where it left off
- Clean YAML-based configuration — no hardcoded paths or keys

## Installation

```bash
git clone https://github.com/KerenWLHe/VC-IFEval.git
cd VC-IFEval

pip install -r requirements.txt
```

Set your OpenAI API credentials:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_API_BASE=https://api.openai.com/v1  # or your custom endpoint
```

## Quick Start

### 1. Prepare your benchmark

Place your benchmark `.jsonl` file under `data/Eval/` and images under `data/sampled_images_50/`.
See [data/README.md](data/README.md) for the expected format.

### 2. Configure

Edit `configs/eval_config.yaml` to set your model path and desired parameters.

### 3. Run full evaluation

```bash
bash scripts/run_eval.sh \
    --model_name Qwen2.5-VL-7B-Instruct \
    --bench_name v3 \
    --project_dir $(pwd)
```

Or run steps individually:

```bash
# Inference only (single GPU)
python inference/run_inference.py

# Inference (multi-GPU via torchrun)
torchrun --nproc_per_node=4 inference/run_inference_mp.py \
    --model_name Qwen2.5-VL-7B-Instruct \
    --bench_name v3 \
    --project_dir $(pwd)

# Scoring only (on an existing predictions file)
bash scripts/run_score_only.sh \
    --inference_file eval_results/Qwen2.5-VL-7B-Instruct/v3/20250826_151614/predictions.jsonl \
    --project_dir $(pwd)
```

## Output

```
eval_results/
└── {model_name}/{bench_name}/{timestamp}/
    ├── predictions.jsonl          # Raw inference output (main + aux_no_image)
    ├── predictions_{judge}.jsonl  # Scored output
    └── predictions_{judge}_summary.jsonl  # Accuracy summary
```

## Project Structure

```
vlm-constraint-eval/
├── configs/            # YAML config files
├── inference/          # Inference entry points
├── scoring/            # Scoring / judging logic
├── vlm_eval/           # Core library (VLM wrappers, dataset utils, API clients)
├── data_gen/           # Benchmark data generation utilities
├── data/               # Benchmark data (images gitignored)
├── scripts/            # Shell scripts for end-to-end runs
└── docs/               # Extended documentation
```

## Citation

```bibtex
@misc{he2026empoweringreliablevisualcentricinstruction,
      title={Empowering Reliable Visual-Centric Instruction Following in MLLMs}, 
      author={Weilei He and Feng Ju and Zhiyuan Fan and Rui Min and Minhao Cheng and Yi R. Fung},
      year={2026},
      eprint={2601.03198},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.03198}, 
}
```

## License

MIT
