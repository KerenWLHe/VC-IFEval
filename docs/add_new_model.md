# How to Add a New Model

## Option A: HuggingFace Transformers Model (Custom Inference)

Edit `configs/eval_config.yaml`:

```yaml
inference:
  model_path: "/path/to/your/model"
  model_name: "YourModel-7B"
```

Then update `inference/run_inference.py` to load your model class.
Currently the inference script is wired for `Qwen2.5-VL` via `transformers`.
For other HF models, replace the `load_model_and_processor` function.

## Option B: VLMEval-style Model (Multi-GPU / torchrun)

Add a model class under `vlm_eval/vlm/your_model.py` following the interface in `vlm_eval/vlm/base.py`, then register it in `vlm_eval/config.py`.

Run with:

```bash
bash scripts/run_eval.sh --model_name YourModel-7B --bench_name v3
```

## Option C: API Model (GPT, Claude, Gemini, etc.)

Add a wrapper under `vlm_eval/api/your_api.py` following the interface in `vlm_eval/api/base.py`.
Set your credentials via environment variables (never hardcode keys).
