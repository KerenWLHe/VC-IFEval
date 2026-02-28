# Data Format Reference

See [data/README.md](../data/README.md) for the benchmark task file format.

## Inference Output Format

After running `inference/run_inference.py`, a `.jsonl` file is produced where each line contains:

```jsonc
{
  // — Original fields from input task —
  "id": "VQA_object_recognition_COCO_train2014_000000027989.jpg",
  "image_name": "...",
  "image": "sampled_images_50/...",   // relative path under data/
  "question": "...",
  "instruction": "...",
  "constraints": [ ... ],
  "answer": "...",

  // — Added by inference —
  "infer_type": "main",               // "main" (with image) or "aux_no_image"
  "prediction": "The model's response text."
}
```

Two records are written per task: one with `infer_type: main` and one with `infer_type: aux_no_image`.

## Scoring Output Format

After running `scoring/run_score.py`, a `.jsonl` file is produced (one record per `main` item):

```jsonc
{
  // — All fields from inference output (main only) —
  ...

  // — Added by scorer —
  "score": {
    "c1": 1.0,
    "c2": 0.0,
    "image_influence": 1,
    "gpt_resp_direct_gpt": "Judgement: ...\nSummary: ...",
    "gpt_resp_image_influence": "Influenced",
    "total_score": 0.5
  }
}
```

A `_summary.jsonl` is also produced with aggregate accuracy across the full benchmark.
