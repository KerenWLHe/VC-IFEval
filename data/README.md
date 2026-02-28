# Data Directory

## Structure

```
data/
├── Eval/
│   ├── v3.jsonl            # Benchmark task file (input to inference)
│   └── sample_50.json      # Small 50-sample subset for quick testing
└── sampled_images_50/      # Image files referenced by the benchmark tasks
    └── *.jpg / *.png
```

## Task File Format

Each line in the `.jsonl` benchmark file is a JSON object with the following fields:

```jsonc
{
  "image_name": "VQA_object_recognition_COCO_train2014_000000027989.jpg",
  "question": "Describe the objects in the image.",
  "instruction": "Describe the objects in the image.",  // alias for question
  "constraints": [
    {
      "key": "c1",
      "value": "The response must mention at least 3 distinct objects.",
      "judge": {
        "method": "direct_gpt"   // Options: direct_gpt | rule_based | cmp_gpt
      }
    }
  ],
  "answer": "ground truth answer (optional, used for P-Level tasks)",
  "tag": "C-Level"   // "C-Level" (constraint) or "P-Level" (perception/VQA)
}
```

### Constraint Judge Methods

| Method | Description |
|---|---|
| `direct_gpt` | GPT evaluates whether the response satisfies the constraint directly (with image). |
| `rule_based` | Deterministic rule check via Python functions in `data_gen/utils/function_and_compare.py`. |
| `cmp_gpt` | GPT compares the response against a version generated without the constraint. |

### Image Influence Score

When `run_without_image: true` in config, inference runs each item twice — once with the image and once without. The scorer then evaluates whether the image actually influenced the model's answer (`image_influence`: 1 = influenced, 0 = not influenced).

## Gitignore Note

Image files (`sampled_images*/`) and intermediate `.pkl` caches are excluded from git. Only the task `.jsonl` files and the small example set are tracked.
