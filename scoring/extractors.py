"""
Score extraction utilities for VLM Constraint Eval.

Each extractor takes a raw GPT response string and returns a numeric score
(float 0/1 or int 0/1).  All raise ValueError with a descriptive message
when the response format doesn't match expectations.
"""

from __future__ import annotations

import re


# ── direct_gpt ────────────────────────────────────────────────────────────────

def extract_direct_gpt(raw: str) -> dict[str, float]:
    """
    Parse the Summary line produced by prompt_direct_gpt.

    Expected format (case-insensitive, asterisks stripped):
        Summary: Score of constraint_1: x/1, Score of constraint_2: x/1, ...

    Returns a dict like {"constraint_1": 0.0, "constraint_2": 1.0, ...}.
    Raises ValueError if no scores can be parsed.
    """
    # Normalize whitespace and strip markdown bold markers
    cleaned = re.sub(r"\s+", " ", raw).strip()
    cleaned = re.sub(r"\*", "", cleaned)

    pattern = re.compile(
        r"Score\s+of\s+([a-zA-Z0-9_\-]+):\s*(\d+)\s*/\s*(\d+)",
        re.IGNORECASE,
    )
    matches = pattern.findall(cleaned)

    if not matches:
        raise ValueError(
            f"Could not parse direct_gpt scores from response:\n{raw}"
        )

    result = {}
    for name, numerator, denominator in matches:
        key = name.strip().lower().replace(" ", "_")
        result[key] = int(numerator) / int(denominator)
    return result


# ── vision_gpt (P-Level) ──────────────────────────────────────────────────────

def extract_vision_gpt(raw: str) -> int:
    """
    Parse a 'right' / 'wrong' response from prompt_vision_gpt.

    Returns 1 for right, 0 for wrong.
    Raises ValueError if neither word is found.
    """
    text = raw.strip().lower()
    if re.search(r"\bright\b", text):
        return 1
    if re.search(r"\bwrong\b", text):
        return 0
    raise ValueError(
        f"Could not parse vision_gpt score (expected 'right'/'wrong'):\n{raw}"
    )


# ── cmp_gpt ───────────────────────────────────────────────────────────────────

def extract_cmp_gpt(raw: str) -> int:
    """
    Parse the Summary line produced by prompt_cmp_gpt.

    Expected format:
        Summary: "True" / "False"

    Returns 1 for True, 0 for False.
    Raises ValueError if the summary line or True/False cannot be found.
    """
    lower = raw.lower()
    summary_idx = lower.rfind("summary")
    if summary_idx == -1:
        raise ValueError(f"No 'summary' section found in cmp_gpt response:\n{raw}")

    after_summary = raw[summary_idx + len("summary"):]
    match = re.search(r"\b(true|false)\b", after_summary, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"No True/False found after 'summary' in cmp_gpt response:\n{raw}"
        )
    return 1 if match.group(1).lower() == "true" else 0


# ── image influence ───────────────────────────────────────────────────────────

def extract_image_influence(raw: str) -> int:
    """
    Parse the 'Influenced' / 'Not influenced' response from
    prompt_image_influence.

    Returns 1 for Influenced, 0 for Not influenced.
    Raises ValueError if neither phrase is found.
    """
    text = raw.strip().lower()
    # Check "not influenced" first (it contains "influenced")
    if re.search(r"\bnot\s+influenced\b", text):
        return 0
    if re.search(r"\binfluenced\b", text):
        return 1
    raise ValueError(
        f"Could not parse image influence result (expected 'Influenced' / "
        f"'Not influenced'):\n{raw}"
    )
