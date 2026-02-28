"""
Judge prompt templates for VLM Constraint Eval scoring.

All prompts are pure functions: they take structured inputs and return
a ready-to-send string.  No I/O or model calls happen here.
"""

from __future__ import annotations

from typing import List


# ── C-Level: direct GPT judge ─────────────────────────────────────────────────

def prompt_direct_gpt(constraints: list, prediction: str) -> str:
    """
    Ask GPT to check whether `prediction` satisfies each constraint.

    Each constraint dict must have a "value" key.
    Output format expected:
        Judgement: ...
        Summary: Score of constraint_1: x/1, Score of constraint_2: x/1, ...
    """
    constraints_str = "\n".join(
        f"Constraint_{i + 1}: {c['value']}"
        for i, c in enumerate(constraints)
    )
    return (
        "Your task is to evaluate whether the response from an AI assistant "
        "adheres to all of the given constraints. "
        "Please follow the requirements below to make the judgment:\n"
        "1. Be strict and consistent in your assessment.\n"
        "2. You should refer to the content of image to make the judgment.\n"
        "3. For each constraint, if the response fails to fully meet the "
        "constraint, give it a score of 0. Otherwise, give it a score of 1.\n\n"
        f"<start of response>\n{prediction}\n<end of response>\n\n"
        f"<start of constraint list>\n{constraints_str}\n<end of constraint list>\n\n"
        "You must evaluate and provide an explanation for each constraint listed, "
        "ensuring no constraint is omitted. "
        "At the end, summarize the scores for all constraints in one sentence.\n\n"
        "Your output should strictly follow the format below:\n"
        "Judgement: ...\n"
        "Summary: Score of constraint_1: x/1, Score of constraint_2: x/1, "
        "Score of constraint_3: x/1, ..., Score of constraint_n: x/1."
    )


# ── P-Level: perception / VQA judge ──────────────────────────────────────────

def prompt_vision_gpt(question: str, prediction: str, ground_truth) -> str:
    """
    Ask GPT whether the model's answer covers all ground-truth points.

    Output: exactly 'right' or 'wrong'.
    """
    return (
        "You are an expert evaluator. Your task is to extract the answer from "
        "the model output and compare it with the ground truth list to determine "
        "whether the model answer covers all the points in the ground truth list. "
        "The ground truth list is provided as a JSON array of strings, and the "
        "model answer is a text string. "
        "An answer is considered correct if every element from the ground truth "
        "list appears in the model answer (substring matching is acceptable). "
        "The order does not matter. "
        "Your response should only be 'right' if the model answer fully covers "
        "the ground truth, or 'wrong' if it does not. "
        "Do not provide any additional commentary.\n\n"
        f"Question: {question}\n"
        f"Response from the model: {prediction}\n"
        f"Ground Truth List: {ground_truth}"
    )


# ── Constraint comparison (cmp_gpt) ──────────────────────────────────────────

def prompt_cmp_gpt(constraint_value: str, pred_with: str, pred_without: str) -> str:
    """
    Ask GPT whether the model satisfied a specific constraint by comparing
    the answer produced WITH the constraint vs. WITHOUT it.

    Output format:
        Reasoning: ...
        Summary: "True" / "False".
    """
    return (
        "You are an expert in judging whether the response follow the given "
        "constraint. Your task is to assess whether the model's response satisfies "
        "the given constraint and return True or False. I will provide you with the "
        "constraint and the model's response under this constraint. To assist with "
        "your evaluation, I will also provide you with the model's response to the "
        "same question without the constraint.\n\n"
        f"<start of constraint>\n{constraint_value}\n<end of constraint>\n\n"
        f"<start of response under the constraint>\n{pred_with}\n"
        f"<end of response under the constraint>\n\n"
        f"<start of response without the constraint>\n{pred_without}\n"
        f"<end of response without the constraint>\n\n"
        "**Please follow the steps below to evaluate**:\n"
        "Step 1. Compare the model's response under the constraint with its "
        "response without the constraint. If you believe these two answers are "
        "very similar, it means the model has not fully considered the impact of "
        "the constraint on the answer. Please return False.\n"
        "Step 2. Compare the model's response under the constraint with the "
        "content of the constraint. If you believe the model's response does not "
        "meet the requirements specified in the constraint, return False. "
        "Otherwise, if the response effectively satisfies the constraint, return True.\n\n"
        "Start by briefly explaining your reasoning based on the above steps. "
        "At the end, provide a one-sentence summary of your evaluation.\n\n"
        'Your output must strictly follow this format:\n'
        'Reasoning: ...\n'
        'Summary: "True" / "False".'
    )


# ── Image influence judge ─────────────────────────────────────────────────────

def prompt_image_influence(question: str, pred_with_image: str, pred_without_image: str) -> str:
    """
    Ask GPT whether having the image caused a substantive difference in the answer.

    Output: exactly 'Influenced' or 'Not influenced'.
    """
    return (
        "You are evaluating whether the availability of IMAGE caused a substantive "
        "influence on the model's answer.\n"
        "You will be given the question and two answers:\n"
        "- Answer A: produced WITH image available.\n"
        "- Answer B: produced WITHOUT image.\n\n"
        "Guidelines:\n"
        "- If Answer A contains details that plausibly come from visual evidence "
        "(objects, layout, colors, counts, attributes) and such details are "
        "missing/incorrect in Answer B, or the final conclusions differ BECAUSE "
        'of visual cues, judge it as "Influenced".\n'
        "- If both answers are essentially the same in conclusions and key details "
        '(only minor wording differs), judge "Not influenced".\n'
        "- Base your judgment strictly on the textual differences and the question. "
        "Do NOT assume seeing the image yourself.\n\n"
        f"Question: {question}\n\n"
        f"Answer A (WITH image): {pred_with_image}\n\n"
        f"Answer B (WITHOUT image): {pred_without_image}\n\n"
        "Return exactly one word: Influenced or Not influenced."
    )
