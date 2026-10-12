"""Extract a predicted final answer from a generated reasoning trace, and
score it against a gold answer with numeric exact-match."""

from __future__ import annotations

import re

_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*([\-\d,\.\/]+)", re.IGNORECASE)
_LAST_NUMBER_RE = re.compile(r"[\-]?\d[\d,]*\.?\d*")


def extract_predicted_answer(generated_text: str) -> str | None:
    """Look for the 'Final Answer: <x>' marker; fall back to the last
    number-like token in the text if the marker is missing."""
    match = _FINAL_ANSWER_RE.search(generated_text)
    if match:
        return match.group(1).strip().rstrip(".").replace(",", "")

    numbers = _LAST_NUMBER_RE.findall(generated_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def _to_float(value: str) -> float | None:
    try:
        if "/" in value:
            num, denom = value.split("/", 1)
            return float(num) / float(denom)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    pred_f, gold_f = _to_float(predicted), _to_float(gold)
    if pred_f is not None and gold_f is not None:
        return abs(pred_f - gold_f) < 1e-4
    return predicted.strip() == gold.strip()
