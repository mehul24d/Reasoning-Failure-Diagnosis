"""Extract a predicted final answer from a generated reasoning trace, and
score it against a gold answer with numeric exact-match."""

from __future__ import annotations

import re

_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*([^\n]*)", re.IGNORECASE)
_LAST_NUMBER_RE = re.compile(r"[\-]?\d[\d,]*\.?\d*")
# first number on the Final Answer line: optional sign, thousands commas,
# decimals, or a simple fraction a/b
_ANSWER_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?|-?\.\d+")
_LATEX_WRAPPERS = re.compile(r"\\boxed\{|\\text\{|\\\(|\\\)|\\\[|\\\]|\$|\\,")
_FRAC = re.compile(r"\\d?frac\{(-?\d+)\}\{(\d+)\}")


def _clean_number(tok: str) -> str:
    return tok.replace(",", "").replace(" ", "").rstrip(".")


def extract_predicted_answer(generated_text: str) -> str | None:
    """First number on the first 'Final Answer:' line (LaTeX/currency
    stripped, e.g. 'Final Answer: Arjun needs $1,402 pounds' -> '1402');
    falls back to the last number-like token in the text if the marker is
    missing or carries no number."""
    match = _FINAL_ANSWER_RE.search(generated_text)
    if match:
        line = _FRAC.sub(r"\1/\2", match.group(1))
        line = _LATEX_WRAPPERS.sub(" ", line)
        found = _ANSWER_NUMBER_RE.search(line)
        if found:
            return _clean_number(found.group(0))

    numbers = _LAST_NUMBER_RE.findall(generated_text)
    if numbers:
        return numbers[-1].replace(",", "").rstrip(".")
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
