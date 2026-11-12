"""Final-answer extraction (plan Section 3.12.3).

Rule: the last number after an answer marker, normalised for commas, units and trailing
periods. If no marker is present the trace has no extractable answer (None); we do NOT fall back
to "last number in the text", because that silently turns format failures into wrong answers.
The rule is released with the data and a sample of extractions is checked by hand.
"""
from __future__ import annotations

import re
from fractions import Fraction

ANSWER_MARKER = re.compile(r"answer\s*(?:is)?\s*[:：]?", re.IGNORECASE)
# signed number with optional thousands separators, decimals, or a simple fraction
NUMBER = re.compile(r"-?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\$?\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?")


def normalise_number(s: str) -> Fraction | None:
    s = s.replace("$", "").replace(",", "").replace(" ", "").rstrip(".")
    if not s:
        return None
    try:
        if "/" in s:
            num, den = s.split("/")
            return Fraction(int(num), int(den)) if int(den) != 0 else None
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def extract_final_answer(text: str) -> Fraction | None:
    """Last number after the LAST answer marker in the text."""
    markers = list(ANSWER_MARKER.finditer(text))
    if not markers:
        return None
    tail = text[markers[-1].end():]
    # stop at the end of that line so later chatter cannot override the answer
    line = tail.strip().split("\n", 1)[0]
    nums = NUMBER.findall(line)
    if not nums:
        return None
    return normalise_number(nums[-1])


def gsm8k_gold(answer_field: str) -> Fraction:
    """GSM8K stores the gold answer after '####'."""
    return normalise_number(answer_field.split("####")[-1].strip())


def is_correct(pred: Fraction | None, gold, tol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    gold = gold if isinstance(gold, Fraction) else normalise_number(str(gold))
    return gold is not None and abs(float(pred) - float(gold)) <= tol * max(1.0, abs(float(gold)))
