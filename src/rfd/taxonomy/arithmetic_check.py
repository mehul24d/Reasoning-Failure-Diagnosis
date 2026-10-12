"""Regex-based arithmetic consistency check: scans a trace for
`a OP b = c` patterns and flags any where the stated result `c` does not
match the actual computed value."""

from __future__ import annotations

import re

_EQUATION_RE = re.compile(
    r"([\-]?\d[\d,]*\.?\d*)\s*([+\-*/x×])\s*([\-]?\d[\d,]*\.?\d*)\s*=\s*([\-]?\d[\d,]*\.?\d*)"
)

_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "x": lambda a, b: a * b,
    "×": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else None,
}


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def has_arithmetic_inconsistency(trace_text: str, tolerance: float = 1e-2) -> bool:
    for a_str, op, b_str, c_str in _EQUATION_RE.findall(trace_text):
        try:
            a, b, c = _to_float(a_str), _to_float(b_str), _to_float(c_str)
        except ValueError:
            continue
        result = _OPS[op](a, b)
        if result is None:
            continue
        if abs(result - c) > tolerance * max(1.0, abs(c)):
            return True
    return False
