"""Multi-digit addition/subtraction problems for Baseline A (Sun et al. 2507.12379).

Sun et al. use 3-digit addition with a + b < 1000 on Gemma-2-2B-IT (1.04%
error rate). We keep their format but choose the digit length on a pilot so
the model makes enough errors (pre-registered in docs/prereg/baseline_a.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArithProblem:
    uid: str
    a: int
    b: int
    op: str  # "+" or "-"

    @property
    def answer(self) -> int:
        return self.a + self.b if self.op == "+" else self.a - self.b


def sample_problems(
    n: int,
    n_digits: int,
    op: str,
    seed: int,
    exclude: set[tuple[int, int]] | None = None,
) -> list[ArithProblem]:
    """n distinct problems with two n_digits-digit operands. For "-", a > b so the
    answer is positive; for "+", no cap on the sum (the carry digit is allowed)."""
    rng = np.random.default_rng(seed)
    lo, hi = 10 ** (n_digits - 1), 10**n_digits
    seen = set(exclude or ())
    out: list[ArithProblem] = []
    while len(out) < n:
        a, b = (int(v) for v in rng.integers(lo, hi, 2))
        if op == "-" and a <= b:
            a, b = b, a
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append(ArithProblem(f"{op}{n_digits}-{len(out):05d}", a, b, op))
    return out


def digits_by_place(value: int, n_places: int) -> list[int]:
    """[units, tens, hundreds, ...] digits of |value|, zero-padded to n_places."""
    v = abs(int(value))
    return [(v // 10**p) % 10 for p in range(n_places)]
