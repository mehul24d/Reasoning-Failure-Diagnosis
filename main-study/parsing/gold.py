"""Quantity slots and gold programs (plan Section 3.5, eq. 1, and Section 3.12.4).

slots(question)   -> Q(x): every number written with digits in the problem text, with its span.
gold_program(...) -> G(x) = (g_1..g_K), g_k = (o_k, a_k, b_k, r_k), from the solution text:
  * GSM8K: the calculator annotations <<expr=result>>;
  * GSM-Symbolic (apple/GSM-Symbolic has no annotations): the inline equations "expr = result".
Compound expressions are split into binary steps in evaluation order.

Each operand is resolved to a reference:
  ("slot", i)    the i-th quantity slot of the question,
  ("res", k)     the result of gold step k (< current step),
  ("const", v)   a number that is in neither (e.g. 60 minutes, 100 for a percentage).
If a value matches both an earlier result and a slot, or several slots, the reference is chosen
by a fixed rule and the operand is flagged ambiguous (Section 3.12.4: ambiguous matches are
excluded from the operand variable but kept for the others).

A program is `ok` only if it has at least one step, every equation's stated result equals its
exact value, and the last result equals the gold final answer. Programs that are not ok are
reported and excluded; nothing is repaired by hand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from .expr import NUM, Equation, find_equations, parse_expression, to_fraction

# a slot: a mixed number "11 1/4", a fraction "3/4", or a number (optionally with "%")
_SLOT = re.compile(rf"(?<![\w.])(?:(?P<whole>\d+)\s+(?P<mn>\d+)/(?P<md>\d+)|(?P<fn>\d+)/(?P<fd>\d+)"
                   rf"|(?P<num>{NUM})(?P<pct>%)?)(?![\w])")
_ANNOT = re.compile(r"<<([^<>=]*)=([^<>]*)>>")


@dataclass(frozen=True)
class Slot:
    index: int
    value: Fraction
    start: int
    end: int
    distractor: bool = False
    percent: bool = False          # written "n%": also matches the operand value n/100

    def matches(self, value: Fraction) -> bool:
        return value == self.value or (self.percent and value == self.value / 100)


def slots(question: str, distractor_span: tuple[int, int] | None = None) -> list[Slot]:
    """Numbers written with digits in the question. A slot inside `distractor_span` (the no-op
    clause, when the variant has one) is the distractor quantity q_delta."""
    out = []
    for m in _SLOT.finditer(question):
        if m.group("whole") is not None:
            a, b = int(m.group("mn")), int(m.group("md"))
            if b == 0 or a >= b:            # "3 4/2" is not a mixed number: two plain slots
                for g in ("whole", "mn", "md"):
                    out.append(_slot(len(out), Fraction(int(m.group(g))), m.start(g), m.end(g),
                                     distractor_span))
                continue
            v = int(m.group("whole")) + Fraction(a, b)
        elif m.group("fn") is not None:
            if int(m.group("fd")) == 0:
                continue
            v = Fraction(int(m.group("fn")), int(m.group("fd")))
        else:
            v = to_fraction(m.group("num"))
        out.append(_slot(len(out), v, m.start(), m.end(), distractor_span,
                         bool(m.group("pct"))))
    return out


def _slot(i, v, st, en, span, pct=False) -> Slot:
    d = span is not None and span[0] <= st < span[1]
    return Slot(i, v, st, en, d, pct)


Ref = tuple  # ("slot", i) | ("res", k) | ("const", Fraction)


@dataclass(frozen=True)
class GoldStep:
    op: str
    a: Fraction
    b: Fraction
    result: Fraction
    a_ref: Ref
    b_ref: Ref
    ambiguous: bool = False


@dataclass
class GoldProgram:
    steps: list[GoldStep] = field(default_factory=list)
    ok: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def answer(self) -> Fraction | None:
        return self.steps[-1].result if self.steps else None


def resolve(value: Fraction, slot_list: list[Slot], results: list[Fraction],
            near: int | None = None) -> tuple[Ref, bool]:
    """Reference for an operand value, and whether the choice was ambiguous.

    Rule: the most recent earlier result with this value wins over slots (a chain of steps
    normally consumes its previous result); among equal-valued slots the one closest to `near`
    (a character position in the question) wins, else the first. Any remaining choice is
    flagged ambiguous."""
    res = [k for k, r in enumerate(results) if r == value]
    sl = [s for s in slot_list if s.matches(value)]
    if res:
        return ("res", res[-1]), bool(sl) or len(res) > 1
    if sl:
        if len(sl) == 1:
            return ("slot", sl[0].index), False
        if near is not None:
            d = sorted(sl, key=lambda s: abs(s.start - near))
            tie = abs(d[0].start - near) == abs(d[1].start - near)
            return ("slot", d[0].index), tie
        return ("slot", sl[0].index), True
    return ("const", value), False


def _program_from_equations(eqs: list[Equation], slot_list: list[Slot],
                            final_answer: Fraction | None) -> GoldProgram:
    prog = GoldProgram()
    results: list[Fraction] = []
    for eq in eqs:
        if not eq.consistent:
            prog.issues.append(f"inconsistent equation: {eq.lhs} = {eq.rhs} (exact {eq.value})")
        base = len(results)
        for st in eq.steps:
            refs, amb = [], False
            for val, sub in ((st.a, st.a_sub), (st.b, st.b_sub)):
                if sub is not None:                      # inner node of the same expression
                    refs.append(("res", base + sub))
                    continue
                near = None
                if refs and refs[0][0] == "slot":
                    near = slot_list[refs[0][1]].start
                ref, a = resolve(val, slot_list, results, near)
                refs.append(ref)
                amb = amb or a
            prog.steps.append(GoldStep(st.op, st.a, st.b, st.result, refs[0], refs[1], amb))
            results.append(st.result)
        # the stated result is what later text refers to
        results[-1] = eq.rhs
    if not prog.steps:
        prog.issues.append("no equations")
    elif final_answer is not None and prog.steps[-1].result != final_answer:
        prog.issues.append(f"last result {prog.steps[-1].result} != answer {final_answer}")
    prog.ok = not prog.issues
    return prog


def gsm8k_equations(solution: str) -> tuple[list[Equation], int]:
    eqs, bad = [], 0
    for m in _ANNOT.finditer(solution):
        steps = parse_expression(m.group(1))
        try:
            rhs = to_fraction(m.group(2).strip().replace("$", ""))
        except (ValueError, ZeroDivisionError):
            rhs = None
        if steps is None or rhs is None:
            bad += 1
            continue
        eqs.append(Equation(m.start(), m.end(), m.group(1), rhs, steps))
    return eqs, bad


def gold_program(question: str, solution: str, final_answer, source: str,
                 distractor_span: tuple[int, int] | None = None) -> tuple[GoldProgram, list[Slot]]:
    """Gold program and slots for one problem. `source` is "gsm8k" or "gsm_symbolic"."""
    if source not in ("gsm8k", "gsm_symbolic"):
        raise ValueError(source)
    sl = slots(question, distractor_span)
    body = solution.split("####")[0]
    fa = None if final_answer is None else (
        final_answer if isinstance(final_answer, Fraction) else to_fraction(str(final_answer)))
    # the written-out equations (GSM8K calculator annotations removed: they repeat the inline
    # equation, and some inline equations have no annotation at all)
    prog = _with_unparsed(*find_equations(re.sub(r"<<[^<>]*>>", "", body)), sl, fa)
    if source == "gsm8k" and not prog.ok:
        # fall back to the annotations alone when the inline text cannot be read
        alt = _with_unparsed(*gsm8k_equations(body), sl, fa)
        if alt.ok:
            alt.issues.append("from annotations")      # informational; program is ok
            alt.ok = True
            return alt, sl
    return prog, sl


def _with_unparsed(eqs, bad, sl, fa) -> GoldProgram:
    prog = _program_from_equations(eqs, sl, fa)
    if bad:
        prog.issues.append(f"{bad} unparsed equation(s)")
        prog.ok = False
    return prog
