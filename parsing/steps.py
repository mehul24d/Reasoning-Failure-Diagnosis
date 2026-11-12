"""Step parsing and alignment to the gold program (plan Algorithm 1, Section 3.12.4).

1. Split the trace into steps: "Step k:" lines for marked prompts; for terse prompts every line
   is a candidate and each equation on it is its own step (line and equation boundaries).
2. In each step take the FIRST equation u o v = w. A step without one is non-computational.
   A compound first equation (more than one operator) is kept as a step with compound=True:
   its stated result is registered as a realised result (so later steps can refer to it), but
   it has no single operation/operand pair, so it gets no error type and is reported separately.
3. Map u and v by value to earlier realised results or to slots (parsing.gold.resolve).
4. k(s) = earliest unmatched gold step whose operands are available given the matched steps < s
   and which shares an operand or the result with s.
5. Error type, in the plan's precedence: no k(s) or wrong operation -> planning; wrong operand
   set -> binding; wrong result -> arithmetic; otherwise none.
s0 = first step whose error type is not "none".

Character spans refer to the original trace text, so token positions (step-first, step-last)
can be found later from the tokenizer's offsets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from .expr import Equation, find_equations, normalise
from .gold import GoldProgram, Ref, Slot, resolve

_STEP = re.compile(r"^[ \t*#>-]*\**\s*step\s*(\d+)\s*\**\s*[:.)]", re.IGNORECASE | re.MULTILINE)
_ANSWER_LINE = re.compile(r"^[ \t*#>-]*\**\s*(?:final\s+)?answer\b", re.IGNORECASE | re.MULTILINE)


@dataclass
class Step:
    index: int
    start: int                     # character span in the trace text
    end: int
    text: str
    equation: Equation | None = None
    op: str | None = None
    a: Fraction | None = None
    b: Fraction | None = None
    result: Fraction | None = None         # stated result w
    a_ref: Ref | None = None
    b_ref: Ref | None = None
    ambiguous: bool = False
    gold: int | None = None                # aligned gold step k(s)
    error: str | None = None               # none | planning | binding | arithmetic

    @property
    def computational(self) -> bool:
        return self.equation is not None

    @property
    def compound(self) -> bool:
        return self.equation is not None and self.equation.compound


@dataclass
class ParsedTrace:
    steps: list[Step] = field(default_factory=list)
    first_error: int | None = None         # s0 (index into steps)
    n_unmatched_gold: int = 0

    @property
    def computational(self) -> list[Step]:
        return [s for s in self.steps if s.computational]


def segment(text: str, style: str) -> list[tuple[int, int]]:
    """Character spans of the steps of a trace (the final answer line is not a step)."""
    ans = list(_ANSWER_LINE.finditer(text))
    stop = ans[-1].start() if ans else len(text)
    if style == "marked":
        heads = [m.start() for m in _STEP.finditer(text) if m.start() < stop]
        if heads:
            ends = heads[1:] + [stop]
            return [(s, _rstrip(text, s, e)) for s, e in zip(heads, ends)]
        # fall back to terse rules if the model ignored the step format
    spans = []
    pos = 0
    for line in text[:stop].split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        if not line.strip():
            continue
        eqs, _ = find_equations(line)
        if len(eqs) <= 1:
            spans.append((s, e))
            continue
        # one step per equation: each runs from the end of the previous equation to the end of
        # its own (the last one to the end of the line). Equation offsets are in the
        # normalised line; a line is only split when normalising did not change its length.
        if len(normalise(line)) != len(line):
            spans.append((s, e))
            continue
        cut = s
        for k, eq in enumerate(eqs):
            end = s + eq.end if k + 1 < len(eqs) else e
            spans.append((cut, end))
            cut = end
    return spans


def _rstrip(text: str, s: int, e: int) -> int:
    while e > s and text[e - 1] in " \t\n":
        e -= 1
    return e


def parse_trace(text: str, style: str, slot_list: list[Slot], gold: GoldProgram) -> ParsedTrace:
    out = ParsedTrace()
    realised: list[Fraction] = []          # stated results of earlier computational steps
    real_step: list[int] = []              # step index of each entry of `realised`
    matched: set[int] = set()

    for i, (s, e) in enumerate(segment(text, style)):
        st = Step(i, s, e, text[s:e])
        eqs, _ = find_equations(st.text)
        if eqs:
            eq = eqs[0]
            st.equation = eq
            st.result = eq.rhs
            if not eq.compound:
                b = eq.steps[0]
                st.op, st.a, st.b = b.op, b.a, b.b
                st.a_ref, amb_a = _map(st.a, slot_list, realised, real_step, out.steps, None)
                near = slot_list[st.a_ref[1]].start if st.a_ref[0] == "slot" else None
                st.b_ref, amb_b = _map(st.b, slot_list, realised, real_step, out.steps, near)
                st.ambiguous = amb_a or amb_b
                st.gold = _align(st, gold, matched)
                if st.gold is not None:
                    matched.add(st.gold)
                st.error = _error_type(st, gold)
            realised.append(eq.rhs)
            real_step.append(i)
        out.steps.append(st)

    errs = [s.index for s in out.steps if s.error not in (None, "none")]
    out.first_error = errs[0] if errs else None
    out.n_unmatched_gold = len(gold.steps) - len(matched)
    return out


def _map(value, slot_list, realised, real_step, steps, near) -> tuple[Ref, bool]:
    """Realised operand -> ("slot", i) | ("res", gold k) | ("rres", realised step) | ("const", v).

    A realised result that was aligned to gold step k is referred to as ("res", k), so operand
    sets can be compared with the gold step's references."""
    ref, amb = resolve(value, slot_list, realised, near)
    if ref[0] == "res":
        step = steps[real_step[ref[1]]]
        ref = ("res", step.gold) if step.gold is not None else ("rres", step.index)
    return ref, amb


def _same_operand(r: Ref, g: Ref, gold: GoldProgram) -> bool:
    if r == g:
        return True
    # a constant matches a gold constant of equal value
    return r[0] == "const" and g[0] == "const" and r[1] == g[1]


def _available(g_step, matched: set[int]) -> bool:
    return all(ref[0] != "res" or ref[1] in matched for ref in (g_step.a_ref, g_step.b_ref))


def _align(st: Step, gold: GoldProgram, matched: set[int]) -> int | None:
    for k, g in enumerate(gold.steps):
        if k in matched or not _available(g, matched):
            continue
        shares = any(_same_operand(r, gr, gold) for r in (st.a_ref, st.b_ref)
                     for gr in (g.a_ref, g.b_ref)) or st.result == g.result
        if shares:
            return k
    return None


def _error_type(st: Step, gold: GoldProgram) -> str:
    if st.gold is None:
        return "planning"
    g = gold.steps[st.gold]
    if st.op != g.op:
        return "planning"
    if sorted(map(repr, (st.a_ref, st.b_ref))) != sorted(map(repr, (g.a_ref, g.b_ref))) and \
            not _const_equal_sets(st, g):
        return "binding"
    exact = {"+": st.a + st.b, "-": st.a - st.b, "*": st.a * st.b,
             "/": st.a / st.b if st.b else None}[st.op]
    if exact is None or st.result != exact:
        return "arithmetic"
    return "none"


def _const_equal_sets(st: Step, g) -> bool:
    """Operand sets equal when constants are compared by value."""
    def key(r):
        return ("const", Fraction(r[1])) if r[0] == "const" else r
    return sorted(map(repr, map(key, (st.a_ref, st.b_ref)))) == \
        sorted(map(repr, map(key, (g.a_ref, g.b_ref))))
