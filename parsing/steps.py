"""Step parsing and alignment to the gold program (plan Algorithm 1, Section 3.12.4).

1. Split the trace into steps: "Step k:" lines for marked prompts; for terse prompts every line
   is a candidate and each equation on it is its own step (line and equation boundaries).
2. In each step take the FIRST equation u o v = w. A step without one is non-computational.
   A compound first equation (more than one operator) is kept with compound=True: it has no
   single operation/operand pair, so it gets no error type, but it is aligned by its result and
   its stated result is registered so later steps can refer to it.
3. Map u and v to earlier realised results or to slots (parsing.gold.resolve).
4. Align each step to an unmatched gold step k(s) (see `_align`).
5. Error type, in the plan's precedence: no k(s) or wrong operation -> planning; wrong operand
   set -> binding; wrong result -> arithmetic; otherwise none.
s0 = first step whose error type is not "none".

Operand comparison (step 5 and alignment). Two operands are the same if they have the same
reference (the same slot, or results aligned to the same gold step) OR the same value. The
reference rule keeps a step that correctly reuses an earlier wrong result from being called a
binding error (the error is upstream); the value rule covers results the parser could not tie
to a gold step (compound expressions, restated numbers). A binding error to a different
quantity with an equal value cannot be seen in the text; such steps are flagged ambiguous.

Alignment (refines Algorithm 1 line 5, which takes the earliest gold step sharing any operand
or the result; in the validation on real traces that rule matched correct steps to the wrong
gold step). Among unmatched gold steps, the best tier wins, ties going to the earliest step
whose operands are available:
  1. same operation and same operand set      (an exact step; may still be an arithmetic error)
  2. same operand set                         (wrong operation)
  3. same result value
  4. same operation and one shared operand    (one operand bound wrongly)
  5. one shared operand
Only tiers 1-3 use up a gold step; a weak match (tiers 4-5) still gets an error type but leaves
the gold step free for a later exact match. a / c and a * (1/c) count as the same computation
(tier 1). To be fixed in the S3 pre-registration together with the error-type definitions.

Valid re-derivations. A tier-3 step (same result, different operation or operands) that is
arithmetically right and uses only problem quantities, earlier results or constants of the gold
program (never the distractor) derives the gold quantity another way (e.g. 300 - (225 + 45)
instead of 300 - 225 - 45): error "none", rederived=True.
A "planning" step (unaligned, or tier 5) that is arithmetically right, has such operands, and
either repeats an earlier computation or produces a value of the gold program is marked
`extra_valid` (redundant or re-derived). `first_error_strict` follows the plan; `first_error`
skips extra_valid steps. Both are reported.

Compound steps: aligned by their result; if their stated result is wrong (and the equation is
not an intermediate link of a chain) the step is an arithmetic error, else it has no type.
Every stated result in a step ("a = b = c") can be referred to by later steps.

Character spans refer to the original trace text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from .expr import Equation, find_equations
from .gold import GoldProgram, GoldStep, Ref, Slot, resolve

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
    tier: int | None = None                # alignment tier (1 = exact)
    error: str | None = None               # none | planning | binding | arithmetic
    extra_valid: bool = False
    rederived: bool = False                # aligned by result; a valid other derivation
    slots: list | None = field(default=None, repr=False)

    @property
    def computational(self) -> bool:
        return self.equation is not None

    @property
    def compound(self) -> bool:
        return self.equation is not None and self.equation.compound


@dataclass
class ParsedTrace:
    steps: list[Step] = field(default_factory=list)
    first_error: int | None = None         # s0, extra_valid steps not counted
    first_error_strict: int | None = None  # s0 as in the plan (every unmatched step counts)
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
        # the model ignored the step format: fall back to the terse rules
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
        # its own; the last one to the end of the line
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


# --------------------------------------------------------------------------- operands
def _same(r_ref: Ref, r_val: Fraction, g_ref: Ref, g_val: Fraction) -> bool:
    if r_ref is not None and r_ref[0] in ("slot", "res") and r_ref == g_ref:
        return True
    return r_val == g_val


def _canon(op: str, a: Fraction, b: Fraction) -> tuple[str, list[Fraction]] | None:
    """a / c is the same computation as a * (1/c)."""
    if op == "/":
        return ("*", sorted([a, 1 / b])) if b != 0 else None
    if op == "*":
        return ("*", sorted([a, b]))
    return None


def _equivalent(st: Step, g: GoldStep) -> bool:
    c1, c2 = _canon(st.op, st.a, st.b), _canon(g.op, g.a, g.b)
    return c1 is not None and c1 == c2


def _pairing(st: Step, g: GoldStep) -> int:
    """2 if both operands match (in either order), 1 if one does, 0 if none."""
    ra = [(st.a_ref, st.a), (st.b_ref, st.b)]
    ga = [(g.a_ref, g.a), (g.b_ref, g.b)]
    best = 0
    for order in (ra, ra[::-1]):
        n = sum(_same(r[0], r[1], q[0], q[1]) for r, q in zip(order, ga))
        best = max(best, n)
    return best


def _available(g: GoldStep, matched: set[int]) -> bool:
    return all(ref[0] != "res" or ref[1] in matched for ref in (g.a_ref, g.b_ref))


def _align(st: Step, gold: GoldProgram, matched: set[int]) -> tuple[int | None, int | None]:
    best = None
    for k, g in enumerate(gold.steps):
        if k in matched:
            continue
        p = _pairing(st, g)
        same_op = st.op == g.op
        if (p == 2 and same_op) or _equivalent(st, g):
            tier = 1
        elif p == 2:
            tier = 2
        elif st.result == g.result:
            tier = 3
        elif p == 1 and same_op:
            tier = 4
        elif p == 1:
            tier = 5
        else:
            continue
        key = (tier, not _available(g, matched), k)
        if best is None or key < best[0]:
            best = (key, k)
    return (best[1], best[0][0]) if best else (None, None)


def _exact(op: str, a: Fraction, b: Fraction) -> Fraction | None:
    if op == "/" and b == 0:
        return None
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]


def _error_type(st: Step, gold: GoldProgram) -> str:
    if st.gold is None:
        return "planning"
    g = gold.steps[st.gold]
    right = st.result == _exact(st.op, st.a, st.b)
    if _equivalent(st, g):
        return "none" if right else "arithmetic"
    if st.tier == 3 and right and st.result == g.result and _valid_operands(st, gold):
        st.rederived = True        # the gold quantity, derived another way (regrouping)
        return "none"
    if st.op != g.op:
        return "planning"
    if _pairing(st, g) < 2:
        return "binding"
    if not right:
        return "arithmetic"
    return "none"


def _gold_values(gold: GoldProgram) -> set[Fraction]:
    vals = set()
    for g in gold.steps:
        vals.update((g.a, g.b, g.result))
    return vals


def _valid_operands(st: Step, gold: GoldProgram) -> bool:
    """Operands are problem quantities, earlier results, or constants the gold program uses;
    never the distractor."""
    gv = _gold_values(gold)
    for ref, val in ((st.a_ref, st.a), (st.b_ref, st.b)):
        if ref[0] == "const" and val not in gv:
            return False
        if ref[0] == "slot" and st.slots is not None and st.slots[ref[1]].distractor:
            return False
    return True


# --------------------------------------------------------------------------- main
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
                st.slots = slot_list
                st.gold, st.tier = _align(st, gold, matched)
                if st.gold is not None and st.tier <= 3:
                    matched.add(st.gold)       # weak matches (tiers 4-5) do not use up a step
                st.error = _error_type(st, gold)
                right = st.result == _exact(st.op, st.a, st.b)
                repeat = any(o.op == st.op and sorted([o.a, o.b]) == sorted([st.a, st.b])
                             for o in out.steps if o.op is not None)
                if st.error == "planning" and (st.gold is None or st.tier == 5) and right and \
                        _valid_operands(st, gold) and \
                        (repeat or st.result in _gold_values(gold)):
                    st.extra_valid = True      # redundant / repeated / re-derived step
            else:
                # a compound step is aligned by its result; it has no single operation, so the
                # only error it can show is a wrong stated result
                for k, g in enumerate(gold.steps):
                    if k not in matched and g.result == eq.rhs:
                        st.gold, st.tier = k, 3
                        matched.add(k)
                        break
                if not eq.consistent:
                    st.error = "arithmetic"
            # every stated result in the step can be referred to later ("a = b = c")
            for e2 in eqs:
                realised.append(e2.rhs)
                real_step.append(i)
        out.steps.append(st)

    # a right, valid "planning" step whose result feeds a correct step is part of a valid
    # alternative derivation (e.g. 225 + 45 = 270 before 300 - 270 = 30)
    for st in reversed(out.steps):
        if st.error != "none" and not st.extra_valid:
            continue
        for ref in (st.a_ref, st.b_ref):
            if ref is not None and ref[0] == "rres":
                sup = out.steps[ref[1]]
                if sup.error == "planning" and sup.op is not None and \
                        sup.result == _exact(sup.op, sup.a, sup.b) and _valid_operands(sup, gold):
                    sup.extra_valid = True
    errs = [s for s in out.steps if s.error not in (None, "none")]
    out.first_error_strict = errs[0].index if errs else None
    lenient = [s for s in errs if not s.extra_valid]
    out.first_error = lenient[0].index if lenient else None
    out.n_unmatched_gold = len(gold.steps) - len(matched)
    return out


def _map(value, slot_list, realised, real_step, steps, near) -> tuple[Ref, bool]:
    """Realised operand -> ("slot", i) | ("res", gold k) | ("rres", realised step) | ("const", v).

    A realised result that was aligned to gold step k is referred to as ("res", k), so operand
    sets can be compared with the gold step's references."""
    ref, amb = resolve(value, slot_list, realised, near)
    if ref[0] == "res":
        step = steps[real_step[ref[1]]]
        strong = step.gold is not None and step.tier is not None and step.tier <= 3
        ref = ("res", step.gold) if strong else ("rres", step.index)
    return ref, amb
