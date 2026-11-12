"""Step parsing and alignment to the gold program (plan Algorithm 1, Section 3.12.4).

1. Split the trace into steps: "Step k:" lines for marked prompts; for terse prompts every line
   is a candidate and each equation on it is its own step (line and equation boundaries).
2. In each step take the FIRST equation u o v = w. A step without one is non-computational.
   A compound first equation (more than one operator) is kept with compound=True: it has no
   single operation/operand pair; it is aligned by its result and typed as described under
   "Compound steps" below, and its stated result is registered so later steps can refer to it.
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
either repeats an earlier computation or produces a result of the gold program is marked
`extra_valid` (redundant or re-derived). `first_error_strict` follows the plan; `first_error`
skips extra_valid steps. Both are reported. The last computational step is never extra_valid
(nor rederived) unless it states the gold answer, and if a trace ends off the gold answer with no other error
step, `first_error` falls back to `first_error_strict` (an extra_valid step was not harmless).
With a - b and a / b, swapped operands are a binding error (tier 4), not the same step.

Compound steps: aligned by their stated result, or by their exact value when the stated result
is wrong. A wrong stated result (not an intermediate link of a chain) is an arithmetic error;
an aligned, right one is "none"; an unaligned, right one is a planning error (extra_valid if it
re-derives a gold value from valid quantities). Every stated result in a step ("a = b = c")
can be referred to by later steps.

Operand references go through tiers 1-4 (see `_map`); a right step from valid quantities whose
result feeds, directly or through such steps, a step reaching a gold value is extra_valid.
Constants are valid if the gold program uses them or their reciprocals.

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
    a_src: int | None = None               # step whose stated result is operand a (if any)
    b_src: int | None = None
    valid_leaves: bool = False             # compound: every leaf is a valid quantity
    leaf_srcs: list = field(default_factory=list)   # compound: steps whose results are leaves
    slots: list | None = field(default=None, repr=False)

    @property
    def computational(self) -> bool:
        return self.equation is not None

    @property
    def compound(self) -> bool:
        return self.equation is not None and self.equation.compound

    @property
    def result_forms(self) -> set:
        """The stated result, and n for a result written "n%" (stored as n/100)."""
        if self.result is None:
            return set()
        eq = self.equation
        pct = eq is not None and 0 < eq.end <= len(self.text) and self.text[eq.end - 1] == "%"
        return {self.result, self.result * 100} if pct else {self.result}


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


def _pairing(st: Step, g: GoldStep, ordered: bool = False) -> int:
    """2 if both operands match (in either order, or in order if `ordered`), 1 if one does,
    0 if none."""
    ra = [(st.a_ref, st.a), (st.b_ref, st.b)]
    ga = [(g.a_ref, g.a), (g.b_ref, g.b)]
    best = 0
    for order in ((ra,) if ordered else (ra, ra[::-1])):
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
        # a - b and a / b with the operands swapped are not the same step
        p_in_order = _pairing(st, g, ordered=st.op in "-/") if same_op else p
        if (p_in_order == 2 and same_op) or _equivalent(st, g):
            tier = 1
        elif p == 2 and not same_op:
            tier = 2
        elif p == 2:
            tier = 4                    # same operation, operands swapped: a binding error
        elif g.result in st.result_forms:
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


def _error_type(st: Step, gold: GoldProgram, allow_rederived: bool = True) -> str:
    if st.gold is None:
        return "planning"
    g = gold.steps[st.gold]
    right = st.result == _exact(st.op, st.a, st.b)
    if _equivalent(st, g):
        return "none" if right else "arithmetic"
    if allow_rederived and st.tier == 3 and right and g.result in st.result_forms and \
            _valid_operands(st, gold):
        st.rederived = True        # the gold quantity, derived another way (regrouping)
        return "none"
    if st.op != g.op:
        return "planning"
    if _pairing(st, g, ordered=st.op in "-/") < 2:
        return "binding"
    if not right:
        return "arithmetic"
    return "none"


def _gold_values(gold: GoldProgram) -> set[Fraction]:
    vals = set()
    for g in gold.steps:
        vals.update((g.a, g.b, g.result))
    return vals


def _gold_results(gold: GoldProgram) -> set[Fraction]:
    """Quantities the gold program derives (problem numbers alone do not count)."""
    return {g.result for g in gold.steps}


def _valid_operands(st: Step, gold: GoldProgram) -> bool:
    """Operands are problem quantities, earlier results, or constants the gold program uses;
    never the distractor."""
    return all(_valid_leaf(ref, val, gold, st.slots) for ref, val in
               ((st.a_ref, st.a), (st.b_ref, st.b)))


def _valid_leaf(ref, val, gold: GoldProgram, slot_list) -> bool:
    """A constant is valid if the gold program uses it, its reciprocal (a / c = a * (1/c)), or
    c - 1 / 1 - c (a growth or discount factor such as 1.15 for a 15% increase)."""
    gv = _gold_values(gold)
    if ref[0] == "const" and not ({val, val - 1, 1 - val} & gv) and (val == 0 or 1 / val not in gv):
        return False                 # (1 + p and 1 - p: "100 * 1.15" for a 15% increase)
    if ref[0] == "slot" and slot_list is not None and slot_list[ref[1]].distractor:
        return False
    return True


_LABEL_LINE = re.compile(r"^[ \t*-]*(?:step\s*\d+\s*[:.)]\s*)?(?P<label>[^=\n]{1,80}?)\s*=\s*"
                         r"(?P<rhs>[^=\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _split_label_equations(text: str) -> list[Equation]:
    """'Volume after spill = 72 cubic feet / 2' followed by 'Volume after spill = 36 cubic feet':
    an expression and its value stated on consecutive lines with the same label. Read as
    '72 cubic feet / 2 = 36 cubic feet'; spans point into `text`."""
    lines = list(_LABEL_LINE.finditer(text))
    out = []
    for m1, m2 in zip(lines, lines[1:]):
        if m1.group("label").strip().lower() != m2.group("label").strip().lower():
            continue
        r1, r2 = m1.group("rhs"), m2.group("rhs")
        syn = f"{r1} = {r2}"
        eqs, _ = find_equations(syn)
        if len(eqs) != 1 or eqs[0].start < 0 or eqs[0].eq_pos != len(r1) + 1:
            continue
        e = eqs[0]
        if not re.fullmatch(r"[\s$€£]*-?[\d.,]+(?:/\d+)?%?[^\d=+*/×÷-]*", r2):
            continue                         # the second line must state a single value
        p1, p2 = m1.start("rhs"), m2.start("rhs")
        e.start = p1 + e.start
        e.end = p2 + (e.end - len(r1) - 3)
        e.eq_pos = m1.end("rhs")
        out.append(e)
        break
    return out


def _leaf_sources(eq: Equation, slot_list, realised, real_step) -> list[int]:
    """Steps whose stated results are leaves of a compound expression."""
    out = []
    for b in eq.steps:
        for val, sub in ((b.a, b.a_sub), (b.b, b.b_sub)):
            if sub is None:
                ref, _ = resolve(val, slot_list, realised, None)
                if ref[0] == "res":
                    out.append(real_step[ref[1]])
    return out


def _compound_valid(eq: Equation, slot_list, realised, gold: GoldProgram) -> bool:
    """Every leaf of a compound expression is a problem quantity, an earlier stated result or a
    gold constant (never the distractor)."""
    for b in eq.steps:
        for val, sub in ((b.a, b.a_sub), (b.b, b.b_sub)):
            if sub is None:
                ref, _ = resolve(val, slot_list, realised, None)
                if not _valid_leaf(ref, val, gold, slot_list):
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
        if not eqs:
            eqs = _split_label_equations(st.text)
        if eqs:
            eq = eqs[0]
            st.equation = eq
            st.result = eq.rhs
            if not eq.compound:
                b = eq.steps[0]
                st.op, st.a, st.b = b.op, b.a, b.b
                st.a_ref, amb_a, st.a_src = _map(st.a, slot_list, realised, real_step,
                                                 out.steps, None)
                near = slot_list[st.a_ref[1]].start if st.a_ref[0] == "slot" else None
                st.b_ref, amb_b, st.b_src = _map(st.b, slot_list, realised, real_step,
                                                 out.steps, near)
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
                        (repeat or st.result_forms & _gold_results(gold)):
                    st.extra_valid = True      # redundant / repeated / re-derived step
            else:
                # a compound step has no single operation: it is aligned by its stated result
                # (or, when that is wrong, by its exact value). A wrong stated result is an
                # arithmetic error; a right one with no gold step is a planning error unless
                # it re-derives a gold value from valid quantities (extra_valid)
                forms = st.result_forms | ({eq.value, eq.value * 100} if len(st.result_forms) > 1
                                           else {eq.value})
                for vals in (st.result_forms, forms):
                    k = next((k for k, g in enumerate(gold.steps)
                              if k not in matched and g.result in vals), None)
                    if k is not None:
                        st.gold, st.tier = k, 3
                        matched.add(k)
                        break
                st.valid_leaves = _compound_valid(eq, slot_list, realised, gold)
                st.leaf_srcs = _leaf_sources(eq, slot_list, realised, real_step)
                if not eq.consistent:
                    st.error = "arithmetic"
                elif st.gold is not None:
                    st.error = "none"
                else:
                    st.error = "planning"
                    st.extra_valid = bool(st.result_forms & _gold_results(gold)) and \
                        st.valid_leaves
            # every stated result in the step can be referred to later ("a = b = c")
            for e2 in eqs:
                realised.append(e2.rhs)
                real_step.append(i)
        out.steps.append(st)

    # The last computation fixes the answer: if it is not the gold answer it is not a harmless
    # extra step.
    comp = [s for s in out.steps if s.computational]
    last = comp[-1] if comp else None
    off_answer = last is not None and gold.answer is not None and \
        gold.answer not in last.result_forms
    if off_answer and last.extra_valid:
        last.extra_valid = False
    if off_answer and last.rederived:
        # equal value by another route, but it ends the trace off the answer: a coincidence
        last.rederived = False
        last.error = _error_type(last, gold, allow_rederived=False)

    # A right step from valid quantities whose result feeds a step that reaches a gold value
    # (directly, or through further such steps) is part of a valid alternative derivation
    # (e.g. 225 + 45 = 270 before 300 - 270 = 30; 13 * 5 = 65 eggs a week before 65 * 52).
    gv = _gold_results(gold)
    feeds_gold: set[int] = set()
    for st in reversed(out.steps):
        ok = st.error == "none" or st.extra_valid
        if not ok or not (st.result_forms & gv or st.index in feeds_gold):
            continue
        for src in [st.a_src, st.b_src] + st.leaf_srcs:
            if src is None:
                continue
            sup = out.steps[src]
            if sup.error not in ("planning", "binding"):
                continue
            right_valid = (sup.op is not None and sup.result == _exact(sup.op, sup.a, sup.b)
                           and _valid_operands(sup, gold)) or \
                (sup.compound and sup.equation.consistent and sup.valid_leaves)
            if right_valid:
                sup.extra_valid = True
                feeds_gold.add(src)
    errs = [s for s in out.steps if s.error not in (None, "none")]
    out.first_error_strict = errs[0].index if errs else None
    lenient = [s for s in errs if not s.extra_valid]
    out.first_error = lenient[0].index if lenient else None
    if out.first_error is None and off_answer:
        # the trace ends off the gold answer, so a step marked extra_valid is not harmless
        out.first_error = out.first_error_strict
    out.n_unmatched_gold = len(gold.steps) - len(matched)
    return out


def _map(value, slot_list, realised, real_step, steps, near) -> tuple[Ref, bool, int | None]:
    """Realised operand -> ("slot", i) | ("res", gold k) | ("rres", realised step) | ("const", v),
    the ambiguity flag, and the step whose stated result it is (None for slots and constants).

    A realised result whose step was aligned to gold step k at tiers 1-4 is referred to as
    ("res", k), so operand sets can be compared with the gold step's references: a step that
    correctly reuses an earlier wrong result (e.g. 120 / 20 = 6 meant as gold's 180 / 20) is
    not itself a binding error; the error stays with the earlier step."""
    ref, amb = resolve(value, slot_list, realised, near)
    src = None
    if ref[0] == "res":
        step = steps[real_step[ref[1]]]
        src = step.index
        strong = step.gold is not None and step.tier is not None and step.tier <= 4
        ref = ("res", step.gold) if strong else ("rres", step.index)
    return ref, amb, src
