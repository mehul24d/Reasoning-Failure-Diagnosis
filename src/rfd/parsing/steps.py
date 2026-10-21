"""Split a marked CoT trace into steps and label each step against the gold program.

Step = the text from a "Step k:" marker up to the next marker (or the
"Final Answer:" line). A step's *main computation* is the last binary
operation of its last equation (the one that produces the step's result).

Per-step labels (heuristic; see docs/prereg and the manual-check CSV):
  op               main computation's operation (+ - * /)
  distractor_used  some operand of the step equals the distractor quantity d
  arithmetic_ok    every stated result equals its computed value
  operands_valid   every leaf operand is a question number (not d), a gold
                   operand/intermediate, or an earlier result in this trace
  on_gold          the stated result equals some gold intermediate/final value
  error_type       first matching of: arithmetic (not arithmetic_ok) ->
                   binding (distractor_used or not operands_valid) ->
                   planning (valid operands but result off the gold program)
                   -> none
  gold_op          op of the first gold step whose result the trace has not
                   produced yet (the step the model "should" do next)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from rfd.parsing.equations import BinStep, Equation, close, parse_equations

_STEP_RE = re.compile(r"(?m)^[ \t]*[*#]*[ \t]*Step[ \t]*(\d+)[ \t]*[*]*[ \t]*[:.]")
_FINAL_RE = re.compile(r"(?m)^[ \t]*[*#]*[ \t]*Final Answer:")
ERROR_TYPES = ("none", "planning", "binding", "arithmetic")


@dataclass
class StepLabel:
    index: int  # k in "Step k:"
    char_start: int  # offset of the "Step" marker in the generated text
    char_end: int
    n_equations: int
    op: str | None = None
    a: float | None = None
    b: float | None = None
    stated: float | None = None
    distractor_used: bool = False
    arithmetic_ok: bool | None = None
    operands_valid: bool | None = None
    on_gold: bool | None = None
    error_type: str | None = None  # None when the step has no parsable equation
    gold_op: str | None = None
    gold_result: float | None = None
    leaves: list[float] = field(default_factory=list)

    @property
    def parsed(self) -> bool:
        return self.error_type is not None

    def to_dict(self) -> dict:
        return asdict(self)


def split_steps(text: str) -> list[tuple[int, int, int]]:
    """(k, char_start, char_end) for every "Step k:" block."""
    marks = list(_STEP_RE.finditer(text))
    final = _FINAL_RE.search(text)
    end_all = final.start() if final else len(text)
    out = []
    for i, m in enumerate(marks):
        if m.start() >= end_all:
            break
        end = marks[i + 1].start() if i + 1 < len(marks) else end_all
        out.append((int(m.group(1)), m.start(), min(end, end_all)))
    return out


def _leaves(eq: Equation) -> list[float]:
    """Operands of an equation that are not results of earlier binary steps in it."""
    produced: list[float] = []
    leaves: list[float] = []
    for s in eq.steps:
        for v in (s.a, s.b):
            if any(close(v, p) for p in produced):
                produced.remove(next(p for p in produced if close(v, p)))
            else:
                leaves.append(v)
        produced.append(s.computed)
    return leaves


def _in(v: float, pool: list[float]) -> bool:
    return any(close(v, p) for p in pool)


def _in_pct(v: float, pool: list[float]) -> bool:
    """Operand membership allowing percent rescaling (7% written as 0.07 or 7)."""
    return _in(v, pool) or _in(v * 100, pool) or _in(v / 100, pool)


def label_steps(
    text: str,
    gold_steps: list[BinStep],
    question_numbers: list[float],
    distractor: float | None = None,
) -> list[StepLabel]:
    gold_values = (
        [s.result for s in gold_steps] + [s.a for s in gold_steps] + [s.b for s in gold_steps]
    )
    question_ok = [q for q in question_numbers if distractor is None or not close(q, distractor)]
    gold_results = [s.result for s in gold_steps]
    produced: list[float] = []
    labels: list[StepLabel] = []
    for k, start, end in split_steps(text):
        eqs = parse_equations(text[start:end], validate=False)
        lab = StepLabel(index=k, char_start=start, char_end=end, n_equations=len(eqs))
        # the gold step the model should do next
        pending = [g for g in gold_steps if not _in(g.result, produced)]
        if pending:
            lab.gold_op, lab.gold_result = pending[0].op, pending[0].result
        if eqs:
            main: BinStep = eqs[-1].steps[-1]
            leaves = [v for e in eqs for v in _leaves(e)]
            lab.op, lab.a, lab.b, lab.stated = main.op, main.a, main.b, main.result
            lab.leaves = leaves
            lab.distractor_used = distractor is not None and _in(distractor, leaves)
            lab.arithmetic_ok = all(s.arithmetic_ok for e in eqs for s in e.steps)
            pool = question_ok + gold_values + produced
            lab.operands_valid = all(
                _in_pct(v, pool) and not (distractor is not None and close(v, distractor))
                for v in leaves
            )
            lab.on_gold = _in(main.result, gold_results)
            if not lab.arithmetic_ok:
                lab.error_type = "arithmetic"
            elif lab.distractor_used or not lab.operands_valid:
                lab.error_type = "binding"
            elif not lab.on_gold:
                lab.error_type = "planning"
            else:
                lab.error_type = "none"
            produced += [s.result for e in eqs for s in e.steps]
        labels.append(lab)
    return labels


def first_deviation(labels: list[StepLabel]) -> StepLabel | None:
    for lab in labels:
        if lab.parsed and lab.error_type != "none":
            return lab
    return None


def content_attributable(lab: StepLabel) -> bool:
    """A content variable deviates from gold at this step: the chosen operation
    differs from the gold next-step operation, or the distractor is bound."""
    return lab.distractor_used or (lab.gold_op is not None and lab.op != lab.gold_op)


def char_to_token(token_texts: list[str], char_pos: int) -> int:
    """Index of the token whose decoded span contains `char_pos`."""
    acc = 0
    for i, t in enumerate(token_texts):
        if acc + len(t) > char_pos:
            return i
        acc += len(t)
    return max(len(token_texts) - 1, 0)
