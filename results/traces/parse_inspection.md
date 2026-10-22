# Pre-registered parser inspection (docs/prereg/phase0.md §0.3)

Done on 2026-10-23 by the AI assistant (not independent of the code's author), on the
30 steps drawn with seed 42 from the parsed steps of the 776 Gemma-2-2B-IT traces
(`parse_sample.md` in the same directory; reduced profile, DECISIONS D19). It
does **not** replace the two-annotator verification with Cohen's κ (report §6).

**Question asked:** do the parsed operation, operands and stated result match the
equation written in the step text?

**Result: 30 / 30 match.**
- Plain `a op b = c` steps (27): parsed exactly, including currency symbols, unit words,
  a leading minus (`= - €341`), percentages (`= 25%`) and decimals.
- Multi-operator or chained steps (3: nos. 10, 23, 28): the parser takes the last binary
  operation and uses the value of the inner expression as an operand.
  - `(370 / 1480) * 100 = 25%` → `0.25 * 100`.
  - `10 + (30/100 * 10) = 13` → `10 + 3`.
  - `13 * 48 * 0.75 = 48 * 1.75 = 84` → `48 * 1.75 = 84`. The first link of this chain is
    arithmetically inconsistent (13·48·0.75 = 468), which is why the step is labelled
    `arithmetic`.
- No step was mis-split, mis-signed or given a wrong operand.

**Observations on the labels (not parse errors).**
- `gold_op` is the operation of the earliest gold step the trace has not yet produced
  (`rfd.parsing.steps`). A step that correctly completes a *later* gold step first is
  labelled `none` but has a different `gold_op`: this is the case for nos. 6 and 25, and for
  328 of the 1,460 `none` steps (22.5%) over the whole trace set. It is by construction
  and does not affect the pilot's operation variable, which uses the model's own operation,
  but it means `gold_op` should not be read as "the operation of the matching gold step"
  for steps labelled `none`.
- `planning` is assigned whenever a step's result is not a gold value, even if its operation
  equals `gold_op` (nos. 7, 29): it can reflect wrong operands rather than a wrong
  operation. The label does not distinguish the two.

**Limits.**
- This checks parsed steps only. 15% of steps (466 of 3,114) have no parsable equation,
  and none of those were inspected.
- The sample size gives only a coarse bound: 30 / 30 correct is consistent with an error
  rate up to about 10% (one-sided 95% upper bound 9.5%; two-sided 95% interval up to 11.6%).
