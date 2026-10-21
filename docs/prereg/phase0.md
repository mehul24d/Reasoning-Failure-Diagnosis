# Pre-registration: Phase 0, harness checks and data

Committed before running `make phase0`. Model: Qwen2.5-1.5B-Instruct, fp16, MPS
(see DECISIONS D1).

## 0.1 TransformerLens vs HF equivalence (`rfd.experiments.phase0_equivalence`)

We trust hook results only if **all** of the following hold. Otherwise we stop
and report.

| check | statistic | threshold |
|---|---|---|
| TL `blocks.L.hook_resid_post` vs our HF decoder-layer hook (L in {0, 6, 13, 20, 27}), identical tokens | median ‖Δ‖/‖h‖ per layer | < 0.01 at every layer |
| TL logits vs HF logits | top-1 agreement over all positions | ≥ 0.99 |
| Left-padded batch vs single-sequence forward (HF) | median ‖Δ‖/‖h‖ | < 0.01 |
| Teacher-forced argmax vs batched KV-cached greedy generation | token agreement | ≥ 0.99 |

The test inputs are 3 GSM-Symbolic prompts with their gold solution appended,
plus 1 arithmetic prompt (fixed seed). Both models run in fp16, so differences
around 1e-3 are expected numerical noise.

## 0.2 Data

- GSM-Symbolic `main`. An instance is usable only if its text-parsed gold
  program ends at the `####` answer (DECISIONS D2).
- **Template-grouped split**, fixed seed 42, over the usable templates:
  train 50% / dev 20% / test 30%. Roles:
  - **train**: all fitting and every selection (layer, direction, strength).
  - **dev**: pipeline debugging only. Never used for selection, never reported
    as a result.
  - **test**: every reported number.
- 8 instances per template, a fixed seeded subset. Each instance has a NoOp
  distractor variant (DECISIONS D3/D6). We generate both variants for all
  splits.

## 0.3 Parser check

We report:
- the fraction of generated steps that have at least one parsable equation;
- the fraction of traces with a `Final Answer:` line;
- for 30 randomly drawn steps (seed 42), our own manual inspection of whether
  the parsed (operation, operands, result) matches the step text.

The 30-step check is done by the AI assistant. It does **not** replace the
two-annotator verification listed in the report's TODOs.
