# Pre-registration: Baseline A, replication of Sun et al. (2507.12379)

Committed before any non-debug run of `make baseline-a`. Config:
`configs/baseline_a.yaml`. Model: `configs/harness.yaml` (DECISIONS D1/D8).

## What the paper does (verified from the arXiv HTML; DECISIONS D5)

- Gemma-2-2B-IT; 3-digit addition with a + b < 1000; 2-shot; the system
  message asks for `<<x+y=z>>`.
- Residual stream at the `=` token, every layer. Targets: the hundreds digit of
  the model output and of the ground truth. Probes: linear, logistic, MLP and
  circular.
- Probe data: about 800 examples balanced by predicted hundreds digit,
  **including errors**; 70/30 split. No error-only accuracy is reported, and
  there are no shuffled or random controls.
- Selective re-prompting with an MLP error detector at layer 25, evaluated on
  GSM8K CoT steps. Reports TP-correction (about 12%) and FP-preservation
  (95-100%), but **no task-accuracy comparison**.

## Our protocol and the differences from the paper

1. **Task size, chosen by a fixed rule before collection.** Addition, two
   n-digit operands (the carry digit is allowed). We run an error pilot with
   128 problems per n in 3..8. We pick the smallest n for which
   N = max(2000, ceil(300 / e)) ≤ 6000, where e is the pilot error rate; if no
   n qualifies, we take the most error-prone length with N = 6000. Target:
   ≥ 200 errors (brief) and about 300 by design.
2. **Prompt:** Sun's system message and user format, 2-shot with demos of the
   same length. The assistant turn is teacher-forced up to `<<a+b=`, and the
   model generates the digits. Sun's model writes that prefix itself; it is a
   deterministic copy, so we treat this as equivalent.
3. **Probe position.** Each answer digit is probed at its *emission position*,
   the token whose output predicts that digit. For the leading digit this is
   exactly Sun's `=` token. For later digits it is the previous output digit, as
   in Sun's free-form setting (App. H.3: the token before the predicted digit).
   This generalizes the 3-digit setup, where only the leading (hundreds) digit
   was ever wrong.
4. **Probes:** standardized L2 logistic regression (C = 0.1) per (layer, place).
   Sun's MLP and circular probes are not run (time). Instances are the
   independent unit; there are no templates in this task. The split is 70/30
   train/test by instance (seed 42).

## Primary analysis (protocol P1, from the brief: train on correct cases only)

- For each (layer, place), we train on correct **train** items to predict the
  gold digit; on correct cases, gold equals output.
- **Layer selection** is per place, by balanced accuracy on a held-out 20% of the
  correct train items. Error cases are never used for training or selection.
- **Evaluation on every error case**, using only cases whose output has the
  same number of digits as the gold answer, so places align. At the first
  (most significant) wrong place, we record whether the probe predicts the
  **gold digit**, the **model's wrong digit**, or another digit.
- **Controls** at the same layer: a probe trained on shuffled labels; a
  random-weight linear readout with the same standardization; and the prior,
  i.e. the gold digit's train frequency.
- **Known bias, declared in advance.** On correct cases gold equals output, so
  selecting the layer there favours late layers that encode the next token, and
  therefore the output. We report the selected-layer result as primary and the
  full layer curve as descriptive. Any "best layer" read off the error curve is
  post-hoc and will be labelled as such.

## Criteria (thresholds from the brief)

- **C1, decodability:** at the selected layer, held-out balanced accuracy of the
  gold-digit probe on correct test cases is ≥ 0.75 **and** above the
  shuffled-label probe. The 95% bootstrap CI (10,000 resamples) of the accuracy
  difference must exclude 0. Assessed at the place with the most errors.
- **C2, recovery in failures:** the gold digit is the probe's top-1 in ≥ 20% of
  error cases **and** the recovery rate exceeds the shuffled-label probe's. The
  paired bootstrap CI of the difference must exclude 0.
- **Verdict:** PASS if C1 and C2 both hold; PARTIAL if exactly one holds; FAIL
  if neither holds.

## Secondary analyses (descriptive)

- **Protocol P2 (paper):** output-digit and gold-digit probes trained on the
  mixed train split, including errors. We report per-layer test accuracy, gold
  accuracy restricted to errors, and a disagreement error detector.
- **Selective re-prompting**, run on the same arithmetic task rather than
  GSM8K CoT (a deviation from the paper):
  - **Detector:** a logistic "answer is wrong" probe at the `=` token. Its layer
    is chosen by AUROC on a 20% part of train, and its threshold maximizes
    balanced accuracy there.
  - **Re-prompt:** Sun's "suspicious" message is appended after the model's
    `<<a+b=z>>`, followed by `<<a+b=`, and the model continues greedily.
  - **Conditions on the test split:** no intervention, always re-prompt,
    probe-selective and oracle-selective. We report accuracy with 95% bootstrap
    CIs, plus the paper's TP-correction and FP-preservation.
  - **Expectation from the paper:** about 10% of true errors corrected. We make
    no directional prediction for accuracy.

## Amendment 1 (2026-10-22, before any non-debug run; DECISIONS D13)

- **Model.** Gemma-2-2B-IT, float32 with eager attention (`configs/gemma2_2b.yaml`).
  This is Sun et al.'s own model, so the model difference listed above
  disappears; Qwen2.5-1.5B becomes a secondary replication.
- **Unchanged:** the digit-length rule, N = max(2000, ⌈300/e⌉) ≤ 6000, the
  emission-position probing, the P1/P2 protocols, the C1/C2 criteria and
  every threshold. With Gemma the rule may select 3 digits, the paper's setting.
- Activations are now written in resumable shards of 256 problems (DECISIONS
  D15). This does not change any analysis.
