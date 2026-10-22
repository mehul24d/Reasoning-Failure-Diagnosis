# Pre-registration: Phase 3 PILOT, feasibility slice of self-sourced repair

Committed before any non-debug run of `make pilot`. Configs: `configs/pilot.yaml`,
traces from `configs/traces.yaml`. **This is a pilot.** At this scale
statistical power is low. We report descriptive effect sizes with CIs and make
no significance claims beyond those listed here.

## Population

- **Distractor failures:** pairs where the `main` instance is solved and its
  NoOp variant fails.
- **First deviating step:** the first parsed step whose `error_type` ≠ none
  (labels from `rfd.parsing.steps`).
- **Content-attributable failure:** at that step, either the distractor
  quantity is bound, or the chosen operation ≠ the gold next-step operation.
- **Matched solved set** (for damage): for each failure, one solved NoOp trace
  from the same template. Its edit position is the step with the same step
  index, clipped to its length.
- **Split roles:** train templates are used for fitting and selection, test
  templates for every reported number, and dev is unused.

## 3.1 Variables (the 2×2 core)

All four are probed at the **first token of the step** (the `Step` token), with
the same procedure for each. That means: logistic L2 (C = 0.1) and
difference-in-means; all layers; the same train size (every variable
subsampled to the smallest variable's count of train steps); template-grouped
5-fold CV on train templates; and a shuffled-label null.

| | binary | 4-class |
|---|---|---|
| **content** | distractor quantity used at this step (NoOp traces only) | chosen operation + − × ÷ |
| **evaluation** | this step is correct (`error_type == none`) | error type: none / planning / binding / arithmetic |

## 3.2 Readability

- A layer × step-position map (positions 1-5 and 6+) for each variable: the
  probe is fit on train templates and scored on test templates. The metric is
  AUROC for binary variables and balanced accuracy for 4-class ones.
- At the layer chosen by train CV, a variable is **readable** if its held-out
  metric is ≥ 0.75 **and** above the shuffled-label probe (the
  template-bootstrap CI of the difference excludes 0).

## 3.3 RQ3 pilot (self-sourced feasibility)

- **Question:** in test-template content-attributable failures, is the gold
  value the probe's top-1 at the first deviating step? For operation errors the
  gold value is the gold op; for distractor-bound errors it is "not used".
- **Threshold:** the rate is ≥ 20% **and** above a shuffled-label probe at the
  same layer (the template-bootstrap CI of the difference excludes 0).
- **Reported alongside:** how often top-1 equals the bound (wrong) value, and,
  for ops, a step-position prior baseline.
- **Stable vs borderline:** k = 5 samples at T = 0.7 per failure (`make
  pilot-sample`). A failure is stable if 0/5 samples are correct and borderline
  otherwise. This runs only if compute allows; if skipped, we say so.

## 3.4 Mini repair test

- **Mechanics:** replay the model's own trace (teacher-forced) up to the first
  deviating step, apply one edit at that step's first token, and continue
  greedily (exact divergence shortcut).
- **Layer:** the content variable's train-selected layer. Evaluation steering
  uses the evaluation/binary variable's layer.
- **Conditions** (class means μ are computed on train templates):

| condition | edit |
|---|---|
| oracle content steering | add μ[gold] − μ[bound] |
| self-sourced gated steering | if probe top-1 (decoded) ≠ bound: add μ[decoded] − μ[bound]; otherwise no edit |
| evaluation steering → "correct" | add μ[step correct] − μ[step incorrect] |
| oracle residual patch (positive control) | set h to the residual at the same step index of the matched correct `main` run |
| random null | 100 Gaussian directions |

- **Norm matching:** every additive edit is rescaled to the same per-item norm,
  m × the content variable's mean class-mean distance. One multiple m in
  {1, 2, 4, 8} is chosen on **train-template** items by the oracle content
  condition's net gain, then used for all conditions. On solved items the
  oracle edit is zero by construction.
- **"Bound" value:** read from the model's own original step (replay setting).
  A deployable version would need a second probe for it; this is a stated
  simplification.
- **Metrics on test templates:**
  - repair: failure → correct;
  - damage: solved → wrong;
  - net gain: accuracy change over failures + solved;
  - template-clustered 95% CIs, and p = (1 + #{random ≥ real}) / 101 against
    the random null.

## Pre-registered expectations (the project hypothesis, not a claim)

- The content variables are readable.
- Oracle content steering and self-sourced steering beat the random null and
  evaluation steering.
- Evaluation steering ≈ random.
- The minimal effect of interest is 5 points of net gain. With n of roughly
  20-60 failures, the CIs will be wide, so we will call the direction
  "supported", "not supported" or "inconclusive", never "significant", unless
  a CI excludes 0.

## Amendment 1 (2026-10-22, before any non-debug run; DECISIONS D13)

- **Model.** Gemma-2-2B-IT, float32 with eager attention (`configs/gemma2_2b.yaml`);
  Qwen2.5-1.5B becomes the secondary model.
- **Unchanged:** the population, variables, probes, strength multiples
  {1, 2, 4, 8}, repair conditions, the 100 random directions and all
  thresholds.
- The T = 0.7 stability samples are now seeded per item, so that a resumed run
  draws the same samples (DECISIONS D15). The sampler settings themselves are
  unchanged.
