# Pre-registration: Baseline B, replication of Yuan et al. (2605.09502)

Committed before any non-debug run of `make baseline-b`. Configs:
`configs/traces.yaml` (data) and `configs/baseline_b.yaml`. Model:
`configs/harness.yaml`.

## What the paper does (verified; DECISIONS D5)

- Qwen2.5 (1.5B-72B), Llama-3.1-8B, Phi-3.5, R1-distill. MATH-500, with 100
  problems for training and 200 for evaluation. Greedy decoding; `Step k:` ...
  `ANSWER:` format.
- Probe: standardized L2 logistic regression (C = 0.1) at the **last token of
  the trace**, one layer chosen by 5-fold CV. Qwen2.5-3B reaches AUROC 0.953;
  the first-step hidden state reaches 0.787.
- Baselines: TF-IDF text (0.59), verbalized confidence, sequence log-prob
  (0.676), self-consistency, CCS and P(True). There is no question-only baseline.
- Steering: `h' = h - a (h.w) w` along the normalized probe weights during
  generation, a in {0.5, 1, 2, 5, 8}. The layer and positions are not stated,
  and there are no random-direction controls. The best gain is about +4 points.

## Our protocol and the differences from the paper

- **Data:** GSM-Symbolic `main` plus our NoOp variants (DECISIONS D2/D3/D6), not
  MATH-500. We use 8 instances per usable template and a template-grouped split
  (train 50% / dev 20% / test 30%, seed 42). Dev is for debugging only.
- **Labels:** final-answer exact numeric match. Failure = 1.
- **Probe positions:** last generated token (Yuan); first token of step 1
  (Yuan's first-step result); last prompt token, i.e. question only (the brief).
- **Layer selection:** template-grouped 5-fold CV AUROC on **train templates
  only**. Then we refit on all of train and evaluate once on **test templates**.
- **Shuffled-label control:** one probe trained on permuted train labels
  (seed 43), at the same layer. Its CV curve is also reported for the
  last-token position.
- **Uncertainty baselines** (sign set on train): mean, min and sum of chosen-token
  log-prob; mean and max predictive entropy; length. Plus TF-IDF on the full
  trace text (1-2-grams, logistic regression, fit on train).
- **Metrics:** AUROC (primary), AUPRC, ECE (10 bins) and Brier. 95% CIs come from
  a template-clustered bootstrap with 10,000 resamples over test templates.

### Steering

- **Layer:** the layer selected for the last-token probe. **Positions:** every
  position from the last prompt token on, so the first generated token is
  already steered.
- **Directions:**
  1. `yuan_project`: Yuan's projection removal along the probe weight vector,
     a in {0.5, 1, 2, 5, 8}.
  2. `add_diffmeans`: add the unit vector of (μ_correct − μ_wrong), scaled to
     c·‖μ_correct − μ_wrong‖, c in {0.5, 1, 2, 4, 8}.
  3. `add_probe`: add the negative probe weight direction (toward "correct"),
     norm-matched to direction 2 at each c.
- **Selection set:** train templates, 64 originally-wrong plus 64
  originally-right traces. The strength maximizes net gain at the natural
  failure rate (p_w·repair − (1−p_w)·damage); ties go to the smaller strength.
- **Evaluation set:** test templates, 80 + 80 traces (stratified, seed 43). At
  the selected strength we report repair, damage and natural-rate net gain with
  template-clustered CIs.
- **Random null:** **100 matched-norm random directions**. For the additive
  types these are Gaussian directions with the same norm; for projection, random
  unit vectors with the same a. We report the null mean, its 2.5-97.5% range,
  and p = (1 + #{null ≥ real}) / 101.
- **Controls and figure:** a zero-strength run measures numerical noise (it must
  give 0 divergences). A descriptive sweep over the full grid on test, with a
  min-max band from 10 random directions per strength for `add_diffmeans`,
  feeds the figure only.
- Greedy decoding; outputs are computed with the exact divergence shortcut
  (tests/test_steering.py).

## Criteria

- **D1, diagnostic.** Last-token probe test AUROC ≥ 0.75 **and** above the
  shuffled-label probe (the template-bootstrap CI of the difference excludes 0).
- **D2, not causal.** For every steering type, the natural-rate net gain at the
  selected strength is **below the 5-point minimal effect of interest**.
- **Verdict:**
  - PASS (the paper's claim replicates) if D1 and D2 both hold.
  - PARTIAL if D1 holds but some direction reaches ≥ 5 points, which would
    contradict "not causal" for that direction. We would then also check that
    direction against the random null.
  - FAIL if D1 fails.
- **Expected, from the literature:** high AUROC and a net gain of at most about
  4 points. We report whatever we get.

## Amendment 1 (2026-10-21, before any non-debug run)

Two changes, both made before any Qwen2.5-1.5B data existed:

- **Additive strength grid** widened downward to c in {0.125, 0.25, 0.5, 1, 2, 4,
  8}. In the 0.5B debug run, c = 1 already broke every correct trace, so the
  original grid might never have contained a non-destructive strength. Yuan's
  projection grid {0.5, 1, 2, 5, 8} is unchanged.
- **Divergence rule in the steering engine.** A token now counts as changed
  only if the steered teacher-forced argmax differs from an *unsteered*
  teacher-forced pass over the same batch, rather than from the recorded token.
  The debug run showed that fp16 near-ties between batched generation and
  teacher forcing flipped tokens even under random unit-vector projections,
  which made every trace look diverged (DECISIONS D11).
