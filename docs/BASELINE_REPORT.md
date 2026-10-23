# Baseline report: Self-Sourced Repair (Minor Project, GP099)

**Status, 2026-10-23: the reduced-scale Gemma-2-2B-IT run is complete** (Phase 0,
Baseline A, Baseline B, pilot). It is **not** the pre-registered scale: because of a time
limit it used half the instances per template, fewer evaluation traces, 20 instead of
100 random directions and smaller strength grids (DECISIONS D19; pre-registration
amendments dated 2026-10-23, written before any of that data existed). The thresholds
and criteria are unchanged. Present it as a **baseline replication at pilot scale**.
Sample-size requirements are not met (the brief's "≥ 100 random directions"), and
every verdict that depends on a reduced quantity is marked *provisional*.

All numbers and every PASS / PARTIAL / FAIL verdict come from `make report`, in
[`BASELINE_REPORT_TABLES.md`](BASELINE_REPORT_TABLES.md), computed from the files in
`results/`. This document adds only the paper summaries and the interpretation. The
criteria are the pre-registered ones in [`prereg/`](prereg/). The table "Scale vs
pre-registration" lists exactly which quantities fall short.

Model: **Gemma-2-2B-IT** in float32 with eager attention, on one NVIDIA L4 (GCP project
`icl-failure`, [DECISIONS](DECISIONS.md) D13, D18). Qwen2.5-1.5B-Instruct is the
secondary model and was **not run**.

## 0. Harness and data (Phase 0)

| item | what was done | evidence |
|---|---|---|
| TL ↔ HF equivalence | TL `blocks.L.hook_resid_post` is compared with our HF decoder-layer hook at 5 layers, along with logits top-1. The script also checks padding invariance, that teacher forcing reproduces greedy tokens, that the effective generation config is pure greedy, and that each prompt has one BOS. Thresholds are pre-registered (phase-0 amendment 1). On a tiny random Gemma-2, hook vs TL agree to < 1e-5 (`tests/test_gemma2.py`). | `results/phase0/equivalence.json`: **all six checks pass** (worst-layer TL-vs-HF relative error 1.5e-6, top-1 agreement 1.0000, greedy reproduction 1.0000 over 990 tokens); see D14 |
| Hook / steering exactness | On a tiny random Qwen2: captures equal `hidden_states`; padding does not change activations; teacher-forced argmax equals generation; the steered divergence shortcut equals brute-force steered generation for whole-generation and single-position edits. | `tests/test_hooks.py`, `tests/test_steering.py` (passing) |
| Pure greedy decoding | The Phase-0 check run on the debug model caught that Qwen's checkpoint `generation_config` applies `repetition_penalty=1.1` even under `do_sample=False`: teacher-forced agreement was 0.941 in both fp16 and fp32. With the penalty neutralised, agreement is 0.9993 (fp16, 0.5B debug; not reported). This also affected the archived Semester-7 pilot ([DECISIONS](DECISIONS.md) D12). | `tests/test_hooks.py::test_greedy_ignores_checkpoint_repetition_penalty` |
| Steering noise control | The 0.5B debug run showed that fp16 near-ties make teacher forcing and batched generation disagree on some tokens, so random unit-vector projections appeared to change every trace. Divergence is now measured against an unsteered pass with identical batching ([DECISIONS](DECISIONS.md) D11; Baseline-B pre-registration Amendment 1, made before any real data). The zero-strength control is logged in every `steer.json`. | `rfd.steering.engine`; `results/baseline_b/steer.json`: the zero-strength control diverged on **0** traces |
| GSM-Symbolic | 100 templates × 50 instances. **The HF data has no `<<>>` annotations** (contrary to the brief). Gold programs are parsed from the solution text and accepted only when the last step equals the `####` answer. This keeps **4790/5000 instances from 97 templates**; templates 11, 50 and 85 are dropped entirely, 21 and 98 partly. The program agrees with the GitHub template annotations on **2262/2262** instances where those render exactly. | `rfd.data.gsm_symbolic`; `tests/test_gsm_symbolic.py` |
| GSM-NoOp | Not released; we built our own: one no-op clause per template with an explicit distractor slot d. **Clauses were written by the AI assistant and are not yet human-reviewed** (D6). | `rfd.data.distractor` |
| Splits | Template-grouped, seed 42: train 50% / dev 20% (debug only) / test 30%; 8 instances per template; main + NoOp variants. | `configs/traces.yaml` |
| Parser | One evaluation-validated equation parser for gold and model text. On the 0.5B debug traces it exposed and fixed: unit words in parentheses, bullet hyphens crossing clauses, and fraction right-hand sides. On the 776 Gemma traces, **85.0%** of steps parse (2,648 of 3,114) and every trace has a `Final Answer:` line. The pre-registered 30-step inspection found **30 / 30** parsed steps consistent with the step text (done by the AI assistant, not independent). | `results/traces/parse_report.json`, `parse_inspection.md` |

## 1. Baseline A: Sun et al., *Probing for Arithmetic Errors in Language Models* (2507.12379)

**What the paper claims.** From hidden states, simple probes decode both the
model's output digit and the correct digit for 3-digit addition (Gemma-2-2B-IT,
`=` token, deeper layers above 90%). Error detectors built from them reach over
90%, transfer to addition-only GSM8K CoT at about 85%, and guide selective
re-prompting that fixes about 10-12% of flagged true errors while preserving
95-100% of flagged correct steps.

**What we replicate.** Probes for the correct digit vs the model's wrong digit,
by layer. The brief's stricter protocol (train on correct cases only, test on
errors) is the primary analysis, and the paper's mixed-sample protocol is
secondary. Selective re-prompting is compared with no intervention, always
re-prompting, and oracle-selective re-prompting.

**Protocol differences (all pre-registered in [`prereg/baseline_a.md`](prereg/baseline_a.md)):**
- Same model as the paper (Gemma-2-2B-IT; pre-registration amendment 1). The
  digit length is chosen by a fixed rule so that there are ≥ 200 errors; it
  may or may not be the paper's 3 digits.
- Digits are probed at their emission position. This is Sun's `=` token for the
  leading digit, and the previous output digit for later digits.
- Logistic probes only; no MLP or circular probes.
- We add shuffled-label and random-probe controls, which the paper lacks.
- Re-prompting is evaluated on the arithmetic task, not on GSM8K CoT.
- **Paper gaps we fill:** no error-only accuracy, no controls, no task-accuracy
  comparison for re-prompting.

**Results** (5-digit addition, 5,486 problems; Gemma's error rate was 5.5%, so the digit-length
rule chose 5 digits; 285 errors, 269 analysed; tables §Baseline A):
- **The correct digit is cleanly readable when the model is right:** held-out balanced
  accuracy 0.996 at the place with most errors (shuffled-label probe 0.105), and 0.996-0.999 at
  every digit place.
- **In errors, the probe mostly reads the model's own wrong output.** The probe's top
  choice is the correct digit in **16.0%** of errors [11.5, 20.4] and the model's wrong digit
  in 84.0% [79.6, 88.5]. A shuffled-label probe gets 9.7% and a random-weight probe 5.9%, so
  the probe is 6.3 points above shuffled [0.4, 12.3].
- **Selective re-prompting does not help:** the detector is weak (test AUROC 0.614); it flags 597
  of 1,646 test answers, 44 of them real errors; 2.3% of those are corrected (the paper
  reports about 10%). Accuracy: none 94.8%, probe-selective 94.3%, always 93.7%, oracle 94.9%.
- **Verdict (pre-registered C1 ∧ C2): PARTIAL.** C1 holds; C2 fails (16.0% < 20%).
- Caveat, pre-registered: the selected layers (23-25 of 26) are late layers that tend to
  read the model's output, which biases C2 downward. Per-layer curves are in
  `results/baseline_a/analysis.json` and the figures.
- Not checked: the carry-digit place (place 5) shows a balanced accuracy of 0.000 in the
  decodability table, apparently a degenerate case; it does not enter the C1 / C2 figures above.

## 2. Baseline B: Yuan et al., *Hidden Error Awareness in CoT: The Signal Is Diagnostic, Not Causal* (2605.09502)

**What the paper claims.** A linear probe on the last-token hidden state
predicts trace correctness with AUROC 0.95 (Qwen2.5-3B, MATH-500), and 0.79
from the first step. Text-only (0.59), sequence log-prob (0.68) and verbalized
confidence are weak. Four interventions fail: steering (projection removal
along the probe, at most about +4 points), probe-guided best-of-N,
self-correction, and patching.

**What we replicate.** Failure probes by layer at the last token, the first
step and the question only, compared with token-uncertainty and TF-IDF
baselines. Steering along the failure direction during generation, with
strength chosen on train templates and 100 matched-norm random directions on
test templates.

**Protocol differences ([`prereg/baseline_b.md`](prereg/baseline_b.md)):**
- GSM-Symbolic main + NoOp instead of MATH-500, with template-grouped splits.
- We add a question-only probe, a shuffled-label control and random-direction
  nulls; the paper has none of these.
- We add additive diff-in-means and probe-weight steering (the brief) next to
  the paper's projection removal.
- The paper does not state its steering layer or positions. We use the
  train-selected probe layer and every position from the last prompt token on.
- No best-of-N, self-correction or patching (out of scope for this baseline).

**Results** (776 traces, 48 train / 30 test templates, reduced scale; tables §Baseline B):
- **Probe (D1): not met.** The last-token logistic probe (layer 20) reaches held-out
  AUROC **0.710** [0.580, 0.827], below 0.75, and only 0.126 [-0.009, 0.263] above its
  shuffled-label control, whose interval includes 0. At the first step the logistic
  probe is at 0.499 [0.356, 0.643], and on the question alone 0.589 [0.445, 0.725]. Yuan et
  al. report 0.95 on MATH-500 with Qwen2.5-3B.
- **The probe is no better than simple uncertainty baselines:** sequence log-prob 0.732
  [0.608, 0.831], max token entropy 0.734 [0.618, 0.830]. Calibration is poor (ECE 0.198; Brier
  0.253 against a base-rate Brier of 0.252).
- **Steering (D2): holds.** At layer 20, no direction's net gain reaches 5 points, and none
  beats the random null (p = 0.29 to 1.00): Yuan's projection 0.0 [0.0, 0.0]; additive
  diff-in-means +1.3 [0.0, 4.2]; additive probe direction +1.5 [-3.7, 7.6] (random-direction
  means 0.4-0.7, n = 20 each). On the selection set, stronger additive steering lowers accuracy: down to
  -7.6 points (diff-in-means) and -18.4 (probe direction) at the largest strength.
- **Verdict: FAIL** (D1 not met; D2 met), **provisional: reduced scale.** The steering result is
  consistent with the paper's "not causal" statement, but with 80 test traces the probe
  direction's interval reaches +7.6, so a 5-point effect is not ruled out. The "diagnostic"
  half of the claim is **not** replicated at this scale: the probe gives no more than cheap
  uncertainty baselines, with only 384 training traces for 2,304-dimensional features.

## 3. Phase 3 PILOT: feasibility of self-sourced repair

Clearly a pilot ([`prereg/pilot.md`](prereg/pilot.md)): 2×2 variable
readability, the RQ3 decodability check, and a mini repair test on distractor
failures with a matched solved set. Statistical power is low by design.
**Results** (tables §Phase 3; 3,114 steps; 74 distractor failures, 55 of them
content-attributable, 15 of those in the 10 test templates):
- **Readability (RQ1).** Held-out, logistic probes at the first token of a step:

  | variable | metric [95% CI] | shuffled | readable (≥ 0.75, above shuffled) |
  |---|---|---|---|
  | content / binary: distractor used | AUROC 0.865 [0.750, 0.938] | 0.453 | yes |
  | content / 4-class: operation | balanced acc. 0.524 [0.449, 0.601] | 0.218 | no |
  | evaluation / binary: step correct | AUROC 0.629 [0.531, 0.722] | 0.536 | no |
  | evaluation / 4-class: error type | balanced acc. 0.302 [0.252, 0.353] | 0.259 | no |

  Only one of the four variables passes. Its pattern fits the hypothesis: the binary content
  variable is much more readable than the binary evaluation variable (0.865 vs 0.629), but the
  4-class operation variable is not readable at the threshold, so "content variables are
  readable" is supported for one of the two.
- **RQ3.** The gold value is the probe's top choice in 46.7% [21.4, 75.0] of the 15 test-template
  failures, but the shuffled-label probe gets 66.7% [30.8, 88.9], because for the binary
  variable the gold value ("distractor not used") is the majority class. The difference is -20.0
  points [-52.9, 33.3]; **the criterion (≥ 20% and above shuffled) is not met.** With 15
  failures the data cannot tell the two apart.
- **Repair test (RQ2), at the pre-registered strengths ({1, 2, 4, 8} × the edit norm):
  uninformative.** In every condition, including the oracle content edit and the oracle
  residual patch, **no item's continuation changed** (0 of 22), so repair, damage and net are
  all 0.0. This test shows no effect because the edit never alters the output.
- **Exploratory diagnostic (D20; not pre-registered, not part of the verdicts).** To check that
  the edit is applied at all, the same test was run at 64× and 512×:
  - These edits do change continuations: oracle content steering changed 68% of items, the
    self-sourced gated edit 59%, evaluation steering 100%. So the machinery works, and the
    pre-registered strengths are far too weak for a one-token edit.
  - But the oracle content edit (net +22.7 points [5.0, 42.1]) is not distinguishable from
    random directions of the same norm (means of 2: +18.2 at 512×, +15.9 at 64×) or from evaluation steering
    (+13.6 [0.0, 26.1]); at these sizes, any large perturbation "repairs" some failures.
    The oracle residual patch changed 0 items. There is no sign of a content-specific effect.

## 4. Feasibility verdict and implications for Semester 8

Everything here is at pilot scale and **provisional**; none of it is a significance claim.

- **RQ1, readability: partly supported.** The binary content variable (distractor used) is
  readable and well above the binary evaluation variable (AUROC 0.865 vs 0.629); the
  4-class operation variable (0.524) and both evaluation variables are below the 0.75
  threshold. One of four variables passes. Semester-8 implication: build the method around
  binary content variables first, and do not rely on 4-class operation readout.
- **RQ2, causal asymmetry: inconclusive.** At the pre-registered strengths, single-token edits
  change nothing, so the test has no power. In the exploratory check, large edits change
  outputs but not in a content-specific way (oracle +22.7 vs random +15.9 to +18.2, with 15
  failures). Semester-8 implication: the repair test needs a redesign, with edits across the
  whole step (as Baseline B steering does) or at the operation's tokens, norms calibrated so
  that edits change some outputs but not all, and at least 100 test failures and ≥ 100
  random directions.
- **RQ3, self-sourcing: not shown.** The gold value is not more decodable than a majority-class
  guess in 15 test failures, and in Baseline A the correct digit is the probe's top choice in
  16.0% of errors (below the 20% bar). Semester-8 implication: decodability in failures needs a
  baseline that does not reward the majority class, and probes trained to separate gold from
  bound values rather than to predict the output.
- **The baselines.** Sun et al.'s decodability claim holds for correct cases, and the correct
  digit is carried above chance in errors (16.0% vs 9.7% shuffled), but probe-guided
  re-prompting corrected 2.3% of flagged errors and did not raise accuracy. Yuan et al.'s probe
  did not reach their AUROC at this scale, and probe-direction steering did not repair
  failures, matching "not causal". These results support the project's premise that
  evaluation-style signals and probe-guided steering do not repair reasoning, but they do not
  yet show that content editing does.

## 5. Limitations

- **Reduced scale (D19).** Traces use 4 instances per template (776 traces). Baseline B
  steering used 64 selection and 80 test traces, 20 random directions per condition (not
  100), 2 and 4 strengths in the grids; the pilot used 20 random directions. The brief's
  "≥ 100 random directions" requirement is **not met**, and the 5-point effect cannot be
  resolved with 80 test traces. The 20 random directions are the start of the 100 and
  can be extended without recomputation (`baseline_b steer --force --n-random 100`).
- **One run, one seed, one model.** No repeated runs, and the Qwen secondary run did not happen.
- **Pilot sample.** The pilot has 15 test-template failures (10 templates), 22 items in the
  repair test, and 2 random directions in the exploratory diagnostic.

- **Model:** a single 2B model (Gemma-2-2B-IT); the Qwen2.5-1.5B secondary run
  is reported separately if it is run. Results may not transfer to larger
  models. The CoT prompt format was chosen with Qwen-0.5B (D9) and not
  re-tuned for Gemma.
- **Sample size:** about 97 templates × 8 instances. The test set has about 29
  templates, so template-clustered CIs will be wide. Pilot repair n is small.
- **Labels are heuristic:** step labels, error types and the "content
  attributable" classification come from our parser and labeller.
  Free-form model steps are only partly parseable; the parse rate is reported.
  They need human verification (below).
- **Distractors:** written by the AI assistant, not GSM-NoOp's originals, and
  not yet human-reviewed.
- **Deviations from the papers:** listed per baseline above and in DECISIONS D5.
  Highlights: emission-position probing (A), GSM-Symbolic instead of MATH (B),
  no MLP/circular probes (A), additive steering added (B).
- **Self-sourced "bound" value:** read from the model's own original step
  (replay), not from a second probe. This is a simplification of the deployable
  method.
- **Prompt selection leak:** the prompt format was chosen on 16 problems from
  all templates using the 0.5B model, by marker adherence only (D9).

## 6. Must be done by hand

1. **Manual verification of about 20% of parsed steps and failure labels by
   two annotators.**
   - Each annotator labels independently: (operation, operands, result)
     correctness, `error_type`, `distractor_used`, and "content-attributable".
   - Sample with seed 42 from `results/traces/steps.jsonl`, stratified by split
     and variant.
   - Report Cohen's κ per field; if κ < 0.6, revise the labeller before citing
     Phase-3 results.
2. Review all 97 NoOp clauses (`rfd.data.distractor.NOOP_CLAUSES`) for being
   truly inconsequential.
3. Fill the "manual verdict" lines in `results/traces/parse_sample.md` (30
   steps).
4. Run the GPU jobs and write sections 1-4 against the generated tables.
