# Progress

## 2026-11-13: Clause generator fix and pre-review rewrites (D40–D42, S2 A18–A20); human review of the final clauses pending

No GPU time used, and none of the 25-hour S2–S4 budget.

- **Bug (D40):**
  - Partitive clauses could contradict the instance.
  - GSM-Symbolic varies the entities between instances, so 57 of the 89 drafts failed an
    instance-level check on instances 1–16: 32 partitive, 48 naming an instance-specific word.
- **64 clauses rewritten before the human review:** the 20 flagged by the AI pre-screen, every
  instance-check failure, and 53, 70 and 96 from reading.
- **All checks pass on 89 × 16 instances.** For every rewrite, an independent solver gives the
  same answer with and without the clause (`main-study/reports/m1/distractor_checks_final.json`).
- **Human sheet ready:** `docs/review/human_review_final.csv`. After it is filled:
  `scripts/review_agreement.py` (kappa per question, agreed changes), then the review is
  recorded in S2 and the binding commit is made.
- **Temptation covariate (D41):** `main-study/data/distractor_temptation.json`. Final: 16
  PASS, 54 WEAK, 19 UNSURE; the drafts were 47 PASS.
- **Top-up (D42):** with instances 9–16 the projection is 20.71 T4-h, × 1.25 = 25.89 > 25. No
  top-up; reported as underpowered.

## 2026-11-13: Group decisions D35–D39 (S2 A14–A17); S2 still waiting for the clause review

No GPU time used, and none of the 25-hour S2–S4 budget.

**Recorded before any study data:**
- **Population (D35, A14):** distractor failures are primary (right without the clause, wrong
  with it), natural failures secondary. Both conditions are generated for every instance:
  89 × 8 × 6 × 2 = 8,544 greedy traces.
- **Design effect (D36, A15):** the target uses the deff of the primary paired outcomes,
  estimated on train and dev templates. The top-up to instances 9–16 is pre-specified and
  applied once, before any primary analysis. The failure-occurrence deff (19.76) is
  descriptive. No top-up now.
- **Clustering (D37, A16):** every result states n, templates, ICC, deff and n_eff.
- **Weights (D38):** the authenticated Hugging Face comparison is ready but not run, because
  no HF_TOKEN is available here.

**Re-projection** from the measured throughputs (`main-study/reports/m1/estimate_gemma-2-2b-it_reprojected.md`):

| Stage | T4 GPU-hours |
|---|---|
| Generation, 8,544 traces | 4.60 |
| Verification + cache | 1.03 |
| Sampled runs, 7,610 | 4.79 |
| **Total** | **10.43** (× 1.25 = **13.04**) |

- Within budget, so the budget rule does not apply. Disk: 13.2 GB.
- If the equivalence gate fails, TL generation brings the total to 33.9 even after the rule.
  The run would then stop for a group decision.

**Pilot, for the primary population:** 66 of 352 pairs are distractor failures (0.1875), with
ICC 0.432 and a descriptive deff of 9.97. At that rate, 8 instances give about 292 primary test
failures with s0, against the 320 floor. At the planning rate the figure is about 429.

**Template 50** had no clause, and it is a test template, so a clause was drafted: "The room has
{d} windows." It passes all automated checks (8/8 instances) and was added to
`docs/review/distractor_clauses.csv`. The other 88 rows are byte-identical to the version sent
for review.

**Clause review: not found.** The reviewer columns of `docs/review/distractor_clauses.csv` are
empty. No other sheet is in `docs/review/`, on `origin/main` or on `origin/neil`. Kappa, the
rewrites and the S2 binding commit wait for the sheets. `scripts/review_agreement.py` is ready,
and the `problems` stage refuses to run until it has produced
`main-study/data/distractor_clauses_reviewed.json`.

## 2026-11-13: Phase 0 on the Kaggle weights and the measured compute estimate (Gemma-2-2B-IT)

Kaggle kernel `btp-minor-estimate` v3, code `b920121`, 2 × Tesla T4, 0.38 h of session time.
Decisions D28–D34 (S2 A8–A13) were committed before it ran.

**Phase 0 on Kaggle's copy of the weights** (`google/gemma-2/transformers/gemma-2-2b-it/2`):
**all 7 checks pass**. The report is `main-study/reports/m1/phase0_kaggle_gemma-2-2b-it.json`.

| Check | Value | Threshold |
|---|---|---|
| TL vs HF residual stream, worst layer (relative) | 1.50e-6 (L25 after ln_final) | < 0.01 |
| TL vs HF top-1 logit agreement | 1.000 (344 tokens) | ≥ 0.99 |
| Left-padded batch vs single, last-token logits | 9.4e-7 | < 0.01 |
| Teacher-forced greedy reproduction | 1.000 (394 tokens) | ≥ 0.99 |
| Pure-greedy generation config | true | true |
| One BOS per prompt | 1, 1, 1, 1 | 1 |
| Second greedy run identical (added, A12) | 4 / 4 | all |

The first four values equal those of the M0 run on the Hugging Face download to every printed
digit. Weight provenance: Kaggle Models `google/gemma-2/transformers/gemma-2-2b-it`, **version 2**. The sha256 of all 11 files is in `main-study/reports/m1/weights_gemma-2-2b-it.json`: the two weight shards are `532d792c9178…` and `6d6d9ce84db3…`, and their combined hash is `bdad5059211d…`. Compared with the Hugging Face repository (revision `299a856`):
- 7 non-LFS files are byte-identical by git blob hash: config, generation config, weight index, tokenizer config, special tokens, README and `.gitattributes`.
- The 4 LFS files (the two weight shards, `tokenizer.json` and `tokenizer.model`) have exactly the published sizes.
- Their sha256 could not be compared. Hugging Face hides the LFS hashes of a gated model from requests without a token (tried from the Mac and from Kaggle).

Together with the Phase 0 values matching M0 to every digit, this is strong but indirect evidence that the weights are the same. A direct check needs one authenticated look at the HF file list; a group member with Gemma access can do it, and no token goes into the repository.

**Measured estimate** (`main-study/reports/m1/estimate_gemma-2-2b-it.md`). The smoke test ran
384 greedy traces on GSM8K train, through the real pipeline, on 2 × T4. It projects to the
pre-registered sizes: 89 × 8 × 6 = 4,272 greedy traces, and 4,965 sampled runs (40 × 8 × 6 test
traces × 0.517 pilot failure rate × 5).

| Stage | Measured on T4 | Projected T4 GPU-hours |
|---|---|---|
| Greedy generation (HF batched, batch 16) | 71 new tokens/s per GPU, mean 138 tokens | 2.31 |
| Verification + residual cache (one TL pass) | 1.6 ms per sequence token, 0.39 s per trace | 0.58 |
| Sampled runs (T = 0.7, k = 5, test failures) | 69 new tokens/s per GPU, mean 155 tokens | 3.13 |
| **Total** (model loads included) | | **6.02** |
| **Total × 1.25 safety** | | **7.52** (budget 25) |

- **Within budget**, so the A13 rule did not apply: 8 instances and the sampled runs are kept.
- Kaggle quota: about 3.0 h of session time, in 1 session.
- Disk: 6.51 GB of float16 residual shards, 12.2 cached positions per trace; **6.66 GB** in total.
- **TL-generation fallback** (only if the equivalence gate fails): 12.5 tokens/s, so 13.1
  GPU-hours for generation and a total of 16.8 (21.0 with the safety factor), still within
  budget.
- **The proposal's top-up** to instances 9–16 (D31) would roughly double the total, to about
  12 GPU-hours (15 with the safety factor), also within budget.
- Smoke checks (not results):
  - 2 of 384 traces hit the 512 cap (0.5%); the other stops were 377 at the answer line and 5 at end of turn;
  - TL reproduced the HF tokens exactly (1.0000, max logit difference 3.0e-4);
  - float16 storage max relative error 4.1e-4 (max absolute 0.50, on values up to 1,353).
- The interim projection of the earlier entry (4.7 GPU-hours) is replaced. Generation is slower
  than assumed (71 tokens/s, not 100), the TL pass is faster, and the sampled runs are sized
  at the pilot failure rate (51.7%) instead of 29%.

**GPU time used so far** on the minor project: about 0.5 h of Kaggle quota, made up of this run
(0.38 h on 2 × T4, so 0.76 T4 GPU-hours) and the two earlier runs that stopped at setup. S2–S4
has used none of its 25 T4 GPU-hours.

**Waiting for people:**
1. The human review of the 88 distractor clauses (`docs/review/`, about 1.5 h per reviewer).
2. A group decision on the design-effect shortfall (D31).

S2 generation does not start before both.

# Progress Report — Pilot Session

> **Archived (2026-10-21).** This report describes the SAE
> phase-segmentation direction, which is now deprecated after the project
> pivoted to *Self-Sourced Repair*. The code lives in `src/rfd/legacy/`
> (module paths below are pre-move: `rfd.X` is now `rfd.legacy.X`), and
> the outputs in `results/legacy_pilot/`. Current work and results:
> `docs/BASELINE_REPORT.md`; decisions: `docs/DECISIONS.md`. Note: the
> "greedy" pilot below actually ran with the checkpoint's repetition_penalty
> = 1.1 (found 2026-10-22, DECISIONS D12).

**Date:** 2026-10-12
**Pilot config:** Qwen/Qwen2.5-0.5B-Instruct, GSM8K test[0:50], greedy decoding.

This document reports only what the code in this repo actually produced
in this session. No numbers here are estimated or fabricated. If a stage
below has no numbers next to it, that stage was not run this session —
see the TODO list.

## What is done (implemented AND executed this session)

- [x] Repo skeleton, pyproject, configs, `.gitignore`, pre-commit config.
- [x] GSM8K loader wired end-to-end (fixed index-sliced splits, seed 42).
- [x] Trace generation harness (greedy, batch size 1, chat-templated
      prompts) — run results are in "Pilot run results" below.
- [x] Answer extraction (`Final Answer:` marker + fallback) and numeric
      exact-match scoring.
- [x] TransformerLens activation extraction — confirmed working against
      `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, d_model=896,
      `hook_resid_post`). Layer used for the pilot: middle layer = 12.
- [x] JumpReLU SAE (`src/rfd/sae/jumprelu_sae.py`), trained at **pilot
      scale** (d_sae=512, 200 steps) on a 20-problem GSM8K *train*-split
      slice, kept disjoint from the 50-problem pilot *test* traces.
- [x] Coactivation graph -> Louvain community detection -> changepoint
      detection (ruptures Pelt) -> positional 4-phase labeling.
- [x] `boundary_f1`, `temporal_iou`, `phase_recovery_rate` implemented
      and unit-tested on synthetic data (not yet run against real manual
      annotations — see TODO).
- [x] Heuristic failure classifier + 20%-manual-check CSV generator.
- [x] 37 pytest unit tests, all passing, all on synthetic/local data.

## What is stubbed / simplified (clearly marked TODO in code)

- **SAE scale**: `configs/sae.yaml:full_scale_training` specifies the
  Semester-7 target (JumpReLU, D=65,536, target L0≈64, trained on the
  full GSM8K train split). This session only ran `pilot_training`
  (D=512, 200 steps, 20-problem train slice) — a pipeline-validation
  artifact, not a publication-quality dictionary. **Not fabricated as
  "the real SAE" anywhere in code or docs.**
- **Pretrained SAE check**: we could not confirm a publicly released
  SAELens-compatible pretrained SAE for Qwen2.5-0.5B-Instruct this
  session (SAELens' public catalogue is mostly gpt2-small/pythia/gemma-2).
  Verifying this properly is TODO for Semester 7.
- **JumpReLU implementation**: rectangle-STE threshold, no decoder
  unit-norm constraint, no dead-feature resampling. Functionally
  correct but simplified relative to the full DeepMind recipe.
- **Phase labeling**: segments are labeled by *position* (first ->
  problem_understanding, last -> self_verification, middle split
  between planning/execution), not by feature interpretation. This is
  the single biggest methodological gap relative to the synopsis's
  framing of SAE-driven segmentation, and the first thing to fix in
  Semester 7.
- **Failure classifier**: regex/heuristic only (arithmetic-equation
  consistency check + phase-presence from segmentation). Not validated
  against manual labels yet — that's exactly why `csv_template.py`
  flags 20% of rows for a human check.
- **Sparsity-collapse-delta**: first-half-mean-L0 minus second-half-mean-L0
  per trace. A proxy, not a validated collapse detector.
- **MATH / BBH-Logic loaders**: implemented
  (`src/rfd/data/loader.py`) and config-driven, but not exercised beyond
  a design pass this session — the pilot only runs GSM8K.

## What is explicitly NOT done (requires a human or more compute/time)

- [ ] Manual annotation of 20 traces (`rfd.segmentation.annotate_cli` is
      ready to use; requires a human annotator sitting down with it).
      Without this, `boundary_f1`/`temporal_iou`/`phase_recovery_rate`
      have **no gold labels to run against yet** for the pilot traces.
- [ ] 20% manual verification of the taxonomy CSV (same reason).
- [ ] Full-scale SAE training (D=65536, full train split).
- [ ] MATH and BBH-Logic full pipeline runs.
- [ ] 3-model comparison (only Qwen2.5-0.5B-Instruct used this session).
- [ ] Leiden community detection (code path exists, raises
      `NotImplementedError` — `leidenalg`/`igraph` not installed).

## Pilot run results (actual, from this session — reproduce with `bash scripts/run_pilot.sh`)

- **Exact-match accuracy:** 22/50 (44%) on GSM8K test[0:50], Qwen2.5-0.5B-Instruct,
  greedy decoding. Traces were spot-checked manually and read as coherent,
  step-by-step reasoning (not garbled) — see `results/pilot/traces.jsonl`.
- **SAE training** (pilot scale: d_sae=512, 200 steps, 5,796 tokens from a
  20-problem GSM8K *train* slice disjoint from the pilot test set):
  MSE 0.402 -> 0.085, mean L0 243.5 -> 47.8 over training. The SAE
  converges and sparsifies as expected for a toy run.
- **SAE features on the 50 pilot traces:** mean L0 across traces = 45.0
  (out of d_sae=512), i.e. roughly 9% of pilot-SAE features active per
  token on average.
- **Segmentation — a real bug we found and fixed in this session:** the
  first run of the changepoint stage produced **1.00 average segments
  per trace** (effectively no boundaries detected at all). Root cause:
  `configs/segmentation.yaml` used a fixed absolute Pelt penalty (3.0)
  that was far too large for the actual scale of mean-aggregated SAE
  feature activations (series mean ~0.11, std ~0.13) — the l2-cost
  benefit of any split never exceeded that penalty. We replaced it with
  an adaptive, BIC-style penalty (`penalty_scale * var(series) * n_dims *
  log(n_tokens)`, see `src/rfd/segmentation/changepoint.py`), which is
  the standard approach for Pelt when a signal's scale isn't known in
  advance. `penalty_scale=1.5` was picked once, uniformly, by sweeping a
  few values against the *shape* of the resulting boundary-count
  distribution across all 50 traces (not by cherry-picking per trace) —
  see the sweep in this session's transcript. After the fix: **3.82
  average segments per trace** (range 1-10), which is in the right
  ballpark for a 4-phase model. **This has not been validated against
  manual annotations** — it is a sanity-check fix (stopped the pipeline
  from being degenerate), not evidence the phases are *correct*.
- **Failure taxonomy** (heuristic, unvalidated — 20% manual-check CSV at
  `results/pilot/taxonomy.csv` is generated but not yet human-reviewed):

  | error_type | count | mean L0 | mean entropy | mean sparsity-collapse delta |
  |---|---|---|---|---|
  | correct | 22 | 44.6 | 3.456 | **+0.69** |
  | execution_error | 15 | 45.9 | 3.472 | -1.49 |
  | planning_error | 9 | 44.6 | 3.456 | -0.18 |
  | arithmetic_error | 4 | 44.7 | 3.461 | -0.63 |
  | parsing_error | 0 | — | — | — |
  | verification_error | 0 | — | — | — |

  Two things to flag honestly about this table:
  1. The sparsity-collapse-delta sign flips between correct (positive:
     activity *drops* in the second half) and all three error types
     (negative: activity *rises* in the second half). With n=4-22 per
     group and no significance testing, **this is a pattern worth
     following up, not a result** — it could easily be an artifact of
     trace length differences between correct/incorrect answers.
  2. `verification_error` and `parsing_error` both have **zero** traces,
     and that is very likely a methodology artifact, not a real finding:
     the positional phase-labeling heuristic (`label_segments` in
     `phase_segmenter.py`) assigns `self_verification` to the *last*
     segment of almost every trace with >=2 segments regardless of what
     that segment's text actually says, and most traces had >=3
     segments (`operation_planning` likewise gets assigned whenever
     there are >=3 segments). This makes `classify_error`'s
     presence/absence check for those two phases nearly useless until
     phase labeling is based on content rather than position — see
     "Next steps" below, this is now the top-priority fix.

- **Mean feature entropy** is nearly identical across all groups
  (3.46-3.47), so in this pilot it carries no discriminative signal for
  failure type — worth noting rather than omitting.

## E1 / E2 mapping (per synopsis)

| Synopsis item | Status this session | Module |
|---|---|---|
| E1: pilot feasibility (1 model, GSM8K, small scale) | Pipeline implemented and run end-to-end | all of `src/rfd/` |
| E1: SAE activation extraction | Done (pilot scale) | `rfd.sae` |
| E1: phase segmentation, first version | Done (positional labeling heuristic) | `rfd.segmentation` |
| E1: failure taxonomy, first version | Done (heuristic classifier) | `rfd.taxonomy` |
| E2: full-scale SAE (D=65536, L0≈64) | **TODO** | `configs/sae.yaml:full_scale_training` |
| E2: 3-model comparison | **TODO** | — |
| E2: MATH / BBH-Logic | Loaders done, runs **TODO** | `rfd.data.loader` |
| E2: manual annotation + validated metrics | CLI/schema done, annotations **TODO** | `rfd.segmentation.annotate_cli` |
| E2: taxonomy ranking / validation | CSV template done, manual check **TODO** | `rfd.taxonomy.csv_template` |

## Next steps for Semester 7

1. Get 20 manual phase annotations and 20% manual taxonomy checks done,
   then re-run the metrics modules against real gold labels (currently
   only unit-tested on synthetic data).
2. Replace positional phase labeling with a feature-interpretation-based
   labeling scheme (e.g. inspecting top-activating tokens per community).
3. Scale SAE training to the full-scale config and verify/replace the
   pilot dictionary.
4. Extend to MATH and BBH-Logic, and to 2 additional models for the
   3-model comparison.
5. Validate the heuristic failure classifier against manual labels and
   report agreement (currently un-validated).
