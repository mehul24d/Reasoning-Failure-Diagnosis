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
