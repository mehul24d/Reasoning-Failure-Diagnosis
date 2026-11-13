# Main study: Diagnosing Reasoning Failures — Content vs. Evaluation Variables

DTU CSE minor project (BTP), Semester VII 2026–27, group GP099.

This folder implements the **Detailed Research Plan** (`report (1).pdf`) from scratch, step by
step, with a pre-registration committed before each experiment. Section and table numbers in
the code refer to that plan. The baseline and feasibility work at the top of this repository
came first and motivated the design; this folder is the full study.

**Main question (RQ3).** When a model gets a maths word problem wrong, is the correct
intermediate value (operation, operand, result) still decodable from its activations? If yes,
a failure could be repaired from the inside ("self-sourced repair"); if no, that is reported as
a negative result.

**Scope (amended 2026-11-13).** The minor project (RQ1–RQ4, S1–S11) runs on **Gemma-2-2B-IT
only**, on free Kaggle T4s, and its conclusions apply to 2B only. Gemma-2-9B-IT and
Qwen2.5-7B-Instruct move to the Semester 8 major project under the RQ8 replication. There, the
RQ3 feasibility gate is re-checked on 9B before any 9B repair experiment. The research questions,
the design and every threshold are unchanged. See `docs/PLAN_AMENDMENT_2026-11-13.md` and
`prereg/S2.md` (A1–A7).

## Status (2026-11-13)

| Step | Milestone | What | Status |
|---|---|---|---|
| S1 | M0 | Equivalence checks, SAE validation, E0 sparsity check | **Done (Gemma-2-2B-IT).** 9B and Qwen-7B: Semester 8 |
| S2 | M1 | Full-scale generation on Gemma-2-2B-IT; equivalence gate; step parsing | Pipeline built and tested; parser v4 frozen; `prereg/S2.md` amended, still a draft; run pending |
| S3 | M1 | Sampled subset (T = 0.7, k = 5, test failures); gold extraction | Built into the same pipeline; run pending |
| S4 | M1 | Activation caching (residuals, float16, pre-registered positions) | Built into the verified TL pass; run pending |
| S5–S6 | M2 | E1 directions and probes; E2 readout map; gate G1 | Not started |
| S7–S8 | M3 | E3 self-sourced feasibility (RQ3); gate G2 | Not started |
| S9–S11 | M4 | E4 attribution and typology (RQ4); report | Not started |

The detailed tracker is `docs/STEPS.md`.

## What has been done

All M0 runs used Gemma-2-2B-IT in float32 on Kaggle (2 × T4), on 500 GSM8K **train** problems,
so the GSM-Symbolic study problems stay untouched.

1. **S1b — equivalence checks: 6/6 pass.** The TransformerLens copy matches Hugging Face:
   residual relative error 1.5e-6, top-1 agreement 344/344 tokens, padding error 9.4e-7,
   teacher-forced reproduction 394/394, pure greedy config, exactly one BOS per prompt.
   `reports/m0/equivalence_gemma-2-2b-it.json`
2. **S1c — 1,000 greedy traces** (marked and terse prompts): accuracy 71.2% (marked 68.2%,
   terse 74.2%), no answer 1.1%, hit the 512-token limit 0.4%. The trace file's sha256 is in
   `reports/m0/m0_traces_provenance.txt`. The trace file itself is not in git.
3. **SAE validation (Gemma Scope 65k): 24/26 layers pass** (FVU ≤ 0.40 and ΔCE ≤ 0.20).
   Layers 11 (ΔCE 0.234) and 25 (ΔCE 0.233) fail and are excluded from SAE analyses.
   `reports/m0/sae_validation_gemma-2-2b-it.json`
4. **E0 — no sparsity collapse.** With matched tokens, the early and middle bands show no
   difference. The late band differs by ΔL0 = −1.92 [−2.61, −1.32], d = −0.03 (Holm p < 1e-4),
   which is negligible. An independent re-check passed 84/85 checks; the one miss was a
   plausibility range guessed too narrow. `reports/m0/e0_gemma-2-2b-it.json`, `verify_e0.txt`
5. **Step parser (plan Algorithm 1) — v4.** It splits traces into steps, reads equations,
   aligns them to the gold program (tiers 1–5) and labels planning, binding, arithmetic or none,
   giving the first error step s0 (lenient and strict).
   - Hand checks of 40 random steps went from 21/40 (v1) to 28, 33 and **39/40 (v4)**.
   - False error steps on correct traces fell from about 31% to **4.5–5.2%**. Errors are found
     in 81–92% of failed traces.
   - Known limits: other unit systems (cents vs dollars), coincidental value matches, algebra.
   - `reports/m0/parsing_manual_check_v4.md`, `parsing_validation_v4.json`
6. **First observation (2B, GSM8K train, tool-building only).** The first error in failed
   traces is planning in 42%, binding in 39%, arithmetic in 11%, and none is found in 8%.
7. **Pre-registrations.** `prereg/E0.md` is committed. `prereg/S2.md` was **signed off on
   2026-11-13** (A8–A13; amended A14–A17). It becomes binding with the commit made immediately before the first
   study data. It covers:
   - the population (A14): primary = GSM-Symbolic distractor failures (right without the
     no-op clause, wrong with it), secondary = natural failures;
   - **89 templates** (the ones with gold programs) × 8 instances × 6 prompts × 2 conditions =
     8,544 greedy traces, with a 33/16/40 template split, seed 0, committed;
   - the frozen paraphrases;
   - the design effect for the target on the primary paired outcomes, estimated on train + dev
     templates before any primary analysis, with a pre-specified top-up rule (A15); the pilot's
     failure-occurrence deff (19.76) is descriptive;
   - template clustering and effective n stated with every result (A16);
   - the automatic over-budget rule;
   - and, from the draft, max_new_tokens 512 with a stop at the
   answer line, parser v4 frozen with lenient s0 primary, the batched-generation equivalence
   gate, the residual cache and the acceptance criteria.
8. **S2–S4 pipeline for Kaggle T4s** (`harness/pipeline.py`, `harness/batched.py`, `cache/`).
   - Stages: problems → generate → score → verify → cache → sampled → report.
   - Every stage is resumable, uses both T4s, and stops cleanly before the session deadline.
   - Driver: `notebooks/run_on_kaggle.ipynb` (repository root), run as Save & Run All commits.
   - `make estimate` projects the GPU-hours from a timed smoke test.
9. **Colab notebook for the 9B proof of concept:** `notebooks/colab_9b_poc.ipynb`, kept for
   Semester 8.
10. **Tests:** 146 pass and 2 are skipped (`pytest`). They are offline and use a tiny random
    model, including an end-to-end run of the pipeline with two workers.

## What is left

1. **The reviewers' answers for the 89 distractor clauses** (`docs/review/`, D34, D39), which
   now include template 50's new clause. Then `scripts/review_agreement.py` computes kappa and
   the agreed rewrites are re-checked. Study generation waits for it.
2. **S2–S4 on Gemma-2-2B-IT:** 8,544 greedy traces (re-projected: 13.0 T4-hours with the
   safety factor), the equivalence gate, the residual cache,
   the sampled subset, and parsing with a fresh 40-step hand check.
3. **S5–S11:** E1–E4 as in the plan, with gates G1 and G2. E4 needs two annotators on 20% of the
   failures.
4. **Semester 8:** S1–S4 on Gemma-2-9B-IT and Qwen2.5-7B-Instruct, then the RQ3 gate re-check
   on 9B.

**Compute for S2–S4 (minor project):** free Kaggle T4s, budget 25 T4 GPU-hours. The measured
projection is in `reports/m1/estimate_gemma-2-2b-it.md` and `docs/PLAN_AMENDMENT_2026-11-13.md`.

### Deviations from the plan so far

All deviations are dated in the pre-registrations.

- Minor project on Gemma-2-2B-IT only; 9B and Qwen-7B move to Semester 8 (RQ8). The RQ3 gate is
  re-checked on 9B there (2026-11-13, A1).
- Batched Hugging Face generation with a TL equivalence gate, instead of batch size 1 (A2).
- Stop at the answer line, with max_new_tokens 512 (A3).
- Residual cache in float16 storage at the pre-registered positions; SAE features computed on
  CPU (A4).
- Sampled runs only on test failures (A5).
- 89 templates with gold programs, split 33/16/40 (A8, replacing A6's 97).
- Design effect estimated on the pilot (Kish, unequal clusters): N* = 4,209 (A11), descriptive
  since A15. The target uses the primary-outcome deff.
- Primary population: distractor failures with the study's own reviewed clauses, both
  conditions per instance (A14, A17).
- Over budget, a fixed rule applies automatically: fewer instances down to the power floor,
  then no sampled runs (A13).
- Kaggle's copy of the Gemma weights, with Phase 0 re-run on it (A12).
- Python 3.13 instead of 3.11.
- Kaggle 2 × T4 instead of one large GPU.
- 131k SAEs for 9B (Semester 8), because 65k exists only at one 9B layer.
- Lenient s0 as the primary label, with strict s0 also reported.
- The Qu & Gomes no-op data is not public. If it does not arrive, the fallback loses the
  matched distractor pairs.
- The S2 draft covers GSM-Symbolic only; the natural GSM8K and MATH-500 failure sets are still
  to be added.

## Layout

| Folder | Contents |
|---|---|
| `harness/` | Model loading, equivalence checks, prompts, greedy and sampled generation, scoring, datasets, template splits |
| `parsing/` | Answer extraction, equation reader, gold programs, step parser and alignment (v4) |
| `analysis/` | SAE validation, E0 sparsity check, statistics (clustered bootstrap, Holm) |
| `cache/` | Positions, the verified TL pass with residual caching, and the store reader |
| `probes/` | Placeholder for S5–S6 |
| `configs/` | Models, SAEs, hook sites, every threshold (plan Appendix E) |
| `prereg/` | Pre-registrations (`E0.md` committed, `S2.md` draft) |
| `scripts/` | `s1_equivalence.sh`, `s1_sae.sh`, `validate_parsing.py` |
| `notebooks/` | Kaggle notebooks used for M0, and the Colab 9B notebook (the S2–S4 driver is `../notebooks/run_on_kaggle.ipynb`) |
| `reports/m0/` | Every M0 result, run log and hand check |
| `reports/m1/` | S2–S4 reports, including the compute estimate |
| `docs/STEPS.md` | Step tracker |

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # 3.13: see requirements.txt
pip install -r requirements.txt
pytest                                                    # offline, no downloads
# GPU (Gemma-2 is gated: accept the licence, set HF_TOKEN in the environment, never in a file)
# From the repository root, on Kaggle T4s (or run notebooks/run_on_kaggle.ipynb):
make estimate           # timed smoke test -> reports/m1/estimate_gemma-2-2b-it.md
make ms-pipeline        # S2-S4, refused while prereg/S2.md is a draft
make ms-status
```
