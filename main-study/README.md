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

## Status (2026-11-13)

| Step | Milestone | What | Status |
|---|---|---|---|
| S1 | M0 | Equivalence checks, SAE validation, E0 sparsity check | **Done for Gemma-2-2B-IT.** 9B and Qwen-7B pending a GPU |
| S2 | M1 | Full-scale generation on Gemma-2-9B-IT; step parsing | Parser v4 done and frozen; `prereg/S2.md` drafted; run pending |
| S3 | M1 | Qwen2.5-7B; sampled subset (T = 0.7, k = 5); gold extraction | Not started |
| S4 | M1 | Activation caching | Not started |
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
7. **Pre-registrations.** `prereg/E0.md` is committed. `prereg/S2.md` is a **draft**: it covers
   the population, 8 instances per template, a 40/20/40 template split with seed 0,
   max_new_tokens 1024, parser v4 frozen with lenient s0 primary, and acceptance criteria.
8. **Colab notebook for the 9B proof of concept:** `notebooks/colab_9b_poc.ipynb`. It runs
   S1b, 1,000 traces with a speed log, parser statistics, SAE validation and E0 on one A100.
9. **Tests:** 108 pass and 2 are skipped (`pytest`). They are offline and use a tiny random model.

## What is left

1. **S1 on Gemma-2-9B-IT and Qwen2.5-7B.** This needs one GPU with at least 40 GB, such as an
   A100. The Colab notebook is ready.
2. **Group sign-off on `prereg/S2.md`:**
   - the population: the Qu & Gomes no-op data, or the fallback of Apple GSM-Symbolic `main`
     with natural failures;
   - the prompt paraphrases 1 and 2 in `harness/prompts.py`;
   - the GPU.
3. **S2–S4:** about 4,800 greedy traces per model, parsing with a fresh 40-step hand check, the
   sampled subset, gold extraction and activation caching.
4. **S5–S11:** E1–E4 as in the plan, with gates G1 and G2. E4 needs two annotators on 20% of the
   failures.

**Compute still needed (estimate):** about 50–70 A100 hours, 64 GB RAM and 300–500 GB of disk.

### Deviations from the plan so far

All deviations are dated in the pre-registrations.

- Python 3.13 instead of 3.11.
- Kaggle 2 × T4 for the 2B model instead of one large GPU.
- 131k SAEs for 9B, because 65k exists only at one 9B layer.
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
| `probes/`, `cache/` | Placeholders for S4–S6 |
| `configs/` | Models, SAEs, hook sites, every threshold (plan Appendix E) |
| `prereg/` | Pre-registrations (`E0.md` committed, `S2.md` draft) |
| `scripts/` | `s1_equivalence.sh`, `s1_sae.sh`, `validate_parsing.py` |
| `notebooks/` | Kaggle notebooks used for M0, and the Colab 9B notebook |
| `reports/m0/` | Every M0 result, run log and hand check |
| `docs/STEPS.md` | Step tracker |

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # 3.13: see requirements.txt
pip install -r requirements.txt
pytest                                                    # offline, no downloads
# GPU (Gemma-2 is gated: accept the licence, set HF_TOKEN in the environment, never in a file)
bash scripts/s1_equivalence.sh gemma-2-9b-it cuda
bash scripts/s1_sae.sh gemma-2-9b-it
```
