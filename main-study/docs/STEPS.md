# Work plan status (plan Table 21)

Each step starts only when the steps it depends on are complete and its pre-registration is
committed. Status: [x] done · [~] in progress · [ ] not started.

**Scope (2026-11-13, `prereg/S2.md` A1):** the minor project (S1–S11) runs on Gemma-2-2B-IT
only. Gemma-2-9B-IT and Qwen2.5-7B-Instruct move to Semester 8 (RQ8). There, S1–S4 are repeated
for them and the RQ3 gate G2 is re-checked on 9B first.

| Step | Milestone | What | Status |
|---|---|---|---|
| 0 | – | Repository layout, configs, environment, tests | [x] |
| S1a | M0 | Generation harness: loading, prompts, pure-greedy generation, resume | [x] (2B, Kaggle) |
| S1b | M0 | Six TL–HF equivalence checks (Table 5) | [x] 2B: 6/6 pass · 9B, Qwen: Semester 8 |
| S1c | M0 | SAE validation (FVU ≤ 0.40, ΔCE ≤ 0.20) and E0 sparsity check; pre-register E0 | [x] 2B: 24/26 layers, E0 done · 9B: Semester 8 |
| S2 | M1 | Full-scale generation on Gemma-2-2B-IT (batched HF, equivalence gate); step parsing | [~] parser v4 frozen (`e559c46`); pipeline `harness/pipeline.py` built and tested; `prereg/S2.md` DRAFT amended 2026-11-13 (needs sign-off: population, paraphrases, templates) · [~] `make estimate` · [ ] run |
| S3 | M1 | Sampled subset (test failures); gold extraction and alignment; pre-register E1–E2 | [~] sampled stage and gold s0 built into the pipeline · [ ] run · [ ] E1–E2 prereg |
| S4 | M1 | Activation caching (residuals, float16, pre-registered positions) | [~] built into the verified TL pass (`cache/`) · [ ] run |
| S5 | M2 | E1: directions and probes, 5 seeds, shuffled nulls | [ ] |
| S6 | M2 | E2: readout map, baselines, RQ2 paired test | [ ] |
| G1 | M2 | Gate: RQ1 and RQ2 criteria | [ ] |
| S7 | M3 | Pre-register E3; gold-labelled content probes | [ ] |
| S8 | M3 | E3 on test templates (2B; 9B and Qwen in Semester 8) | [ ] |
| G2 | M3 | Gate: RQ3 criterion | [ ] |
| S9 | M4 | Pre-register E4; attribution labels; 20% two-annotator check | [ ] |
| S10 | M4 | Typology features; RQ4 classifier | [ ] |
| S11 | M4 | Report and release | [ ] |

## Open items for the group

* Review the DRAFT prompt paraphrases 1 and 2 in `harness/prompts.py` before M0.
* ~~Confirm the Gemma Scope SAE widths and layers~~: checked against the SAELens registry on 2026-11-11. The 9B check uses 131k SAEs, not 65k; see the deviation in `prereg/E0.md`.
* Review `prereg/E0.md`, fill in the date and commit it before running `scripts/s1_sae.sh`.
* ~~`max_new_tokens` 1024~~: withdrawn. 512 with a stop at the answer line (`prereg/S2.md` A3).
* Which 97 templates, and whether test is drawn from templates with gold programs (`prereg/S2.md` open item 4).
* Decide the S2 population: Qu & Gomes no-op data if it arrives, else Apple GSM-Symbolic `main` with natural failures (`prereg/S2.md`).
