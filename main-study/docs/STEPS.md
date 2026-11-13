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
| S2 | M1 | Full-scale generation on Gemma-2-2B-IT (batched HF, equivalence gate); step parsing | [~] parser v4 frozen (`e559c46`); pipeline `harness/pipeline.py` built and tested; `prereg/S2.md` signed off 2026-11-13 (A8–A13; binding at the first study data) · [x] Phase 0 on the Kaggle weights (7/7) · [x] `make estimate` (6.0 T4-h, 7.5 with safety) · [x] distractor auto-checks · [ ] human review of the clauses · [ ] run |
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

* ~~Review the DRAFT prompt paraphrases 1 and 2~~: frozen as drafted (`prereg/S2.md` A9).
* ~~Confirm the Gemma Scope SAE widths and layers~~: checked against the SAELens registry on 2026-11-11. The 9B check uses 131k SAEs, not 65k; see the deviation in `prereg/E0.md`.
* Review `prereg/E0.md`, fill in the date and commit it before running `scripts/s1_sae.sh`.
* ~~`max_new_tokens` 1024~~: withdrawn. 512 with a stop at the answer line (`prereg/S2.md` A3).
* ~~Which 97 templates~~: the 89 with gold programs, for train, dev and test (`prereg/S2.md` A8).
* ~~Decide the S2 population~~: distractor failures primary, natural failures secondary (A14,
  replacing A9).
* ~~Clause generator bug~~: instance-level partitive and grounding checks on instances 1–16; 64
  clauses rewritten before the review; all checks pass (A18, D40).
* Human review of the 89 FINAL clauses (`docs/review/human_review_final.csv`), then
  `scripts/review_agreement.py` (kappa per question, agreed changes), then the review is
  recorded in S2. Waiting for the filled sheet.
* ~~The design-effect shortfall~~: the primary-outcome deff is reported with the failure deff;
  the top-up to 16 instances is over budget (25.89 T4-h), so none, and the study is reported as
  underpowered (A20, D42).
