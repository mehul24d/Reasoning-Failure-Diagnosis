# Work plan status (plan Table 21)

Each step starts only when the steps it depends on are complete and its pre-registration is
committed. Status: [x] done · [~] in progress · [ ] not started.

| Step | Milestone | What | Status |
|---|---|---|---|
| 0 | – | Repository layout, configs, environment, tests | [x] |
| S1a | M0 | Generation harness: loading, prompts, pure-greedy generation, resume | [x] code · [ ] server run |
| S1b | M0 | Six TL–HF equivalence checks (Table 5) | [x] code · [ ] server run per model |
| S1c | M0 | SAE validation (FVU ≤ 0.40, ΔCE ≤ 0.20) and E0 sparsity check; pre-register E0 | [x] code · [ ] commit `prereg/E0.md` · [ ] server run |
| S2 | M1 | Full-scale generation on Gemma-2-9B-IT; step parsing | [~] parser v4 frozen (`e559c46`); `prereg/S2.md` DRAFT (needs group sign-off: population, paraphrases, GPU); `scripts/validate_parsing.py --problems` for GSM-Symbolic traces · [ ] run (pending GPU) |
| S3 | M1 | Qwen2.5-7B; sampled subset; gold extraction and alignment; pre-register E1–E2 | [ ] |
| S4 | M1 | Activation caching | [ ] |
| S5 | M2 | E1: directions and probes, 5 seeds, shuffled nulls | [ ] |
| S6 | M2 | E2: readout map, baselines, RQ2 paired test | [ ] |
| G1 | M2 | Gate: RQ1 and RQ2 criteria | [ ] |
| S7 | M3 | Pre-register E3; gold-labelled content probes | [ ] |
| S8 | M3 | E3 on test templates (9B, then Qwen and 2B) | [ ] |
| G2 | M3 | Gate: RQ3 criterion | [ ] |
| S9 | M4 | Pre-register E4; attribution labels; 20% two-annotator check | [ ] |
| S10 | M4 | Typology features; RQ4 classifier | [ ] |
| S11 | M4 | Report and release | [ ] |

## Open items for the group

* Review the DRAFT prompt paraphrases 1 and 2 in `harness/prompts.py` before M0.
* ~~Confirm the Gemma Scope SAE widths and layers~~: checked against the SAELens registry on 2026-11-11. The 9B check uses 131k SAEs, not 65k; see the deviation in `prereg/E0.md`.
* Review `prereg/E0.md`, fill in the date and commit it before running `scripts/s1_sae.sh`.
* `max_new_tokens`: the S2 draft proposes 1024 (512 stays for E0, including the 9B E0 run).
* Decide the S2 population: Qu & Gomes no-op data if it arrives, else Apple GSM-Symbolic `main` with natural failures (`prereg/S2.md`).
