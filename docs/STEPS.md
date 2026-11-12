# Work plan status (plan Table 21)

Each step starts only when the steps it depends on are complete and its pre-registration is
committed. Status: [x] done · [~] in progress · [ ] not started.

| Step | Milestone | What | Status |
|---|---|---|---|
| 0 | – | Repository layout, configs, environment, tests | [x] |
| S1a | M0 | Generation harness: loading, prompts, pure-greedy generation, resume | [x] code · [x] server run gemma-2-2b-it (Kaggle 2×T4: 1,000 greedy GSM8K-train traces, acc 0.712, no-answer 0.011, max-tokens 0.004; `b1b6f85`) · [ ] gemma-2-9b-it, qwen2.5-7b-instruct (pending ≥24 GB GPU or 2-GPU HF split) |
| S1b | M0 | Six TL–HF equivalence checks (Table 5) | [x] code · [x] gemma-2-2b-it: all 6 pass (resid 1.50e-6, top-1 1.0, padded 9.4e-7, teacher-forced 1.0, greedy, BOS [1,1,1,1]; `b2feb8b`) · [ ] gemma-2-9b-it, qwen2.5-7b-instruct (pending GPU) |
| S1c | M0 | SAE validation (FVU ≤ 0.40, ΔCE ≤ 0.20) and E0 sparsity check; pre-register E0 | [x] code · [x] `prereg/E0.md` committed (`b2feb8b`) · [x] gemma-2-2b-it: SAE validation 24/26 layers pass, 11 and 25 fail on ΔCE (`4035602`); E0 early/middle no matched difference, late ΔL0 −1.92 [−2.61, −1.32], d −0.032, Holm p < 1e-4 (`e687124`; verified 84/85 + D8 resolved, `reports/m0/verify_e0.txt`) · [ ] gemma-2-9b-it (pending GPU) |
| S2 | M1 | Full-scale generation on Gemma-2-9B-IT; step parsing | [ ] |
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
* ~~Review `prereg/E0.md`, fill in the date and commit it~~: committed 2026-11-11 (`b2feb8b`) on neil's instruction; the group should still read it.
* Kaggle 2×T4 runs need `CUDA_LAUNCH_BLOCKING=1` for the TL copy split over both GPUs (CUDA error 719 otherwise); see `reports/m0/s1c_run_notes.txt`.
* Fix `max_new_tokens` (currently 512, provisional) in the S2 pre-registration.
