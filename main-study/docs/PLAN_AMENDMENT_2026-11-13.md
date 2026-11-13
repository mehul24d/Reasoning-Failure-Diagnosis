# Plan amendment, 2026-11-13: minor project on Gemma-2-2B-IT, free Kaggle T4s

**Amends:** the Detailed Research Plan (`report (1).pdf`), Sections 4, 6.1, 10, 12 and 14, and
the main-study README. **Made before any S2, S3 or S4 data exist.** Pre-registration amendments
are `prereg/S2.md` A1–A13 and the dated deviation in `prereg/E0.md`; decisions are
`docs/DECISIONS.md` (repository root) D21–D34. A8–A13 (D28–D34) are the group's decisions of
the same day, taken once the Gemma licence had been accepted on Kaggle.

**Why.** The college cannot provide the ~70 GPU-hours the original minor-project scope needed.
The target is zero cost: free Kaggle T4s (2 × 16 GB per session, about 30 GPU-hours a week).

**What does not change.** The research questions RQ1–RQ9, their hypotheses and falsification
criteria, the experimental design (E0–E14), every threshold, the number of random directions
and permutations, the clustered bootstrap and CI procedure, Holm correction, and the manual
annotation requirements. No quantised compute and no float16 compute for Gemma are introduced.
`tests/test_config_prompts.py` pins all of these.

The plan PDF and its source are not in the repository, so the replacement text is given here
section by section. Copy it into the next version of the plan.

---

## Section 4: Two-Stage Research Design (replacement rows)

| Stage | Description |
|---|---|
| **Pre-requisite (week 1 of Semester 7)** | SAE validation on instruct-model traces and the matched sparsity check (Appendix A), **on Gemma-2-2B-IT**. Done in M0: 24/26 layers pass, no sparsity collapse. |
| **Minor Project (Semester 7)** | Unchanged in content (failure sets, gold variables, content and evaluation directions, readout maps, self-sourced decodability, attribution labels), but **run on Gemma-2-2B-IT only**. Every minor-project conclusion is a conclusion about Gemma-2-2B-IT. |
| **Major Project (Semester 8)** | Unchanged, with one addition at the start. **Gemma-2-9B-IT and Qwen2.5-7B-Instruct enter here, under the RQ8 replication.** S1 (equivalence checks, SAE validation, E0) and the S2–S4 data production for them run first. **The RQ3 feasibility gate (G2) is then re-checked on Gemma-2-9B-IT before any 9B repair experiment**, with the same criterion and procedure. A 2B pass does not carry over to 9B. |

## Section 6.1: Common Infrastructure (replacement bullets)

- **Models.** Gemma-2 family. **Minor project: Gemma-2-2B-IT** (float32), for RQ1–RQ4.
  **Major project:** Gemma-2-9B-IT, the model on which Qu and Gomes' head result was
  replicated, and Qwen2.5-7B-Instruct for cross-family replication (RQ8). Qwen lacks public
  SAEs, so it is not used at the SAE level.
- **SAEs.** Gemma Scope JumpReLU residual-stream SAEs. The validation rule and thresholds are
  unchanged (FVU ≤ 0.40, ΔCE ≤ 0.20 nats per token). For 2B, 24/26 layers pass; layers 11 and
  25 are excluded. SAE activations are computed on CPU from the stored float32-computed
  residuals (stored in float16; the effect of float16 storage on SAE features is measured and
  reported).
- **Datasets and failure sets.** Population as in the proposal (A14): the primary population
  is **GSM-Symbolic distractor failures** (right without the no-op clause, wrong with it, same
  instance and prompt), using the study's own clauses after a two-person review, because the
  no-op data of Qu and Gomes are not public. Natural failures (clean condition) are secondary.
  Both conditions are generated for every instance.
  - **The 89 templates with a gold program** under parser v4 are used, for train, dev and test
    (A8). The 11 without one are listed with reasons in DECISIONS D28.
  - 8 instances per template, split 33/16/40 by template with seed 0.
  - Target: N* = max(320, ⌈213 × deff⌉) held-out primary test failures with a first-error
    step, from at least 40 test templates.
    - The Kish deff, 1 + (m̄ (1 + CV²) − 1) × ICC, is that of the **primary paired outcomes**:
      E3 gold-value decodability and the paired repair difference. It is estimated on train
      and dev templates before any primary analysis (A15).
    - If it exceeds 1.5, or the primary test failures fall short of N*, and the budget allows,
      instances 9–16 are generated. Otherwise the study is reported as underpowered.
    - The pilot's failure-occurrence deff (ICC 0.586, deff 19.76) is descriptive.
    - Every result states its template clustering and effective n (A16).
- **Generation.** Greedy decoding (T = 0), fixed templates. Two prompt styles and three
  paraphrases, as before.
  - **Batched** Hugging Face generation: KV cache, float32, eager attention, left padding,
    pure greedy.
  - Generation stops at the end of the "Final Answer:" line, with max_new_tokens 512. The
    fraction of traces that hit the cap is reported, and truncated traces count as failures.
  - The sampled subset (T = 0.7, k = 5) is drawn from **test-split greedy failures of the two
    populations only**. It
    is used solely to separate stable from borderline failures.
- **Tooling.** TransformerLens hooks, SAELens and the released code of Qu and Gomes. Each
  finished trace gets **one verified TransformerLens forward pass**, which does two things:
  - **Equivalence check.** On a random sample of at least 200 traces, stratified by template,
    teacher-forced TransformerLens logits must reproduce the Hugging Face greedy tokens at the
    Phase 0 threshold (0.99). The maximum logit difference is reported. If the check fails,
    generation falls back to TransformerLens greedy decoding, and the report says so.
  - **Activation caching.** In the same pass, the residual stream at all layers is cached at
    the pre-registered positions only: the first token of each step (primary), the last token
    of each step, and the final-answer token. It is stored in float16 with an index, and the
    maximum float16-vs-float32 error is reported.
- **Hardware.** Minor project: free Kaggle T4s (2 × 16 GB, about 30 GPU-hours a week), with a
  25 T4 GPU-hour budget for S2–S4 that is checked by a timed smoke test (`make estimate`).
  Over budget, a rule fixed beforehand applies automatically: first fewer instances, down to
  the power floor, then no sampled runs (A13). Weights: Kaggle's copy of Gemma-2-2B-IT
  (version 2), with Phase 0 re-run on it and the file hashes recorded (A12).
  Major project: a larger GPU (≥ 40 GB) for 9B and Qwen-7B.

## Section 10: Work Plan (replacement table)

| Milestone | Deliverables | Gate / check |
|---|---|---|
| M0 (Sem. 7, week 1) | Harness; TransformerLens–HuggingFace equivalence; SAE validation; Appendix A check, **on 2B** | Done: 6/6 equivalence checks, 24/26 SAE layers |
| M1 (weeks 2–3) | Phase 0 on the Kaggle weights; `make estimate` on Kaggle T4s; human review of the distractor clauses; failure sets and gold variable extraction **on 2B**; batched-generation equivalence gate; residual cache | Phase 0 6/6 (+ repeat-greedy); equivalence gate ≥ 0.99 on ≥ 200 traces; N* = max(320, ⌈213 × deff_primary⌉) primary test failures with s0 from ≥ 40 templates, or the A15 top-up / reported as underpowered |
| M2 (weeks 4–6) | Direction construction (E1) and readout map (E2), on 2B | RQ1 and RQ2 criteria |
| M3 (weeks 7–10) | Self-sourced feasibility (E3), on 2B | RQ3 criterion on 2B; provisionally decides inclusion of RQ7 |
| M4 (weeks 11–13) | Content attribution and typology (E4); minor report including negative results, with its claims restricted to 2B | RQ4 criterion |
| Semester 8 | **S1–S4 on Gemma-2-9B-IT and Qwen2.5-7B-Instruct; RQ3 gate re-check on 9B (decides RQ7 for 9B)**; positive controls (E5); role × structure steering and nonlinear arm (E6, E7); level comparison (E8); editability window (E9); self-sourced repair (E10); path patching and mechanism (E11); baselines (E12); replication and generalization (E13, E14); final report | RQ5–RQ9 criteria |

## Section 12: Limitations and Scope (replacement and added bullets)

- **Models.** The minor project studies **one model, Gemma-2-2B-IT**. Its conclusions (RQ1–RQ4)
  are about that model only. Whether they hold for Gemma-2-9B-IT and Qwen2.5-7B-Instruct is
  tested in Semester 8 under RQ8. Larger, frontier and reasoning-trained models are not tested.
- **Model size.** 2B is the smallest model in the plan, and it fails more often for reasons that
  may differ from those of larger models. In particular, the RQ3 decodability result and the
  RQ7 decision based on it may not transfer to 9B, which is why the gate is re-checked there.
- **Decoding.** Greedy decoding throughout, generated in batches with Hugging Face and verified
  in TransformerLens. Outputs end at the answer line, with a 512-token cap. A small sampled
  subset of test failures gives stability estimates. Conclusions about sampled generation are
  not claimed.
- **Statistical power.** Power is judged on the design effect of the primary paired outcomes,
  estimated on train and dev templates before any primary analysis (A15). If that deff is as
  large as the pilot's for failure occurrence (19.76, descriptive), 40 test templates give at
  most about 68 effective units, short of the 213 the power analysis needs.
  - The pilot also projects about 292 primary test failures with s0 at 8 instances, below the
    320 floor. So the A15 top-up may apply, and the study may be reported as underpowered.
  - Every result states its template clustering and effective n (A16). The template-clustered
    bootstrap CIs, unchanged, carry the clustering into every interval.
- **Storage precision.** Residuals are computed in float32 and stored in float16. The measured
  maximum error is reported. Analyses that need full precision recompute from the traces.
- Task domain, variables, edit family and mechanism evidence: unchanged.

## Section 14: Risks and Mitigation (changed and added rows)

| Risk | Mitigation |
|---|---|
| Not enough GPU time (college cannot provide ~70 GPU-hours) | Minor project on 2B only, on free Kaggle T4s. Batched generation, stop at the answer line, one TL pass per trace for verification and caching, and sampled runs on test failures only. `make estimate` projects GPU-hours before any full run. Above 25 T4-hours, the rule fixed beforehand applies: fewer instances down to the power floor, then no sampled runs; it is recorded, and anything beyond it goes to the group. |
| Failures cluster strongly by template (pilot ICC 0.59) | Descriptive since A15. The target uses the primary-outcome deff, with a pre-specified top-up before any primary analysis. Template-clustered CIs and effective n with every result (A16). |
| Batched Hugging Face generation differs from TransformerLens | Pre-registered equivalence gate on ≥ 200 traces stratified by template, at the Phase 0 threshold, with a max logit difference report. If it fails: TransformerLens generation, reported. |
| Kaggle sessions end (12 h limit, preemption) | Every stage is resumable per trace or shard and is run as Save & Run All commits. Each session restores the previous output and stops cleanly before its deadline. |
| Results do not transfer from 2B to 9B (replaces "from 9B to 2B") | The minor-project report claims 2B only. The RQ3 gate is re-checked on 9B at the start of Semester 8, and RQ8 reports effect sizes per model. |
| Content variables are not decodable inside failures | RQ3 decides in week 10 of Semester 7 **for 2B**. If it fails on 2B, RQ7 is dropped provisionally. The Semester 8 re-check on 9B decides for 9B. Oracle steering and counterfactual patching carry RQ5 either way. |
| Compute limits (Semester 8) | All pipelines are debugged on 2B. 9B and Qwen runs are only for pre-registered conditions. |
| Other rows | Unchanged. |

---

## Compute: measured estimate

Measured on 2026-11-13 by `make estimate` on Kaggle 2 × T4 (code `b920121`), after Phase 0 had
passed on the Kaggle weights (7/7). Full report: `reports/m1/estimate_gemma-2-2b-it.md`.

| Stage | Measured on T4 | Projected T4 GPU-hours |
|---|---|---|
| Greedy generation, 4,272 traces (89 × 8 × 6) | 71 new tokens/s per GPU, mean 138 tokens | 2.31 |
| Verification + residual cache (one TL pass) | 0.39 s per trace | 0.58 |
| Sampled runs, 4,965 (test failures × 5) | 69 new tokens/s per GPU | 3.13 |
| **Total** (model loads included) | | **6.02** |
| **Total × 1.25 safety** | | **7.52**, budget 25 |

- Within budget, so the A13 rule did not apply.
- Kaggle quota about 3.0 h, in one session.
- Disk: **6.66 GB**, of which 6.51 GB is float16 residual shards at 12.2 positions per trace.
- TL-generation fallback, if the equivalence gate fails: a total of 16.8 (21.0 with the
  safety factor).
- The proposal's instance top-up to 16 would take it to about 12 (15).
- Smoke checks:
  - 0.5% of traces hit the cap;
  - TL reproduced the HF tokens exactly, with a max logit difference of 3.0e-4;
  - float16 storage max relative error 4.1e-4.
- This replaces the interim projection (4.7 GPU-hours) given here before the measurement.

**Re-projected for the distractor design (A14)** from the same measurement
(`reports/m1/estimate_gemma-2-2b-it_reprojected.md`):

| Stage | Projected T4 GPU-hours |
|---|---|
| Generation, 8,544 traces (2 conditions) | 4.60 |
| Verification + cache | 1.03 |
| Sampled runs, 7,610 (2,645 primary, 4,965 natural) | 4.79 |
| **Total** | **10.43** (× 1.25 = **13.04**), budget 25 |

- Disk: 13.2 GB.
- If the equivalence gate fails, TL generation would bring the total to 33.9 even after the
  budget rule, so the run would not start and the group would decide.
