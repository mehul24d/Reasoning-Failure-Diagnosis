# Compute estimate: gemma-2-2b-it (re-projected for the current design, 2026-11-13)

Measured on **Tesla T4, Tesla T4**, code `b920121`, smoke test on 384 greedy traces (GSM8K train, never the study population).

| Stage | measured | projected T4 GPU-hours |
|---|---|---|
| Generation (HF batched, batch 16) | 71 new tokens/s per GPU, mean 138 tokens | 4.60 |
| Verification + caching (TL) | 1.60 ms per sequence token, 0.39 s per trace | 1.03 |
| Sampled runs (T = 0.7, k = 5) | 69 new tokens/s per GPU, mean 155 tokens | 4.79 |
| **Total** (model loads included) | | **10.43** (x1.25 = 13.04) |

* Sizes: 8544 greedy traces (89 templates x 8 instances x 6 prompts x 2 condition(s)), 7610 sampled runs (2645 for primary distractor failures, 4965 for natural failures).
* Kaggle quota: about **5.2 h** of session time with 2 GPU(s) in parallel, in 1 session(s) (setup 1.4 min each).
* Disk: activation store 13.01 GB, traces 0.10 GB, total **13.22 GB** (12.2 cached positions per trace).
* Smoke checks (not results): cap hit 0.005, stop reasons {'answer_line': 377, 'eos': 5, 'max_new_tokens': 2}, TL reproduction 1.0000, max logit diff 0.00029754638671875, float16 max rel. error 0.00040624456596560776.
* If the equivalence check failed, TL generation would cost about 26.104 T4 GPU-hours for generation (total 31.948).

Within budget (25 T4 GPU-hours, safety factor included).

**If the equivalence gate fails (TL generation), with the budget rule:** first projection 31.95 T4 GPU-hours (x1.25 = 39.94); reduce_instances_to_power_floor: not possible; drop_sampled_runs: applied; final 33.94 with safety: **still over budget: the run does not start; the group decides.**

**Top-up (A20):** with instances 9-16 the projection is 20.71 T4 GPU-hours (x1.25 = 25.89) against 25: **no top-up; the study is reported as underpowered.**

Throughputs measured on 2026-11-12 (code `b920121`), estimate_gemma-2-2b-it.json.
