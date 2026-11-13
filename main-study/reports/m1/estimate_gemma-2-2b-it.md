# Compute estimate: gemma-2-2b-it (`make estimate`, 2026-11-12)

Measured on **Tesla T4, Tesla T4**, code `b920121`, smoke test on 384 greedy traces (GSM8K train, never the study population).

| Stage | measured | projected T4 GPU-hours |
|---|---|---|
| Generation (HF batched, batch 16) | 71 new tokens/s per GPU, mean 138 tokens | 2.31 |
| Verification + caching (TL) | 1.60 ms per sequence token, 0.39 s per trace | 0.58 |
| Sampled runs (T = 0.7, k = 5) | 69 new tokens/s per GPU, mean 155 tokens | 3.13 |
| **Total** (model loads included) | | **6.02** (x1.25 = 7.52) |

* Sizes: 4272 greedy traces (89 templates x 8 instances x 6 prompts), 4965 sampled runs.
* Kaggle quota: about **3.0 h** of session time with 2 GPU(s) in parallel, in 1 session(s) (setup 1.4 min each).
* Disk: activation store 6.51 GB, traces 0.06 GB, total **6.66 GB** (12.2 cached positions per trace).
* Smoke checks (not results): cap hit 0.005, stop reasons {'answer_line': 377, 'eos': 5, 'max_new_tokens': 2}, TL reproduction 1.0000, max logit diff 0.00029754638671875, float16 max rel. error 0.00040624456596560776.
* If the equivalence check failed, TL generation would cost about 13.052 T4 GPU-hours for generation (total 16.778).

Within budget (25 T4 GPU-hours, safety factor included).
