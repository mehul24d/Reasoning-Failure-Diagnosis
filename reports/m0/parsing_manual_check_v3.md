# Hand check of parser v3 (sample `parsing_sample_v3.md`, seed 2)

The v3 validation was reproduced off the Mac with the same code and traces. GSM8K train was read
from the official `openai/grade-school-math` `train.jsonl` (7,473 rows). All 1,000 trace indices
matched their stored gold answers. The `s1c_traces` and `gsm8k_train` blocks of
`parsing_validation_v3.json`, and the sample file, came out identical.

Each sampled step was judged with the whole trace and the gold program in view, against the
plan's rules: no k(s) → planning; wrong operand set → binding; wrong result → arithmetic. A step
that correctly reuses an earlier wrong value is not a new error.

**Result: 33 correct, 7 wrong** (v2: 28/12, v1: 21/19).

| # | trace | verdict | cause |
|---|---|---|---|
| 1 | 647 terse | WRONG | `40 - 38 = 2` is labelled binding in a correct trace. 38 comes from `34 + 1 + 1 + 1 + 1 = 38`, which was not parsed: the '.' in "Ms. Delmont" was read as a sentence end. |
| 7 | 3572 marked | WRONG | `8 + 6 = 14` correctly reuses the wrong 6 from step 2, but is labelled binding. The reference to a step aligned at tier 4 became `rres`, which never matches gold. s0 = step 2 is right. |
| 14 | 1666 marked | WRONG | Compound `0.10 * 7 * 4 = 2.80` is not aligned to any gold step, yet has no error type. The plan says no k(s) → planning. (s0 = step 0 is right.) |
| 18 | 5599 terse | WRONG | Same cause: the unaligned compound `40 + 24 + 4.8` gets no type. |
| 22 | 5566 marked | WRONG | Same cause, and here it matters: `5 + 7 + 5 = 17` is the only error, so the failed trace ends up with no s0. |
| 26 | 4166 marked | WRONG | The trace works in dollars (`10 * $0.10`), the gold in pennies. Its constants 1/10, 1/20 and 1/100 are not gold values, so this valid route is labelled planning/binding (s0 = 0 in a correct trace). |
| 40 | 1827 terse | WRONG | `13 * 5 = 65` (eggs per week) is a valid regrouping (then `65 * 52 = 3380`), but it is labelled binding (tier 4). The support rule only rescued planning steps. |

Other problems seen in the same traces (not sampled steps):
- 3671: `3 mph x 2 hours = 6` was not parsed, because "x" after a unit word was not read as times.
  `Time at 3 mph + Time at 1 mph = 2 hours + 4 hours = 6 hours` was read as `3 + 1 = 2`, which
  gave s0 in a correct trace.
- 6794: `$4 + $3 = $7` (Kiera's subtotal) is binding at tier 4. Same cause as #40.
- 1762: `(1/5) * 225 = 45` is binding because 1/5 is not a gold constant (gold uses 5).

Failed traces in the sample with no error step: #9 (6783), #10 (2356) and #13 (6420) are
omissions, where every step is right but the trace stops before the last gold step. #22 (5566)
is a parser miss (see above).

## Why "gold matched" in failed traces fell from about 0.49 to 0.31

This is expected. v3 stopped tiers 4–5 from using up gold steps. In failed traces 205/451 (marked)
and 141/358 (terse) of the computational steps are tier 4–5 matches. Under v2 those took gold
steps, which inflated the matched share. With v4's tier-4 references (a step reusing an earlier
wrong result counts against the gold step it was meant for) the share is 0.375/0.40.
