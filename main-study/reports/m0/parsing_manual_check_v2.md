# Manual check of parser v2 (reports/m0/parsing_sample_v2.md, seed 1)

I applied the same standard as v1: a step counts as **correct** only if the op, operands, result,
aligned gold step and error type are all right. I checked every step against the question, the GSM8K
solution and the full trace. Numbering follows parsing_sample_v2.md.

**Result: 28 correct, 12 wrong** (v1: 21 / 19). 8 of the 12 are false error labels on steps that are
right, and all 8 are in traces that are correct.

| # | problem | verdict | note |
|---|---|---|---|
| 1 | 1762 marked | WRONG | `225 + 45 = 270` is a valid regrouping (the gold does `300-225-45`), but it is labelled planning (gold 3, tier 5). Step 1 (`1/5 * 225 = 45`, also valid) is labelled binding. This puts s0=1 in a correct trace. extra_valid can't catch these because 270 is not a gold result and step 1 is aligned. |
| 2 | 831 terse | correct | |
| 3 | 3623 marked | WRONG (parse) | `8 minutes/day * 5 days + 90 minutes = 40 minutes + 90 minutes = 130 minutes` is parsed as `5 + 90 = 40`, a non-compound step with an inconsistent result, and labelled planning. Step 3 (`... = 60 minutes + 30 minutes = 90 minutes`) registers 60 instead of 90. This gives s0 in a correct trace. |
| 4 | 1540 terse | WRONG | `$17280 * 3 = $51840` is exactly gold 3, but it is unaligned (extra_valid, planning). Gold 3 was taken by step 7: `A = 10000(1 + 0.20/1)^(1*3) = 10000` was parsed as `1 * 3 = 10000`. Exponents and implicit multiplication are accepted, and so is an inconsistent result. (The real arithmetic error in step 12, `51840 * 1.15 = 59312`, is caught correctly.) |
| 5 | 6967 terse | correct | The inline gold is now used. |
| 6 | 6209 terse | WRONG | `4 basil + 1 parsley = 5` is a valid alternative path in a correct trace (basil 4 + parsley 1 + mint 0). It is aligned to gold 3 by result (tier 3), and because the op differs it is labelled planning. |
| 7 | 6648 terse | correct | |
| 8 | 5331 terse | correct | |
| 9 | 2764 marked | WRONG | `106 / 2 = 53` is exactly gold 4 but is unaligned and labelled planning. The redundant valid `138 - 69 = 69` took gold 3 at tier 5, then `69 + 37` took gold 4 at tier 5. Tier-5 matches still use up gold steps. (This is v1 #18 again.) |
| 10 | 1203 marked | correct | |
| 11 | 6847 terse | correct* | `40,000 + 28,000 = 68,000` → gold 4, none, via value comparison. The operands are consts because steps 6 and 7 (`4000 soldiers * 10 pounds/soldier = 40,000 pounds`) still don't parse. |
| 12 | 374 terse | WRONG | `50 / 2 = 25` is equivalent to gold `1/2 * 50`, but it is aligned by result and labelled planning because the op differs. This gives s0 in a correct trace. |
| 13 | 5510 terse | correct | |
| 14 | 5956 terse | correct* | Correct via value. `300 km / 15 km/liter = 20 liters` and `150 km (each way) * 2 = 300 km` don't parse. |
| 15 | 12 terse | correct | |
| 16 | 6191 terse | correct | |
| 17 | 3691 marked | correct* | `5 plants * 14 strawberries/plant = 70 strawberries` (steps 0–3) still doesn't parse, while `10 baskets * $9/basket` does. |
| 18 | 3155 terse | correct | |
| 19 | 1265 terse | correct | |
| 20 | 4459 marked | correct | Same trace: `2.50 + 1.20 = 3.70` (planning) and `4.00 - 3.70` (binding) are false errors from an alternative decomposition. |
| 21 | 395 marked | correct* | Step 0 (`= 2,880 elephants/hour * 4 hours = 11,520 elephants`) doesn't parse. |
| 22 | 333 marked | WRONG | The real error is step 1 (`3 + 10 = 13`: the extra 10 is added to the weekly count, which is a plan error). It is labelled binding through a tier-4 match to gold 3. Step 0 (`2 + 1 = 3` per week, a valid alternative) is labelled planning, so s0 is step 0, one step too early. |
| 23 | 346 marked | correct | Compound, aligned by result. |
| 24 | 117 terse | correct | |
| 25 | 5368 marked | WRONG | `132 / 6 = 22` (sum first, then divide) is a valid alternative in a correct trace. It is aligned by result (gold 6, `15+7`) and labelled planning. extra_valid doesn't apply because the step is aligned. |
| 26 | 2933 terse | correct | |
| 27 | 5808 terse | correct | Same trace: step 2 (`8 * (1/4)` vs gold `8/4`) is labelled planning, the same equivalent-form problem as #12. |
| 28 | 374 terse | correct | Compound. |
| 29 | 7452 terse | correct | Compound. Same trace: the repeated `900 / 3 = 300` for stocks B and C is labelled planning, not extra_valid, because the const 3 operand is excluded from extra_valid. This gives s0 in a correct trace. |
| 30 | 3032 terse | WRONG | `400 + 200 = 600` is a valid regrouping (the gold does `2000-400-200`) and is labelled planning at tier 5. Step 1 (`400/2` vs `1/2*400`) is also labelled planning. |
| 31 | 5992 terse | correct | Compound `1000 * 1/1000 = 1`. But the next step, `500 pounds / 1 pound/day = 500 days`, doesn't parse, so this failed trace has no error step at all. |
| 32 | 6967 terse | correct | |
| 33 | 3234 marked | WRONG | `124600 - 62300 = 62300` (remaining = total − half) is valid but labelled planning at tier 5, which takes gold 1. The real gold-1 step (`62300 / 10`) then becomes extra_valid. Step 0 (`124600 / 2` vs `1/2 * 124600`) is labelled planning, so s0=0 in a correct trace. |
| 34 | 4870 marked | correct | (v1 #1, now fixed) |
| 35 | 3183 marked | WRONG | `400 - 320 = 80` is a valid regrouping (sum, then subtract) and is labelled binding (tier 3). Step 0 (`120 + 200`) is labelled planning, so s0=0 in a correct trace. |
| 36 | 3005 marked | correct | |
| 37 | 6468 marked | correct | |
| 38 | 4051 terse | correct* | Correct via value. `$10 (salad) + $25 (hamburgers) + ... = $55` doesn't parse because of the parenthesised words. |
| 39 | 331 terse | correct | Same trace: `7 miles * 3 truckloads/mile = 21` and `11 miles * 3 truckloads/mile = 33` don't parse. |
| 40 | 5746 marked | WRONG (alignment) | `$12 + $48 = $60` comes after the real error. It should align to gold 2 (`12+12`), but gold 2 was taken at tier 4 by step 1 (`4 teachers + 12 students = 16`, which should be unaligned). |

`correct*` means the label is right, but only because v2 compares values; an earlier step that should
have parsed did not.

## The 19 steps that were wrong in v1, re-run with v2

**Fixed (12):** v1 #1, #3, #8, #11, #20, #26, #30, #31, #32, #35, #37, #39. Each now has the right
gold step and the label none; #39 is now non-computational.

**Still wrong (7):**
- **#6 (6074)** `500 - 250 = 250`: still binding (tier 4). It should be extra_valid or none.
- **#10 (4459)** `$4.00 - $3.70`: still binding, an alternative decomposition. Cause 7 from v1 was not addressed.
- **#15 (3445)** `$5.00 / 3`: the planning label is right, but it is still aligned to a gold step (gold 1, tier 5) when it should be None.
- **#18 (2764)** `106 / 2 = 53`: still unaligned and planning, for the tier-5 reason above.
- **#29 (5387)** `600 + 750`: the operands are still consts because `20 emails/day * 30 days = 600` doesn't parse, so s0 is still late.
- **#33 (5473)** `30 / 2 = 15 days`: now binding instead of planning, but it is a valid alternative in a correct trace.
- **#38 (117)**: still aligned to gold 3 (op `-`) instead of gold 2. The label is acceptable because the step comes after s0.

There is one new trace-level finding from v1 #3 (606, a failed trace). The only error is an arithmetic
slip inside a compound step (`180 - 20 - 60 - 12 = 98`). Compound steps get no error type, so this
failed trace has s0=None.

## Remaining causes, ranked

1. **Equivalent forms and valid alternative paths in correct traces** (#1, #6, #12, #25, #30, #33, #35, and steps in #20/#27/#29). This is now the main source of false s0 in correct traces.
   - `x/2` vs `1/2*x` and `x*(1/4)` vs `x/4`: the op differs, so the step is labelled planning. Dividing by a constant could be normalised to multiplying by its reciprocal before comparing.
   - Regroupings: `a-b-c` vs `a-(b+c)`, and sum-then-divide vs divide-then-sum.
   - Repeated identical computations whose operand is a const are not extra_valid.
   - A step aligned by result alone (tier 3) with a different op is called planning, even when it is a correct re-derivation.
2. **Weak alignment tiers still use up gold steps** (#9, #33, #40, v1 #15/#18/#38). A tier-5 (or tier-4) match takes the gold step and blocks the later exact match. Options: let only tiers 1–3 consume a gold step, or align the whole trace globally (an assignment problem, preferring exact matches).
3. **Parser accepts inconsistent binary equations** (#3: `5 + 90 = 40`; #4: `1 * 3 = 10000` from `10000(1+0.20/1)^(1*3)`). Steps with `^`, implicit multiplication or a truncated left-hand side should become non-computational, rather than count as arithmetic slips or be aligned. A chained compound registers an intermediate value (60) instead of the final one (90).
4. **Unit forms that still don't parse**:
   - `5 plants * 14 strawberries/plant`
   - `4000 soldiers * 10 pounds/soldier`
   - `300 km / 15 km/liter`
   - `150 km (each way) * 2`
   - `$10 (salad) + $25 (hamburgers)`
   - a leading `= 2,880 elephants/hour * 4 hours`
   - `500 pounds / 1 pound/day`
   - `7 miles * 3 truckloads/mile`

   The pattern seems to be a word unit, `/` and another word unit on an operand without `$`; `10 baskets * $9/basket` does parse. Value comparison saves most labels, but these steps are missing, so errors in them are invisible (#31, and #39 context).
5. **Arithmetic errors inside compound steps get no type** (v1 #3), so some failed traces show no error at all.

Aggregates: 18–19% of correct traces have an s0 (v1: 31–32%), against 76–77% of failed traces (v1: 47–50%).
The gap is now large. From this sample, most of the remaining false positives in correct traces come
from cause 1, then cause 2.
