# Manual check of the 40 sampled parsed steps (reports/m0/parsing_sample.md)

I checked each step against the question, the GSM8K solution and the whole trace. A step counts as
**correct** only if the op, operands, result, aligned gold step and error type are all right. Numbering
follows the order in parsing_sample.md.

**Result: 21 correct, 19 wrong.** 13 of the 19 wrong ones are false error labels on steps that are
actually right, most of them in traces that are correct.

| # | problem | verdict | note |
|---|---|---|---|
| 1 | 4870 marked | WRONG | `28 * 13 = 364` is exactly gold 2, so the label should be none. It is marked binding because step 0 (`26/2`) was aligned to gold 0 (`183*26`) just because they share slot 0. That makes 13 `res 0` rather than `res 1`. The cause is greedy alignment. |
| 2 | 5252 marked | correct | `(15/100) * 120 = 18` is correctly marked compound and has no type. |
| 3 | 606 marked | WRONG | `60 * 0.20 = 12` is equivalent to gold `60*20/100`. The gold splits this into `60*20=1200`, then `/100`, so the step is marked binding (const 1/5 vs slot 20). The label should be none. |
| 4 | 3005 marked | correct | `400 + 560 = 960` → gold 1, none. |
| 5 | 6490 marked | correct | |
| 6 | 6074 marked | WRONG | `500 - 250 = 250` is a redundant step that is not in the gold. It was aligned to gold 1 (`250-5`) through the shared res 0 and marked binding. Then the real gold-1 step (`250 - 5 = 245`, step 4) gets planning because gold 1 was already taken. |
| 7 | 5083 marked | correct | |
| 8 | 3691 terse | WRONG | `$90 + $96 = $186` is exactly gold 6, but it is marked planning with const operands. None of the earlier equations parsed because units sit between the operands (`10 baskets * $9/basket = $90`, `14 strawberries/plant * 5 plants = 70 ...`). |
| 9 | 5885 terse | correct | |
| 10 | 4459 terse | WRONG | `$4.00 - $3.70 = $0.30` is a valid alternative decomposition (the gold does `4-2.5-1.2`) and is marked planning. Its inputs (`10 quarters * $0.25/quarter = $2.50`) did not parse because of the units. |
| 11 | 7306 terse | WRONG | In `2 * $2.50 = $5`, the 2 is mapped to slot 0 (the lemonade cups) instead of slot 2 (the sandwiches), so it is marked binding. The step is flagged ambiguous, but binding is still assigned. It should be none, or excluded. |
| 12 | 2484 marked | correct | `21 + 15 + 18 = 54` is correctly marked compound. |
| 13 | 6384 marked | correct* | `18 + 3 = 21` → gold 4, none. *Both sides resolve "3 years" to res 1 (Sue's age = 3) instead of slot 2, so the label is right only by coincidence. Step 2 in the same trace (`6+3=9`) is aligned to gold 2 (`3+3`) instead of gold 3 and is falsely marked binding. |
| 14 | 1551 terse | correct | |
| 15 | 3445 marked | WRONG (alignment) | `$5.00 / 3 = $1.67` really is a planning error, so the label is right. But it is aligned to gold 0 (`5+3`) through a shared slot when it should be None. $5.00 is the kids' meal (slot 9), not slot 0. |
| 16 | 1551 terse | correct | |
| 17 | 1144 terse | correct | |
| 18 | 2764 marked | WRONG | `106 / 2 = 53` is exactly gold 4 and is marked planning. Step 0 (`100 horses + 29 sheep + 9 chickens = 138`) did not parse, so the whole chain becomes const/rres. |
| 19 | 6739 marked | correct | `10 - 3 = 7` → gold 1, none. (Later steps like `8 ounces ... per candle * 7 candles = 56` did not parse.) |
| 20 | 1560 terse | WRONG | `$60 - $54 = $6` is exactly gold 5 and is marked planning. 54 came from a compound step (`(3/4) * $72 = $54`), so it is `rres` and can never match gold `res 4`. |
| 21 | 3896 terse | correct | |
| 22 | 1203 marked | correct | |
| 23 | 1019 marked | correct | `93 - 81 = 12` is a real planning error (the model ignored the re-baked cookies). |
| 24 | 4008 terse | correct | |
| 25 | 5876 terse | correct | |
| 26 | 6967 terse | WRONG | `2 * 20 = 40` is correct, but the gold program has a single step (`44+20`, with 44 as a const). The solution writes `2*20 = 40` and `40+4 = 44` without `<<>>` annotations, and the program is still marked ok. The step aligns to gold 0 through slot 1 and gets planning. |
| 27 | 1224 marked | correct | |
| 28 | 4406 marked | correct | |
| 29 | 5387 terse | WRONG (refs) | The planning label for `600 + 750 = 1350` is defensible, because the trace's error is using 30 days for both rates. But 600 and 750 come out as consts because steps 3/4 (`20 emails/day * 30 days = 600`) did not parse. The true s0 is an unparsed step. |
| 30 | 3904 terse | WRONG | `$80 - $75 = $5` is exactly gold 8 and is marked planning. 75 came from a compound step (rres). In the same trace, `$2 * 18`, `$1 * 3` and `$6 * 6` are all falsely marked binding. |
| 31 | 2308 terse | WRONG | `2 * 12 = 24` should align to gold 0 (`12*2`) with label none. Instead it is aligned to gold 2 (`2+3`) and marked planning, because the earlier `2 + 3 = 5` step took gold 0 through the shared const 2. Also, the second equation on that line (`24 - 8 = 16`) is lost: the line was not split, probably because normalising changed its length. |
| 32 | 6867 marked | WRONG | `$19.00 + $92.00 = $111.00` is exactly gold 1 and is marked planning. 19 is a const because `2 pairs * $9.50/pair = $19.00` did not parse. |
| 33 | 5473 terse | WRONG | `30 / 2 = 15 days` is a valid alternative path in a correct trace (the gold halves the money, not the days) and is marked planning. |
| 34 | 6559 terse | correct | `20 - 7 = 13` is a real planning error (the model left out the purple tiles). It is aligned to gold 2 by its result, and planning is right. |
| 35 | 3047 marked | WRONG (parse) | `200 seats * (3/4) = 150` is parsed as `3 / 4 = 150`, which is non-compound with a result inconsistent with its operands. The left-hand side was cut off after the unit word. It is marked planning, but the step is correct. |
| 36 | 948 marked | correct | |
| 37 | 6847 marked | WRONG | `28000 + 40000 = 68000` is exactly gold 4 but is aligned to gold 3 (`8*3500`) and marked planning. Earlier in the chain, step 0 (`4000-500`) was aligned to gold 0 (`4000*10`) through the shared slot. |
| 38 | 117 terse | WRONG (alignment) | `12 / 3 = 4` is downstream of the real error at step 3 (`24/2`). It is aligned to gold 3 (`24-8`, op `-`) instead of gold 2 (`24/3`). The planning label is acceptable for a post-s0 step. |
| 39 | 1144 marked | WRONG (parse) | `2m - 1 + 1 = 2m` is parsed as `1 + 1 = 2`. Algebra with variables should be non-computational. |
| 40 | 4967 terse | correct | |

## Root causes (ranked by how many sampled errors they explain)

1. **Unit words between operands are not parsed** (#8, #10, #18, #29, #32, #35, and the context of #19 and #30). Examples: `14 strawberries/plant * 5 plants = 70`, `2 pairs * $9.50/pair`, `100 horses + 29 sheep + 9 chickens = 138`. The step becomes non-computational, every later step that uses its result gets const/rres operands, and those steps are marked planning. This is the largest source of false errors in correct traces.
2. **Greedy alignment on any shared operand** (#1, #6, #15, #31, #37, #38, and context of #13 and #23). k(s) takes the earliest unmatched gold step that shares *any* operand or the result. An exact match (same op and operand set, or same result) should be preferred before a partial overlap.
3. **Results of compound steps are never mapped to gold results** (#20, #30, and context of #2). The compound result is registered as `rres`, so a later step that uses it can never match gold `res k`. A compound step could be aligned by its final result to the gold step with that result and registered as `res k`.
4. **The GSM8K gold extractor ignores unannotated equations** (#26, and context of #4, #17 and #29). Inline `2*20 = 40` without `<<>>` becomes a const in the next step, and the program is still `ok`. This probably explains much of the 16% const-operand rate.
5. **Gold splitting of percentage and fraction expressions** (#3, #35): `x*20/100` gives an intermediate `x*20` that no model step produces, so the equivalent `x*0.2` is marked binding.
6. **Ambiguous operand still produces binding** (#11, and the `$6*6` case in #30).
7. **Valid alternative decompositions are marked planning** (#10, #33). This is a design limit, but it contributes to the false-positive rate in correct traces.
8. **Parser robustness**: truncated LHS accepted with an inconsistent result (#35); algebra with variables parsed as arithmetic (#39); a terse line with two equations not split when normalising changes its length (#31).

The aggregates show the same problem. 31–32% of *correct* traces have an error step (marked 31%, terse 32%), compared with 47–50% of failed traces. Planning makes up 30–35% of the labelled steps in correct traces. Error labels are not usable for S3 until causes 1–4 are fixed.

## GSM-Symbolic gold extraction (main 49% ok, p1 37%, p2 21%), seen in a sample of failures

* Currency symbols other than `$` (¥, ₺, ₣, …) are not stripped, so those equations are unparsed or missing. This often leaves a "last result != answer" because the last equations were dropped.
* Unit words between operands (`400 meters / 40 meters/minute = 10 minutes`) → "no equations".
* Percent results (`3 / 4 = 75%`, `80% - 25% = 55%`): 75% is not equated with 3/4, and the answer is in percent.
* Chained equalities (`72 + 48 + 34 + T = 154 + T = 247`), `÷` with fractions (`21/4÷3/4` parsed as `21/(4÷3)/4`), mixed numbers (`11 1/4`), implicit multiplication (`2(38+30)`), and variables.
