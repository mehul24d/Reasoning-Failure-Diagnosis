# Hand check of parser v4 (sample `parsing_sample_v4.md`, seed 3)

Changes from v3 (each with a regression test; 106 tests pass):

1. **Compound steps:** aligned by the stated result, or by the exact value when the stated result
   is wrong. Unaligned and right → planning (extra_valid if it re-derives a gold result from valid
   quantities). Aligned and right → none. Wrong → arithmetic.
2. **References:** go through tiers 1–4. A step that correctly reuses an earlier wrong result is
   not a new error; the error stays with the earlier step.
3. **Valid alternative derivations:** a right step from valid quantities whose result feeds,
   directly or through such steps, a step reaching a gold result is extra_valid. This covers
   planning and binding steps, compound steps and compound leaves.
4. **Constants:** valid if the gold program uses c, 1/c, c − 1 or 1 − c (`× 1/5`, `× 1.15`).
   extra_valid needs a gold *result*, not a problem number.
5. **The last computational step:** never extra_valid or rederived unless it states the gold
   answer. If a trace ends off the answer with no other error, lenient s0 falls back to strict s0.
6. **Swapped operands of `-` and `/`:** a binding error (tier 4), not the same step.
7. **Results written "n%":** also match a gold result n.
8. **Expression reading:**
   - A '.' after Mr/Ms/Mrs/Dr/St/vs does not end a clause, and the clause limit is now 240 chars.
   - "x" or "times" after a unit word means times (`3 mph x 2 hours`).
   - A unit letter on every number is a unit (`3m * 2m`; `2m - 1` is still algebra).
   - Hyphenated words are single words (`time-outs`).
   - A left side made of labels (`Time at 3 mph + Time at 1 mph = 2 hours + 4 hours = 6`) is
     dropped.
   - "Label = expr" followed by "Label = value" on the next line is one equation.

## Real data (S1c traces, GSM8K train gold)

| group | traces parsed | has error step (lenient / strict) | v3 lenient |
|---|---|---|---|
| marked/correct | 310 / 341 | 0.052 / 0.203 | 0.123 |
| marked/failed | 132 / 159 | 0.924 / 0.924 | 0.824 |
| terse/correct | 336 / 371 | 0.045 / 0.205 | 0.122 |
| terse/failed | 106 / 129 | 0.811 / 0.811 | 0.743 |

The GSM8K gold program was ok for 88.45% of problems (v3: 88.18%).

Trace-level changes from v3:
- 49 correct traces lost a false error step and 1 gained one.
- 20 failed traces gained an error step and none lost one.

I checked every changed trace in the risky directions by hand: the 9 failed traces that lost
their error step in an intermediate version, and all correct traces that gained one. A sample of
8 of the 49 correct traces that lost an error step were all valid alternative derivations.

The 30 failed traces still without an error step:
- 14 are omissions: every step is right, but the trace stops early or ends on a gold intermediate.
- 12 have no computational step: algebra with variables, `<year>` placeholders, or no equations.
- 3 have a last step that equals the gold answer; the answer extraction differs.
- 1 hit the token limit.

## Hand check of the seed-3 sample

**Result: 38/40 right at check time.** The one fix made after the check (growth factors as valid
constants) turns #28 right and changes nothing else in the sample. That makes **39/40 on the
final code**.

| # | trace | verdict | note |
|---|---|---|---|
| 27 | 1559 marked | WRONG | The trace works in cents, the gold in dollars, so its constants are not gold values. The valid route is labelled binding (s0 = 0 in a correct trace). This is a known limitation: a different unit system. |
| 28 | 5659 marked | wrong → fixed | `100 * 1.15 = 115` was planning (1.15 not valid). Now rederived/none. |

The other 38 are right. Defensible-but-arguable labels:
- #4 (1326) and #16 (2551): binding on steps inside an already wrong plan.
- #36 (3435): `$6 / 1200 calories` (swapped division) is binding.
- #23 (7006): a correct trace computes an unneeded `30 / 35 = 0.86 packets`. By the plan's rule
  this is a planning step, so s0 is set.

## The wrong steps from earlier hand checks, re-run with v4

- **v3 (7):** 6 now right (647, 3572, 1666, 5599, 5566, 1827). Still wrong: 4166 (dollars vs
  pennies, the same limitation as #27).
- **v2 (12):** 10 now right (1762, 3623, 6209, 2764, 374, 5368, 3032, 3234, 3183, 5746).
  Still wrong:
  - 1540: the compound-interest link `10000 * 1.728 = 17280` is a valid route but is labelled
    binding, so s0 = 7 rather than step 11, where the real arithmetic slip is now caught.
  - 333: s0 is one step early. `2 + 1 = 3` per week is a valid side value that feeds no gold
    result.
- **v1 (7):** 6 now right (6074, 4459, 2764, 5387, 5473, 117). 3445: the label (planning) and
  s0 are right, but the step is still tied to gold 0 through a weak tier-5 match.

## Known limitations (for the S3 pre-registration)

- Routes in another unit system (cents vs dollars) or with derived constants (1.2³) are not
  recognised as valid.
- Coincidental value matches: a wrong step equal in value to a gold quantity is treated as that
  quantity. Rules 5 and 2 limit the damage at the end of a trace.
- Algebra (variables) is out of scope, so such traces have no computational steps.
- Report both lenient and strict s0. Lenient is the main measure, with the rules above fixed in
  the pre-registration.
