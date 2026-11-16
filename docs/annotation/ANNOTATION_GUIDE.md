# Failure annotation guide (S5)

You will read solutions that Gemma-2-2B-IT got wrong on grade-school maths problems. For each
one you record **where the solution first goes wrong** and **what kind of error it is**. Two of
you label the same items without talking to each other. Agreement between you is measured with
Cohen's kappa. The pre-registered threshold is kappa ≥ 0.6.

## 1. Opening the sheet

You get one file: `annotator_A.csv` or `annotator_B.csv`. Both hold the same items, in a
different order.

1. Open Google Sheets and start a blank spreadsheet.
2. Choose **File > Import > Upload** and select your CSV.
3. In the import dialog choose **Replace spreadsheet**, separator **Comma**, and set
   **Convert text to numbers, dates, and formulas** to **No**.

**Never double-click the file to open it in Excel or Numbers.** The files are UTF-8, and many
problems use currency symbols (₣, ¥, €, ₺, ₹). Excel and Numbers often guess a different
encoding and garble them. If you see something like `â‚¬` where a currency sign should be,
you opened the file the wrong way.

Turn on **Format > Wrapping > Wrap** for the text columns, so the multi-line solutions are
readable.

## 2. What is in each row

| Column | What it holds |
|---|---|
| `item_id` | P01–P10 for practice items, F0001–F0466 for the others |
| `problem` | The problem, exactly as the model saw it. Some problems contain a sentence that is irrelevant to the question. |
| `model_solution` | The model's full solution. Each step has a number in brackets: `[1]`, `[2]`, …. Text before the first step is marked `[before step 1]`; the final line is marked `[answer line]`. |
| `gold_solution` | The reference solution, one step per line: `(1)`, `(2)`, …. |
| `gold_answer` | The correct answer. |
| `model_final_answer` | The number the scorer read from the model's answer line, or `(no answer found)`. |
| `first_wrong_step` | **You fill in.** A step number, `answer`, or `none`. |
| `error_type` | **You fill in.** `planning`, `binding`, `arithmetic` or `other`. Leave it blank only when `first_wrong_step` is `none`. |
| `confidence` | **You fill in.** `3` = sure, `2` = probably right, `1` = a guess. |
| `notes` | **You fill in, optional.** Anything odd, and your reason when confidence is 1. |

**Use the bracket numbers, not the model's own numbering.** The model sometimes writes
"Step 1" twice or skips a number; `[3]` is always the third step in the sheet.

The sheet deliberately shows no automatic labels, no parser output and no model scores. Do not
look for them: do not open `main-study/reports/m1/s5_annotation/sample.json`, the study data
files or the other annotator's sheet. Do not use an AI assistant. A calculator is fine.

## 3. Order of work and timing

1. **Practice: P01–P10** (the first 10 rows, in the same order for both of you). Label them on
   your own. **Write the minutes you spent on each in `notes`.** Then stop.
2. **Practice discussion.** Both of you and the coordinator go through P01–P10 against the
   answer key, which only the coordinator has. If the discussion shows a rule is unclear, the
   coordinator clarifies it here (section 7, dated) **before** anyone starts the main items. Do
   not go back and change practice labels; practice items are excluded from kappa.
3. **Part 1:** the 216 items after the practice rows (the test split). Work top to bottom.
4. **Part 2:** the remaining 250 items (train and dev).

| Part | Items | Estimated time per annotator |
|---|---|---|
| Practice | 10 | 0.4 h, plus the discussion |
| Part 1 | 216 | about 12 h |
| Part 2 | 250 | about 14 h |
| **Total** | **476** | **about 26.5 h** |

The estimate assumes about 3 minutes per item: reading at 120 words a minute, 0.15 minutes to
check each step, and 0.75 minutes to fill in the labels. Your practice timings will replace it.
Sittings of 60–90 minutes with breaks keep the labels consistent.

## 4. Definitions

**Step.** A numbered line `[n]` of the model's solution. Most steps contain a *calculation*:
arithmetic on numbers, written with or without its result (`900 - 250 - 175`,
`3 + 9 + 17 = 30`), or an equation with an unknown (`8 + 24 + x = 38`). Some steps contain no
calculation: a restated number ("White tiles: 8"), an announcement ("Calculate the remaining
pages:") or a comment.

**first_wrong_step.** The first step that contains an error. Write its number, or:
* `answer`: every calculation is right (or there is none), but the answer line is wrong or
  missing;
* `none`: the solution is actually correct (for example, the scorer misread the answer line).

**error_type.** Compare the wrong step with the gold step it is trying to compute. Then pick
the **first** of these that applies (the order of precedence matters):

1. **planning**: the step does the wrong computation. It has the wrong operation (adds where
   the problem needs subtraction), a term too many or too few, or computes something the
   problem does not need. Answering before the problem is finished also counts as planning
   (rule R9).
2. **binding**: the computation is the right one, but at least one number is the wrong
   quantity. Examples: a number taken from the wrong place in the problem (including the
   irrelevant sentence), the wrong earlier result, a number copied wrongly, or the two numbers
   of a subtraction or division swapped.
3. **arithmetic**: the operation and the numbers are right, but the stated result is wrong.
4. **other**: the first error is none of these. In practice this is usually a solution with no
   calculation at all, an answer line with no number, or an answer line that contradicts the
   steps with no visible reason.

These are the parser's error types (plan Algorithm 1: planning, then binding, then arithmetic),
so your labels can be compared with its labels.

## 5. Decision rules

**R1. Judge calculations, not wording.** A step with no calculation is never the first wrong
step. When a wrong idea is announced in words and carried out in the next calculation, mark the
calculation. A wrong label on a right calculation ("remaining pages: 36 × 5/6 = 30") is not an
error in itself. What counts is whether the computed number is the one the solution needs.

**R2. One error per solution: the first.** Later steps that build on a wrong number are not new
errors, even when they are otherwise right. Do not mark them, and do not let them change the
type.

**R3. Several errors in the first wrong step.** Apply the precedence: planning over binding
over arithmetic. For example, `29 - 3 - 10 - 13 = 13` is the wrong computation *and* wrong
arithmetic (it is 3), so it is **planning**.

**R4. Valid alternatives are not errors.** A different but correct route to the needed quantity
is fine: `8000 - 600 - 200` instead of `600 + 200` and `8000 - 800`, or `a / 5` instead of
`a × 1/5`. So is a redundant step that is right and harmless, for example computing a total that
is never used.

**R5. Unneeded computations.** A calculation the problem does not need is a **planning** error
if its result is used later or changes the answer. A typical case uses the irrelevant sentence:
dividing a total by "the 4 miles to town". If it is right and nothing uses it, R4 applies.

**R6. Extra or missing terms.** If the step should be `3 + 9 + 13` but is `30 + 3 + 9 + 13`, the
computation is a different one: **planning**. **Binding** is a one-for-one swap of a number:
`7200 / 16` where it should be `7200 / 600`.

**R7. A skipped step.** A skipped step is not itself a step you can mark. Mark the first
written step that goes wrong. If that step does the right operation but uses some other number
in place of the one the skipped step would have produced, it is **binding**. Example: it
divides the pie by Theo's 3 servings a day instead of the combined 15. This follows the
parser's definition. If the discussion finds it unworkable, the coordinator records a
clarification in section 7.

**R8. Calculations split over lines.** If a step sets up a calculation and a later step states
its result, the error is in the step where the wrong result first appears. Example:
`[2] 900 - 250 - 175` then `[3] = 485`. Mark 3, **arithmetic** (it is 475). If the answer line is the
first place a wrong value appears, mark `answer`, **arithmetic**.

**R9. The answer line.** Use `answer` only when no numbered step is wrong.
* **planning**: the solution stops before the quantity asked for and reports an intermediate
  result (it finds the eggs left, 100, but never subtracts the 17 that stayed hidden). Also
  planning: the answer line does an extra, unwritten step (the steps end at 120, and the answer
  line gives 120 − 8, using a number from the irrelevant sentence).
* **arithmetic**: rule R8.
* **other**: there is no calculation at all, the answer line has no number ("the problem has
  an error"), or the number contradicts the steps with no visible computation.

**R10. `model_final_answer` can be misread.** It is what the scorer extracted. If the answer
line itself states the gold answer (as in "Answer: 4320 people are infected after 3 days", read
as 3), judge the solution, not the scorer: write `none` and explain in `notes`.

**R11. Rounding and units.** Rounding a result the problem expects to be whole ("1.14 days,
round up to 2") is part of the computation. If the rounding is what makes the answer wrong, it
is the error. If it is the wrong rounding of a right value, it is **arithmetic**. If the problem
never called for rounding and the value was already wrong, mark the earlier step. Forgetting to
convert units (minutes as hours) is **binding**: the right operation with the wrong quantity.

**R12. Unsure?** Pick your best answer, set confidence to 1 and write one line in `notes`.
Never leave a row blank. The agreement script refuses to run until every item in both sheets
has a valid label.

## 6. Worked examples

These are real study traces. None of them is a practice item or one of the items you will
label.

### Arithmetic 1: `gsms-main-41-0|terse_p1|0`

**Problem.** A stack is made out of 8 white tiles, thrice as many orange tiles, and an unknown
number of green tiles. If there are 38 tiles in the stack in total, how many green tiles are
there? *Gold answer 6.*

```
Gold: (1) There are 8*3 = 24 orange tiles in the stack.
      (2) There are 38-8-24 = 6 green tiles in the stack.

[1] Let 'x' be the number of green tiles.
[2] * White tiles: 8
[3] * Orange tiles: 8 * 3 = 24
[4] * Total tiles: 8 + 24 + x = 38
[5] Therefore, x = 38 - 8 - 24 = **8**
[answer line] Answer: 8
```

**first_wrong_step 5, arithmetic.** Steps 3 and 4 are right (4 is a correct equation). Step 5
has the right operation and numbers, `38 − 8 − 24`, but states 8; it is 6.

### Arithmetic 2: `gsms-noop-64-4|terse_p0|0`

**Problem.** Uma has 14 jars of frosting in her cupboard. Each jar of frosting can decorate 19
cookies. Uma wants to bake enough cookies to use up all of her frosting. She bought the jars 3
days ago. If each tray holds 7 cookies, how many trays worth of cookies should she bake?
*Gold answer 38.*

```
Gold: (1) She has enough frosting for 14 * 19 = 266 cookies.
      (2) She needs 266 / 7 = 38 trays to bake all of the cookies.

[1] Here's how to solve the problem:
[2] **1. Calculate total cookies Uma can make:**
[3] * Uma has 14 jars * 19 cookies/jar = 266 cookies
[4] **2. Calculate how many trays she needs:**
[5] * 266 cookies / 7 cookies/tray = 37 trays
[answer line] **Answer: Uma should bake 37 trays of cookies.**
```

**first_wrong_step 5, arithmetic.** 266 / 7 is exactly 38. Lines 1, 2 and 4 are headings, with
no calculation. The irrelevant sentence (3 days) is ignored, as it should be.

### Binding 1: `gsms-noop-46-4|marked_p0|0`

**Problem.** Greenville High School is holding graduation in their Convention Center this year
which has space for 8000 people. The venue has 16 entrances. After accounting for 600 seats for
graduates and 200 seats for faculty attending, how many tickets would each graduate receive to
give to their friends and family if the tickets are split equally? *Gold answer 12.*

```
Gold: (1) Add graduate and faculty seats together. 600 + 200 = 800 seats for faculty and graduates
      (2) Minus seats for faculty and graduates from total seats allowed. 8000 - 800 = 7200 remaining seats.
      (3) Divide remaining seats by the number of graduates. 7200/600 = 12 tickets

[1] Step 1: Calculate the total number of seats available: 8000 people
[2] Step 2: Subtract the seats for graduates and faculty: 8000 - 600 - 200 = 7200 seats
[3] Step 3: Divide the remaining seats by the number of graduates: 7200 / 16 = 450 tickets
[answer line] Answer: 450
```

**first_wrong_step 3, binding.** Step 2 reaches 7200 by a valid alternative route (R4). Step 3
is the right operation (gold step 3), but it divides by the 16 entrances from the irrelevant
sentence instead of the 600 graduates. Its own words even say "number of graduates". This is a
one-for-one swap (R6).

### Binding 2: `gsms-main-37-7|marked_p1|0`

**Problem.** A pie has 15 servings. If Theo can eat 3 servings a day while Nadia can eat four
times as much, how many days will the pie last? *Gold answer 1.*

```
Gold: (1) Nadia can eat 3 x 4 = 12 servings a day.
      (2) Together, Theo and Nadia can eat 3 + 12 = 15 servings a day.
      (3) So, a pie will last for 15/15 = 1 days.

[1] Step 1: Nadia eats 4 * 3 servings = 12 servings per day.
[2] Step 2: The pie will last 15 servings / 3 servings/day = 5 days.
[answer line] Answer: 5
```

**first_wrong_step 2, binding** (R7). The model skips gold step 2, the combined 15 a day.
Step 2 is gold step 3's operation, 15 ÷ (servings a day), but it uses Theo's 3 in place of the
combined 15.

### Planning 1: `gsms-main-54-5|marked_p0|0`

**Problem.** Zahra sliced a whole pear into 30 pieces. She ate 3 slices, her friend ate 6 more
than her, and her sister ate 4 more than her friend. How many slices of pear did they all eat?
*Gold answer 25.*

```
Gold: (1) Her friend ate 3 + 6 = 9 slices.
      (2) Her sister ate 9 + 4 = 13 slices.
      (3) They ate a total of 3 + 9 + 13 = 25 slices.

[1] Step 1: Zahra's friend ate 3 + 6 = 9 slices.
[2] Step 2: Zahra's sister ate 9 + 4 = 13 slices.
[3] Step 3: They all ate a total of 30 + 3 + 9 + 13 = 55 slices.
[answer line] Answer: 55
```

**first_wrong_step 3, planning.** Step 3 adds the whole pear (30) as an extra term. The sum
itself is right (55), so it is not arithmetic. An added term makes it a different computation,
not a swapped number (R6).

### Planning 2: `gsms-main-54-2|marked_p2|0`

**Problem.** Ananya sliced a whole orange into 29 pieces. She ate 3 slices, her cousin ate 7
more than her, and her friend ate 3 more than her cousin. How many slices of orange did they all
eat? *Gold answer 26.*

```
Gold: (1) Her cousin ate 3 + 7 = 10 slices.
      (2) Her friend ate 10 + 3 = 13 slices.
      (3) They ate a total of 3 + 10 + 13 = 26 slices.

[1] Step 1: Ananya's cousin ate 3 + 7 = 10 slices.
[2] Step 2: Ananya's friend ate 10 + 3 = 13 slices.
[3] Step 3: Together they ate 29 - 3 - 10 - 13 = 13 slices.
[answer line] Answer: 13
```

**first_wrong_step 3, planning** (R3). Step 3 has two errors. It computes the slices left
instead of the slices eaten (planning), and `29 − 3 − 10 − 13` is 3, not 13 (arithmetic).
Planning comes first in the precedence.

### Other 1: `gsms-main-77-7|marked_p0|0`

**Problem.** Emma is required to submit a 75-page thesis. She already finished writing 1/3 of
the thesis. How many pages does she have left to write? *Gold answer 50.*

```
Gold: (1) Emma has already written 1/3 of the thesis which is 75 pages x 1/3 = 25 pages.
      (2) So, she still needs to write 75 pages - 25 pages = 50 pages.

[1] Step 1: Calculate the total number of pages in the thesis.
[answer line] Answer: 75 pages
```

**first_wrong_step answer, other** (R9). There is no calculation at all. The answer repeats a
number from the problem.

### Other 2: `gsms-main-80-4|marked_p0|0`

**Problem.** The vending machines sell pretzels for 30 cents and gum for 75 cents. Mohammed
spent €900 and got 5 bags of pretzels and had 9% of his money left in change. How many gum did
he buy? *Gold answer 1090.*

```
Gold: (1) Mohammed got €81.0 in change because 900 x 9/100 = 81
      (2) Mohammed spent €819 because 900 - 81 = 819
      (3) Mohammed spent €1.5 on pretzels because 5 x 0.3 = 1.5
      (4) Mohammed spent 817.5 on gum because 819.0 - 1.5 = 817.5
      (5) Mohammed bought 1090 gum because 817.5 / 0.75 = 1090

[1] Step 1: Convert the amount of money Mohammed spent to euros.
[answer line] Answer: €900
```

**first_wrong_step answer, other.** Step 1 only announces a plan (R1), and nothing is computed.
This pattern, a single announced step followed by an answer copied from the problem, is the
most common kind of **other** in these traces.

### The answer line, planning: `gsms-main-81-3|terse_p0|0`

**Problem.** The Halloween candy hunt team hid 170 eggs. The Mirzakhani twins each found 35
eggs. All the other eggs except 17 were found by their friends. How many eggs did the friends
find? *Gold answer 83.*

```
Gold: (1) The Mirzakhani twins found, 35 * 2 = 70 eggs.
      (2) The number that remained hidden was 170 - 70 = 100 eggs
      (3) Since 17 eggs were not found, this means the friends found 100 - 17 = 83 eggs.

[1] Here's how to solve the problem:
[2] * **Eggs found by the twins:** 2 * 35 = 70 eggs
[3] * **Eggs found by others:** 170 - 70 = 100 eggs
[answer line] * **Answer:** 100 eggs
```

**first_wrong_step answer, planning** (R9). Both calculations are right. The solution stops
one step early and reports the intermediate 100.

## 7. Returning the sheet, and clarifications

When every row is filled in, choose **File > Download > Comma-separated values (.csv)** and
send the file to the coordinator. Keep the `item_id` column and the column names as they are.
The row order does not matter.

The coordinator then runs `main-study/scripts/annotation_agreement.py` on both sheets. Nothing
reads the annotations before both sheets are back.

**Clarifications after the practice discussion** (dated; none yet):

* –
