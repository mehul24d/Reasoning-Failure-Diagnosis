# Human review of the distractor clauses (DECISIONS D34, D40)

**The sheet to fill in: `human_review_final.csv`** (one row per study template, 89 rows).

Each row shows one problem without and with its no-op clause (`original_problem`,
`problem_with_clause`). `rewritten` says whether the clause was rewritten on 2026-11-13, before
this review. The row has no AI verdicts. The clauses are those of the study
(`harness.distractor.FINAL_CLAUSES`); study generation waits until two group members have
reviewed every row.

**Already checked automatically, on instances 1–16 of every template** (`distractor_checks_final.csv`;
`main-study/scripts/distractor_review.py`). All 89 pass every check:
- **The number:** the clause holds exactly one number. It equals no number in the problem and
  no gold value.
- **The gold solution:** the gold program and answer are unchanged, and the gold program never
  uses the clause's number.
- **The sentence:** one well-formed sentence, inserted at a sentence boundary before the final
  question or instruction.
- **Partitive clauses ("d of the X"):** d is smaller than the number of X in every instance.
- **Grounding:**
  - the clause names nothing that only some instances have (GSM-Symbolic changes names,
    objects, places and days between instances);
  - a he/she matches the problem;
  - the clause does not mention a story word before the problem introduces it.
- **Rewritten clauses:** an independent solver gives the same answer with and without the
  clause, on every instance.

**What only a person can judge.** Two reviewers work independently.
1. **`reviewerN_is_noop` (yes/no):** is it really a no-op? Could a careful reader think the
   clause changes the answer, e.g. because it is about a quantity the question asks about? If
   not sure, write `no` and say why in `notes`.
2. **`reviewerN_grammatical` (yes/no):** is it grammatical and natural in place? Does what it
   refers to exist in the problem, and does its tense match? A plausible size for the number?
   (The number changes between instances, from 2 to 19.)
3. Put anything else in `notes`, with your initials.

Reviewer 1 fills the `reviewer1_*` columns and reviewer 2 the `reviewer2_*` columns. Do not
look at the other reviewer's answers until both are done; separate copies passed with
`--sheets` are fine.

**Time:** about 1 minute per row, so about 1.5 hours per reviewer, and 20–30 minutes together
to settle disagreements.

**After the review:**
1. For every row where either reviewer wrote `no`, agree a change in the joint discussion. Put
   it in `clause_rewrites.csv` with the columns `template_id,action,new_clause_template,reason`:
   `action` is `rewrite` (keep `{d}` where the number goes) or `drop`.
2. Run `PYTHONPATH=. python scripts/review_agreement.py` from `main-study/`, or with
   `--sheets A.csv B.csv` for separate copies. It does three things:
   - computes Cohen's kappa per question;
   - re-runs every rewrite through all the checks on instances 1–16. A rewrite of a template
     without an independent solver fails until one is added in `harness/clause_solvers.py`;
   - writes the reviewed table that the pipeline reads.

**Other files here:**
- `distractor_checks_final.csv`: every automated check per template, with the draft clause
  and the reason for each rewrite.
- `distractor_clauses_ai_prescreen*.csv`: the AI pre-screen of the drafts. Its temptation
  rating is kept only as a covariate (D41).
- `distractor_clauses.csv`: the earlier sheet of the draft clauses, which showed instance 1
  only. It is kept as a record and is no longer used.
