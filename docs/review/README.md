# Human review of the distractor clauses (DECISIONS D34)

`distractor_clauses.csv` has one row for each of the 89 study templates, each with a no-op
clause written by the AI assistant and never reviewed by a person. 88 come from the baseline
(D6). **Template 50's clause is new (2026-11-13, A17):** it is a test template, and without a
clause only 39 test templates would have the distractor condition. Study generation waits until
two group members have reviewed every row.

**Already checked automatically, for each template's 8 study instances**
(`main-study/scripts/distractor_review.py`, columns `check_*`):
- the clause holds exactly one number;
- the gold program and answer are unchanged, and the gold program never uses that number;
- the number equals no number in the problem and no gold value;
- the clause is one well-formed sentence, inserted at a sentence boundary before the final
  question or instruction.

All 89 pass.

**What only a person can judge.** Two reviewers work independently, then agree.
1. **Is it really a no-op?** Could a careful reader think the clause changes the answer?
   Examples: a clause about a quantity the question asks about; "spare cups" in a problem that
   measures water in cups (template 5); a subset of an item the question counts. If yes, write
   `no` and say why.
2. **Is it grammatical and natural** in place (read `problem_with_clause`)? Does what it refers
   to exist in the problem ("the remoras", "the boxes sold on Tuesday")? Note any tense that
   does not match the problem.
3. Fill in `reviewerN_initials`, `reviewerN_is_noop`, `reviewerN_grammatical` and `reviewerN_notes`.
   Then fill in `agreed_action` (keep / rewrite / drop) together, with a rewrite in the notes.

**Time:** about 1 minute per row, so **about 1.5 hours per reviewer** for 89 rows. The two
reviewers can work in parallel. Allow another **20–30 minutes** together to settle
disagreements. A rewritten clause is re-run through the script before it is used.

**These clauses define the primary population (A14, D35).** Every instance is generated with
and without its clause.

**After the review** (D39):
1. Write FIX in `reviewerN_verdict` for any clause you want changed (a "no" answer counts as
   FIX too).
2. After the joint discussion, put every agreed rewrite in `clause_rewrites.csv`, with columns
   `template_id,new_clause_template,reason`. Keep `{d}` where the number goes.
3. Fill in `agreed_action`.
4. Run `PYTHONPATH=. python scripts/review_agreement.py` from `main-study/`, or pass each
   reviewer's own copy with `--sheets A.csv B.csv`. It computes Cohen's kappa, re-checks every
   rewrite and writes the reviewed table that the pipeline reads.
