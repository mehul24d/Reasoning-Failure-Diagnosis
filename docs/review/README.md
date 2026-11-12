# Human review of the distractor clauses (DECISIONS D34)

`distractor_clauses.csv` has one row for each of the 89 study templates. 88 have a no-op clause,
written by the AI assistant (D6) and never reviewed by a person. Template 50 has none. Study
generation waits until two group members have reviewed every row.

**Already checked automatically, for each template's 8 study instances**
(`main-study/scripts/distractor_review.py`, columns `check_*`):
- the clause holds exactly one number;
- the gold program and answer are unchanged, and the gold program never uses that number;
- the number equals no number in the problem and no gold value;
- the clause is one well-formed sentence, inserted at a sentence boundary before the final
  question or instruction.

All 88 pass.

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

**Time:** about 1 minute per row, so **about 1.5 hours per reviewer** for 88 rows. The two
reviewers can work in parallel. Allow another **20–30 minutes** together to settle
disagreements. A rewritten clause is re-run through the script before it is used.

Under the S2 sign-off (D29), the S2 population inserts no clause. The review is needed before
any distractor population is generated: that is the plan's preferred primary population, if
it is rebuilt from these clauses.
