"""Automated checks of the no-op distractor clauses and the human-review sheet (DECISIONS D34).

For each of the 89 study templates (prereg/S2.md A8), the study's clause
(`harness/distractor.py`: the baseline's clauses of `src/rfd/data/distractor.py`, D6, plus the
clause drafted for template 50, A17) is inserted into the study's 8 instances (the first 8 by
instance number) exactly as the pipeline does it: `insert_clause` before the final sentence,
the distractor value d drawn by `sample_distractor_value` with the baseline's seed rule
(42 * 1000 + template * 100 + instance) and its forbidden values (question numbers and gold
operands/results, plus the unit constants 7, 10, 12). Checks, per instance:

  contains_number        the rendered clause holds exactly one number, d, and parser v4 marks
                         exactly one question slot as the distractor (value d)
  gold_answer_unchanged  the gold program of the variant gives the same answer
  gold_program_unchanged same steps (op, operands, result), same ok status, and no operand
                         resolves to the distractor slot
  d_no_collision         d equals no question number, gold operand or result, and no unit constant
  sentence_boundary      inserted right after ". ", "! " or "? " (or at the start)
  clause_one_sentence    capitalised (or a numeral), ends with ".", one sentence, no "{" left
  final_question         the text after the clause is one sentence (titles like "Mrs." are not
                         breaks): a question ending in "?", or an instruction ("calculate ...")

Amendment A18 (2026-11-13): the clauses checked are `FINAL_CLAUSES` (the drafts with the
pre-review rewrites of `PREREVIEW_REWRITES`), on instances 1-16 (`data.clause_check_instances`:
the study's 8 and the top-up instances 9-16), with these further checks:

  partitive_size      a partitive clause ("{d} of the X") has d < the size of X in the instance
                      (harness/instance_checks.py)
  solver_same_answer  rewritten clauses: an independent solver (harness/clause_solvers.py) gives
                      the dataset answer on the question with and without the clause
  variable_entity, pronoun_gender, no_forward_ref
                      grounding checks over all instances of the template (instance_checks.py)

Writes docs/review/distractor_checks_final.csv (every check, per template),
docs/review/human_review_final.csv (the reviewers' sheet: no AI verdicts) and
reports/m1/distractor_checks_final.json. Exits with status 1 if any check fails. The pre-A18
sheet (docs/review/distractor_clauses.csv, drafts on instances 1-8) is kept unchanged. Run from
main-study/:

    PYTHONPATH=. python scripts/distractor_review.py
"""
from __future__ import annotations

import csv
import json
import re
from fractions import Fraction
from pathlib import Path

from harness.config import REPO_ROOT, experiment_config
from harness.datasets import gsm_symbolic
from harness.clause_solvers import SOLVERS, solve
from harness.distractor import (_RESERVED, DRAFT_CLAUSES, FINAL_CLAUSES, PREREVIEW_REWRITES, REWRITE_TEMPTATION_AI,
                                TEMPTATION, insert_clause, make_variant)
from harness.instance_checks import grounding_check, partitive_check
from parsing.gold import gold_program, slots

NOOP_CLAUSES = FINAL_CLAUSES        # A18: the drafts (88 baseline + template 50) with the pre-review rewrites

REVIEW = REPO_ROOT.parent / "docs" / "review"
OUT = REVIEW / "distractor_checks_final.csv"            # every automated check, per template
HUMAN = REVIEW / "human_review_final.csv"                # the sheet the two reviewers fill in
OUT_JSON = REPO_ROOT / "reports" / "m1" / "distractor_checks_final.json"
PRESCREEN = REVIEW / "distractor_clauses_ai_prescreen.csv"
INSTANCE_CHECKS = ["partitive_size", "solver_same_answer"]        # per instance (A18)
TEMPLATE_CHECKS = ["variable_entity", "pronoun_gender", "no_forward_ref"]   # over all instances (A18)
NUM = re.compile(r"\d+(?:\.\d+)?")
CHECKS = ["contains_number", "gold_answer_unchanged", "gold_program_unchanged", "d_no_collision",
          "sentence_boundary", "clause_one_sentence", "final_question"]
REVIEW_COLS = ["reviewer1_initials", "reviewer1_is_noop (yes/no)", "reviewer1_grammatical (yes/no)",
               "reviewer1_notes", "reviewer2_initials", "reviewer2_is_noop (yes/no)",
               "reviewer2_grammatical (yes/no)", "reviewer2_notes", "agreed_action (keep/rewrite/drop)"]


_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr)\.")
_IMPERATIVE = re.compile(r"\b(?:calculate|find|determine|compute|work out)\b", re.I)


def _final_sentence_ok(text: str) -> bool:
    """One sentence (titles such as "Mrs." are not breaks), and either a question ending in "?"
    or an instruction ending in "." ("... calculate the number of ...")."""
    body = _ABBREV.sub("Abbr", text)
    one = not re.search(r"[.!?]\s+\S", body[:-1])
    return one and (body.endswith("?") or (body.endswith(".") and bool(_IMPERATIVE.search(body))))


def _steps(prog, sl):
    """Steps with slot references replaced by the slot's value and position class, so a shift
    of slot indices caused by the inserted number does not count as a change."""
    def ref(r):
        if r[0] == "slot":
            return ("slot", sl[r[1]].value)
        return tuple(r)
    return [(s.op, s.a, s.b, s.result, ref(s.a_ref), ref(s.b_ref)) for s in prog.steps]


def check_instance(t: int, row: dict, inst: int, clause_template: str | None = None) -> dict:
    """Checks on one instance, on exactly the text the pipeline generates (harness.distractor)."""
    q = row["question"]
    p0, sl0 = gold_program(q, row["solution"], row["answer"], "gsm_symbolic")
    v = make_variant(t, inst, row, clause_template or NOOP_CLAUSES[t])
    d, clause, vq = v["d"], v["clause"], v["question"]
    start = v["distractor_span"][0]
    span = tuple(v["distractor_span"])
    p1, sl1 = gold_program(vq, row["solution"], row["answer"], "gsm_symbolic", distractor_span=span)
    dslots = [s for s in sl1 if s.distractor]
    before, after = vq[:start], vq[span[1]:]
    nums_in_clause = NUM.findall(clause)
    uses_d = any(r[0] == "slot" and sl1[r[1]].distractor for s in p1.steps for r in (s.a_ref, s.b_ref))
    gold_vals = {s.value for s in sl0} | {v for st in p0.steps for v in (st.a, st.b, st.result)}
    res = {
        "contains_number": nums_in_clause == [str(d)] and len(dslots) == 1 and dslots[0].value == d,
        "gold_answer_unchanged": p1.answer == p0.answer and Fraction(row["answer"]) == p0.answer
                                 if p0.ok else p1.answer == p0.answer,
        "gold_program_unchanged": p1.ok == p0.ok and _steps(p1, sl1) == _steps(p0, sl0) and not uses_d,
        "d_no_collision": Fraction(d) not in gold_vals and float(d) not in _RESERVED,
        "sentence_boundary": start == 0 or (before.endswith(" ") and before.rstrip()[-1:] in ".!?"),
        "clause_one_sentence": bool(re.match(r"[A-Z0-9]", clause)) and clause.endswith(".")
                               and not re.search(r"[.!?]\s+\S", clause[:-1]) and "{" not in clause,
        "final_question": after.startswith(" ") and _final_sentence_ok(after.strip()),
    }
    return {"d": d, "clause": clause, "variant": vq, "offset": start, "checks": res,
            "gold_ok": p0.ok, "before": before.rstrip().rsplit(". ", 1)[-1], "after": after.strip()}


def check_template(t: int, insts: list[dict], clause: str, rewritten: bool, split: str = "") -> tuple[dict, dict]:
    """Every automated check of `clause` on the template's instances: (CSV row, first instance).
    `rewritten` requires the independent solver check (harness/clause_solvers.py)."""
    all_checks = CHECKS + INSTANCE_CHECKS + TEMPLATE_CHECKS
    res = [check_instance(t, r, int(r["problem_id"].rsplit("-", 1)[1]), clause) for r in insts]
    for r, x in zip(insts, res):
        pc = partitive_check(t, clause, r["question"], r["answer"], x["d"])
        x["checks"]["partitive_size"] = pc["ok"]
        x["partitive"] = pc
        if rewritten:
            try:
                gold = Fraction(r["answer"])
                a0, a1 = solve(t, r["question"]), solve(t, x["variant"])
                x["checks"]["solver_same_answer"] = abs(a0 - gold) <= Fraction(1, 100) and a1 == a0
                x["solver"] = (str(a0), str(a1))
            except Exception as e:                          # a solver that cannot read the text fails
                x["checks"]["solver_same_answer"] = False
                x["solver"] = (f"error: {e}", "")
        else:
            x["checks"]["solver_same_answer"] = True        # not required for kept clauses
    gr = grounding_check(clause, [r["question"] for r in insts], insert_clause)
    first = res[0]
    row = {"template_id": t, "split": split, "instances_checked": len(insts),
           "clause_template": clause, "rewritten": "yes" if rewritten else "no",
           "draft_clause_template": DRAFT_CLAUSES[t],
           "rewrite_reason": PREREVIEW_REWRITES[t][1] if t in PREREVIEW_REWRITES else "",
           "instance_shown": insts[0]["problem_id"], "d_shown": first["d"], "clause_rendered": first["clause"],
           "original_problem": insts[0]["question"], "problem_with_clause": first["variant"],
           "insertion_char_offset": first["offset"], "sentence_before_clause": first["before"],
           "sentence_after_clause": first["after"],
           "d_values_all_instances": " ".join(str(x["d"]) for x in res),
           "partitive_sizes": " ".join(str(x["partitive"]["size"]) for x in res)
                              if first["partitive"]["partitive"] else "not partitive",
           "solver_answers": " ".join(x["solver"][1] for x in res) if rewritten else ""}
    notes = []
    for c in CHECKS + INSTANCE_CHECKS:
        k = sum(x["checks"][c] for x in res)
        row[f"check_{c}"] = f"{'PASS' if k == len(res) else 'FAIL'} ({k}/{len(res)})"
        if k < len(res):
            notes.append(f"{c} fails on " + ", ".join(insts[i]["problem_id"] for i, x in enumerate(res)
                                                     if not x["checks"][c]))
    for c in TEMPLATE_CHECKS:
        row[f"check_{c}"] = "PASS" if gr[c] else "FAIL"
    if gr["note"]:
        notes.append(gr["note"])
    if not all(x["gold_ok"] for x in res):
        notes.append("no gold program for " + ", ".join(insts[i]["problem_id"] for i, x in enumerate(res)
                                                       if not x["gold_ok"]) + " (checked on answer only)")
    if re.match(r"\d", first["clause"]):
        notes.append("style: the sentence starts with a numeral")
    row["auto_pass"] = "PASS" if all(row[f"check_{c}"].startswith("PASS") for c in all_checks) else "FAIL"
    row["auto_notes"] = "; ".join(notes)
    return row, first


def ai_prescreen() -> dict[int, dict]:
    if not PRESCREEN.exists():
        return {}
    return {int(r["template_id"]): r for r in csv.DictReader(open(PRESCREEN))}


def main() -> None:
    d = experiment_config()["data"]
    n_inst = d["clause_check_instances"]
    split = json.loads((REPO_ROOT / "data" / "splits" / "gsm_symbolic_main_seed0.json").read_text())
    split_of = {int(t.split("-")[1]): k for k in ("train", "dev", "test") for t in split[k]}
    rows = gsm_symbolic("main", templates=d["study_templates"], instances=n_inst)
    by_t: dict[int, list[dict]] = {}
    for r in rows:
        by_t.setdefault(int(r["template_id"].split("-")[1]), []).append(r)
    pre = ai_prescreen()
    all_checks = CHECKS + INSTANCE_CHECKS + TEMPLATE_CHECKS
    out, human, summary = [], [], {c: 0 for c in all_checks}
    for t in d["study_templates"]:
        insts = by_t[t]
        rewritten = t in PREREVIEW_REWRITES
        row, first = check_template(t, insts, NOOP_CLAUSES[t], rewritten, split_of[t])
        for c in all_checks:
            summary[c] += row[f"check_{c}"].startswith("PASS")
        p = pre.get(t, {})
        row["ai_tempting_prescreen"] = p.get("ai_tempting", "")
        out.append(row)
        human.append({"template_id": t, "split": split_of[t], "original_problem": insts[0]["question"],
                      "problem_with_clause": first["variant"], "rewritten": "yes" if rewritten else "no",
                      "reviewer1_is_noop": "", "reviewer1_grammatical": "", "reviewer2_is_noop": "",
                      "reviewer2_grammatical": "", "notes": ""})
    cols = list(out[0])
    for path, rs, cs in ((OUT, out, cols), (HUMAN, human, list(human[0]))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cs)
            w.writeheader()
            w.writerows(rs)
    fails = [(r["template_id"], r["auto_notes"]) for r in out if r["auto_pass"] == "FAIL"]
    rep = {"instances_checked_per_template": n_inst, "templates": len(out),
           "rewritten": sorted(PREREVIEW_REWRITES), "n_rewritten": len(PREREVIEW_REWRITES),
           "templates_passing_each_check": summary, "all_pass": not fails, "failures": fails,
           "partitive_final": [r["template_id"] for r in out if r["partitive_sizes"] != "not partitive"]}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2))
    # A19: temptation covariate per template (rating of the clause actually used)
    tempt = {}
    for t in d["study_templates"]:
        if t in PREREVIEW_REWRITES:
            tempt[str(t)] = {"rating": REWRITE_TEMPTATION_AI[t], "source": "AI re-rating of the rewritten clause",
                             "prescreen_rating_of_draft": pre.get(t, {}).get("ai_tempting", "")}
        else:
            tempt[str(t)] = {"rating": pre.get(t, {}).get("ai_tempting", ""), "source": "AI pre-screen"}
    TEMPTATION.parent.mkdir(parents=True, exist_ok=True)
    TEMPTATION.write_text(json.dumps({"role": "covariate only (A19); never used to exclude templates",
                                      "scale": "PASS = tempting, WEAK = little pull, UNSURE",
                                      "templates": tempt}, indent=1))
    print(f"{len(out)} templates ({len(PREREVIEW_REWRITES)} rewritten), {n_inst} instances each; all checks "
          f"pass on {len(out) - len(fails)}; per check (templates passing on all instances): {summary}")
    print("FAIL:", fails)
    print(f"wrote {OUT}, {HUMAN}, {OUT_JSON}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
