"""Automated checks of the no-op distractor clauses and the human-review sheet (DECISIONS D34).

For each of the 89 study templates (prereg/S2.md A8), the baseline clause
(`src/rfd/data/distractor.py`, D6) is inserted into the study's 8 instances (the first 8 by
instance number) exactly as the baseline does it: `insert_clause` before the final sentence,
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

Writes docs/review/distractor_clauses.csv (repository root), one row per template, with empty
columns for two human reviewers. Run from main-study/:

    PYTHONPATH=.:../src python scripts/distractor_review.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from harness.config import REPO_ROOT, experiment_config
from harness.datasets import gsm_symbolic
from parsing.gold import gold_program, slots

sys.path.insert(0, str(REPO_ROOT.parent / "src"))
from rfd.data.distractor import (_RESERVED, NOOP_CLAUSES, insert_clause,  # noqa: E402
                                 sample_distractor_value)

OUT = REPO_ROOT.parent / "docs" / "review" / "distractor_clauses.csv"
BASELINE_SEED = 42
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


def check_instance(t: int, row: dict, inst: int) -> dict:
    q = row["question"]
    p0, sl0 = gold_program(q, row["solution"], row["answer"], "gsm_symbolic")
    forbidden = [float(s.value) for s in sl0] + [float(v) for st in p0.steps for v in (st.a, st.b, st.result)]
    d = sample_distractor_value(forbidden, np.random.default_rng(BASELINE_SEED * 1000 + t * 100 + inst))
    clause = NOOP_CLAUSES[t].format(d=d)
    vq = insert_clause(q, clause)
    start = vq.index(clause)
    span = (start, start + len(clause))
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


def main() -> None:
    d = experiment_config()["data"]
    split = json.loads((REPO_ROOT / "data" / "splits" / "gsm_symbolic_main_seed0.json").read_text())
    split_of = {int(t.split("-")[1]): k for k in ("train", "dev", "test") for t in split[k]}
    rows = gsm_symbolic("main", templates=d["study_templates"], instances=d["instances_per_template"])
    by_t: dict[int, list[dict]] = {}
    for r in rows:
        by_t.setdefault(int(r["template_id"].split("-")[1]), []).append(r)
    out, summary = [], {c: 0 for c in CHECKS}
    n_clauses = 0
    for t in d["study_templates"]:
        insts = by_t[t]
        base = {"template_id": t, "split": split_of[t], "instances_checked": len(insts)}
        if t not in NOOP_CLAUSES:
            out.append({**base, "clause_template": "", "auto_pass": "NO CLAUSE",
                        "auto_notes": "no clause was written for this template (baseline excluded it, D2/D6)",
                        "original_problem": insts[0]["question"]})
            continue
        n_clauses += 1
        res = [check_instance(t, r, int(r["problem_id"].rsplit("-", 1)[1])) for r in insts]
        first = res[0]
        row = {**base, "clause_template": NOOP_CLAUSES[t], "instance_shown": insts[0]["problem_id"],
               "d_shown": first["d"], "clause_rendered": first["clause"],
               "original_problem": insts[0]["question"], "problem_with_clause": first["variant"],
               "insertion_char_offset": first["offset"], "sentence_before_clause": first["before"],
               "sentence_after_clause": first["after"],
               "d_values_all_instances": " ".join(str(x["d"]) for x in res)}
        notes = []
        for c in CHECKS:
            k = sum(x["checks"][c] for x in res)
            row[f"check_{c}"] = f"{'PASS' if k == len(res) else 'FAIL'} ({k}/{len(res)})"
            summary[c] += k == len(res)
            if k < len(res):
                notes.append(f"{c} fails on " + ", ".join(insts[i]["problem_id"] for i, x in enumerate(res)
                                                         if not x["checks"][c]))
        if not all(x["gold_ok"] for x in res):
            notes.append("no gold program for " + ", ".join(insts[i]["problem_id"] for i, x in enumerate(res)
                                                           if not x["gold_ok"]) + " (checked on answer only)")
        if re.match(r"\d", first["clause"]):
            notes.append("style: the sentence starts with a numeral")
        row["auto_pass"] = "PASS" if all(row[f"check_{c}"].startswith("PASS") for c in CHECKS) else "FAIL"
        row["auto_notes"] = "; ".join(notes)
        out.append(row)
    cols = (["template_id", "split", "clause_template", "clause_rendered", "instance_shown", "d_shown",
             "original_problem", "problem_with_clause", "insertion_char_offset", "sentence_before_clause",
             "sentence_after_clause", "instances_checked", "d_values_all_instances"]
            + [f"check_{c}" for c in CHECKS] + ["auto_pass", "auto_notes"] + REVIEW_COLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c, "") for c in cols})
    n_pass = sum(r["auto_pass"] == "PASS" for r in out)
    print(f"{len(out)} templates, {n_clauses} with a clause; all checks pass on {n_pass}; "
          f"per check (templates passing on all instances): {summary}")
    print("FAIL:", [(r["template_id"], r["auto_notes"]) for r in out if r["auto_pass"] == "FAIL"])
    print("no clause:", [r["template_id"] for r in out if r["auto_pass"] == "NO CLAUSE"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
