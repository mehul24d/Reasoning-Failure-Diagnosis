"""Reconcile the two-person review of the distractor clauses (prereg/S2.md A17, A18; DECISIONS D39, D40).

Inputs (repository root docs/review/):
  * `human_review_final.csv` (A18): the FINAL clauses (harness.distractor.FINAL_CLAUSES), one
    instance each, with reviewer1_is_noop, reviewer1_grammatical, reviewer2_is_noop,
    reviewer2_grammatical (yes/no) and notes. Or one copy per reviewer with --sheets (reviewer
    1's copy first; in a personal copy the reviewer's columns may be named reviewer1_*).
  * `clause_rewrites.csv` (optional): template_id, action (rewrite / drop), new_clause_template,
    reason. The changes agreed in the joint discussion, one for every clause that either
    reviewer answered "no" on (a rewrite or a drop).

A reviewer's verdict on a row is FIX if they answered "no" to is-it-a-no-op or to
is-it-grammatical (or wrote FIX in an optional reviewerN_verdict column); otherwise OK. Cohen's
kappa is computed on each of the two questions and on the verdict. Every clause changed after
the review is re-run through ALL the automated checks of scripts/distractor_review.py on
instances 1-16 (including the partitive, grounding and independent-solver checks); one failure
stops the script. A changed clause's temptation is set to UNRATED in data/distractor_temptation.json
until it is rated.

Outputs: data/distractor_clauses_reviewed.json (read by the `problems` stage),
reports/m1/distractor_review_agreement.{json,md}.

    PYTHONPATH=. python scripts/review_agreement.py [--sheets A.csv B.csv]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from analysis.stats import cohens_kappa
from harness.config import REPO_ROOT, experiment_config
from harness.distractor import DRAFT_CLAUSES, FINAL_CLAUSES, REVIEWED, TEMPTATION

REVIEW = REPO_ROOT.parent / "docs" / "review"
OUT_JSON = REPO_ROOT / "reports" / "m1" / "distractor_review_agreement.json"
YES, NO = {"yes", "y", "ok", "keep", "true", "1"}, {"no", "n", "fix", "false", "0"}


def _review_module():
    spec = importlib.util.spec_from_file_location("distractor_review", REPO_ROOT / "scripts" / "distractor_review.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _col(row: dict, n: int, key: str, personal: bool = False) -> str:
    """Reviewer n's value for `key`, from reviewerN_* (or, in a personal copy, reviewer1_*)."""
    for prefix in ((f"reviewer{n}_", "reviewer1_") if personal else (f"reviewer{n}_",)):
        for c, v in row.items():
            if c and c.startswith(prefix + key) and (v or "").strip():
                return v.strip()
    return ""


def _yn(v: str) -> bool | None:
    v = v.strip().lower()
    return True if v in YES else False if v in NO else None


def reviewer_labels(rows: list[dict], n: int, personal: bool = False) -> dict[int, dict]:
    out, missing = {}, []
    for r in rows:
        t = int(r["template_id"])
        noop, gram = _yn(_col(r, n, "is_noop", personal)), _yn(_col(r, n, "grammatical", personal))
        verdict = _col(r, n, "verdict", personal).upper()
        if noop is None or gram is None:
            missing.append(t)
            continue
        fix = verdict == "FIX" or not noop or not gram
        out[t] = {"is_noop": noop, "grammatical": gram, "verdict": "FIX" if fix else "OK",
                  "initials": _col(r, n, "initials", personal), "notes": _col(r, n, "notes", personal)}
    if missing:
        raise SystemExit(f"reviewer {n}: no yes/no answers for templates {missing}; the review is not complete")
    return out


def agreement(r1: dict, r2: dict) -> dict:
    ts = sorted(set(r1) & set(r2))
    res = {"items": len(ts)}
    for k in ("verdict", "is_noop", "grammatical"):
        a, b = [r1[t][k] for t in ts], [r2[t][k] for t in ts]
        res[k] = {"kappa": cohens_kappa(a, b), "percent_agreement": sum(x == y for x, y in zip(a, b)) / len(ts),
                  "reviewer1_fix_or_no": sum(x in ("FIX", False) for x in a),
                  "reviewer2_fix_or_no": sum(x in ("FIX", False) for x in b)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", nargs=2, default=None)
    ap.add_argument("--rewrites", default=str(REVIEW / "clause_rewrites.csv"))
    a = ap.parse_args()
    sheets = a.sheets or [REVIEW / "human_review_final.csv"] * 2
    rows1 = list(csv.DictReader(open(sheets[0])))
    rows2 = list(csv.DictReader(open(sheets[1])))
    r1, r2 = reviewer_labels(rows1, 1, bool(a.sheets)), reviewer_labels(rows2, 2, bool(a.sheets))
    flagged = sorted(t for t in r1 if r1[t]["verdict"] == "FIX" or r2[t]["verdict"] == "FIX")
    changes_in = {}
    if Path(a.rewrites).exists():
        changes_in = {int(r["template_id"]): r for r in csv.DictReader(open(a.rewrites))}
    missing = [t for t in flagged if t not in changes_in]
    if missing:
        raise SystemExit(f"templates marked FIX with no agreed change (rewrite or drop) in {a.rewrites}: {missing}")

    dr = _review_module()
    exp = experiment_config()["data"]
    from harness.datasets import gsm_symbolic

    rewrites = {t: c for t, c in changes_in.items() if (c.get("action") or "rewrite").strip().lower() == "rewrite"}
    insts = {}
    if rewrites:
        for r in gsm_symbolic("main", templates=sorted(rewrites), instances=exp["clause_check_instances"]):
            insts.setdefault(int(r["template_id"].split("-")[1]), []).append(r)
    clauses, changes = {}, []
    for t in exp["study_templates"]:
        c = changes_in.get(t)
        act = (c.get("action") or "rewrite").strip().lower() if c else "keep"
        if act == "drop":
            clauses[str(t)] = {"action": "drop", "clause_template": None, "reviewed": FINAL_CLAUSES[t]}
            changes.append({"template_id": t, "action": "drop", "from": FINAL_CLAUSES[t], "reason": c.get("reason", "")})
        elif act == "rewrite":
            new = c["new_clause_template"].strip()
            row, _ = dr.check_template(t, insts[t], new, rewritten=True)
            if row["auto_pass"] != "PASS":
                raise SystemExit(f"rewritten clause for template {t} fails the automated checks: {row['auto_notes']}")
            clauses[str(t)] = {"action": "rewrite", "clause_template": new, "reviewed": FINAL_CLAUSES[t],
                               "reason": c.get("reason", ""),
                               "auto_checks": f"PASS ({len(insts[t])}/{len(insts[t])} on all checks)"}
            changes.append({"template_id": t, "action": "rewrite", "from": FINAL_CLAUSES[t], "to": new,
                            "reason": c.get("reason", "")})
        else:
            clauses[str(t)] = {"action": "keep", "clause_template": FINAL_CLAUSES[t]}
        clauses[str(t)]["draft"] = DRAFT_CLAUSES[t]
    ag = agreement(r1, r2)
    rep = {"agreement": ag, "flagged_fix_by_either": flagged, "changes": changes,
           "sheets_sha256": [hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sheets],
           "kappa_min": experiment_config()["thresholds"]["kappa_min"]}
    REVIEWED.parent.mkdir(parents=True, exist_ok=True)
    REVIEWED.write_text(json.dumps({"source": "two-person review of the final clauses, DECISIONS D39/D40",
                                    "clauses": clauses, "agreement": ag}, indent=1))
    if TEMPTATION.exists() and changes:
        tt = json.loads(TEMPTATION.read_text())
        for ch in changes:
            e = tt["templates"].setdefault(str(ch["template_id"]), {})
            e.update({"rating": "UNRATED" if ch["action"] == "rewrite" else "DROPPED",
                      "source": f"clause {ch['action']} after the human review"})
        TEMPTATION.write_text(json.dumps(tt, indent=1))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2))
    md = ["# Distractor-clause review: agreement and changes", "",
          f"{ag['items']} final clauses (A18), two reviewers.", "",
          "| Question | Cohen's kappa | agreement | reviewer 1 no/FIX | reviewer 2 no/FIX |", "|---|---|---|---|---|"]
    for k in ("is_noop", "grammatical", "verdict"):
        kk = ag[k]["kappa"]
        md.append(f"| {k} | {'n/a' if kk is None else f'{kk:.3f}'} | {ag[k]['percent_agreement']:.3f} | "
                  f"{ag[k]['reviewer1_fix_or_no']} | {ag[k]['reviewer2_fix_or_no']} |")
    md += ["", f"Marked FIX by either reviewer: {flagged or 'none'}.", ""]
    md += [f"* {c['template_id']}: {c['action']}" + (f": \"{c['from']}\" -> \"{c['to']}\"" if c["action"] == "rewrite"
           else "") + (f" ({c['reason']})" if c.get("reason") else "") for c in changes]
    OUT_JSON.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
