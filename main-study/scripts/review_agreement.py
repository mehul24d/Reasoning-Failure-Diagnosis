"""Reconcile the two-person review of the distractor clauses (prereg/S2.md A17, DECISIONS D39).

Inputs (repository root docs/review/):
  * the reviewers' sheets: either `distractor_clauses.csv` with both reviewers' columns filled,
    or one copy per reviewer passed with --sheets (reviewer 1's copy first). In each copy the
    reviewer's columns are `reviewerN_*` (or `reviewer1_*` in a personal copy), and an optional
    `reviewerN_verdict` column (OK / FIX).
  * `clause_rewrites.csv`: template_id, new_clause_template, reason. Rewrites agreed in the
    joint discussion, one for every clause that either reviewer marked FIX (unless the agreed
    action is drop).

A reviewer's verdict on a row is FIX if they wrote FIX, or answered "no" to is-it-a-no-op or
to is-it-grammatical; otherwise OK. Cohen's kappa is computed on the verdict and on each of the
two questions. Every rewritten clause is re-run through the automated checks
(scripts/distractor_review.py) on the template's 8 study instances; one failure stops the script.

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
from harness.distractor import DRAFT_CLAUSES, REVIEWED

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
    sheets = a.sheets or [REVIEW / "distractor_clauses.csv"] * 2
    rows1 = list(csv.DictReader(open(sheets[0])))
    rows2 = list(csv.DictReader(open(sheets[1])))
    r1, r2 = reviewer_labels(rows1, 1, bool(a.sheets)), reviewer_labels(rows2, 2, bool(a.sheets))
    agreed = {int(r["template_id"]): (r.get("agreed_action (keep/rewrite/drop)") or "").strip().lower()
              for r in rows1}
    flagged = sorted(t for t in r1 if r1[t]["verdict"] == "FIX" or r2[t]["verdict"] == "FIX")
    rewrites = {}
    if Path(a.rewrites).exists():
        rewrites = {int(r["template_id"]): r for r in csv.DictReader(open(a.rewrites))}
    missing = [t for t in flagged if t not in rewrites and agreed.get(t) != "drop"]
    if missing:
        raise SystemExit(f"templates marked FIX with no agreed rewrite (and not dropped): {missing}")

    dr = _review_module()
    exp = experiment_config()["data"]
    from harness.datasets import gsm_symbolic

    insts = {}
    for r in gsm_symbolic("main", templates=sorted(rewrites), instances=exp["instances_per_template"]):
        insts.setdefault(int(r["template_id"].split("-")[1]), []).append(r)
    clauses, changes = {}, []
    for t in exp["study_templates"]:
        if agreed.get(t) == "drop" and t not in rewrites:
            clauses[str(t)] = {"action": "drop", "clause_template": None, "original": DRAFT_CLAUSES[t]}
            changes.append({"template_id": t, "action": "drop", "from": DRAFT_CLAUSES[t]})
            continue
        if t in rewrites:
            new = rewrites[t]["new_clause_template"].strip()
            res = [dr.check_instance(t, r, int(r["problem_id"].rsplit("-", 1)[1]), new) for r in insts[t]]
            bad = {c: sum(not x["checks"][c] for x in res) for c in dr.CHECKS}
            if any(bad.values()):
                raise SystemExit(f"rewritten clause for template {t} fails the automated checks: {bad}")
            clauses[str(t)] = {"action": "rewrite", "clause_template": new, "original": DRAFT_CLAUSES[t],
                               "reason": rewrites[t].get("reason", ""), "auto_checks": "PASS (8/8 on all checks)"}
            changes.append({"template_id": t, "action": "rewrite", "from": DRAFT_CLAUSES[t], "to": new,
                            "reason": rewrites[t].get("reason", "")})
        else:
            clauses[str(t)] = {"action": "keep", "clause_template": DRAFT_CLAUSES[t]}
    ag = agreement(r1, r2)
    rep = {"agreement": ag, "flagged_fix_by_either": flagged, "changes": changes,
           "reviewers": {"1": sorted({x["initials"] for x in r1.values()} - {""}),
                         "2": sorted({x["initials"] for x in r2.values()} - {""})},
           "sheets_sha256": [hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sheets],
           "kappa_min": experiment_config()["thresholds"]["kappa_min"]}
    REVIEWED.parent.mkdir(parents=True, exist_ok=True)
    REVIEWED.write_text(json.dumps({"source": "two-person review, DECISIONS D39", "clauses": clauses,
                                    "agreement": ag}, indent=1))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2))
    md = ["# Distractor-clause review: agreement and changes", "",
          f"{ag['items']} clauses, two reviewers.", "", "| Question | Cohen's kappa | agreement |", "|---|---|---|"]
    for k in ("verdict", "is_noop", "grammatical"):
        kk = ag[k]["kappa"]
        md.append(f"| {k} | {'n/a' if kk is None else f'{kk:.3f}'} | {ag[k]['percent_agreement']:.3f} |")
    md += ["", f"Marked FIX by either reviewer: {flagged or 'none'}.", ""]
    md += [f"* {c['template_id']}: {c['action']}" + (f": \"{c['from']}\" -> \"{c['to']}\" ({c['reason']})"
           if c["action"] == "rewrite" else "") for c in changes]
    OUT_JSON.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
