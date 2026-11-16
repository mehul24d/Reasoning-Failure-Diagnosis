"""Two-annotator agreement on the S5 failure annotation (docs/annotation/ANNOTATION_GUIDE.md).

Run ONLY when both annotators have returned their completed sheets. The script refuses to
compute anything if an item it would use is unlabelled in either sheet.

    python -I scripts/annotation_agreement.py <annotator_A.csv> <annotator_B.csv> [--part 1]

Inputs: the two returned sheets (CSV exported from Google Sheets: File > Download > CSV), and
reports/m1/s5_annotation/sample.json (item -> template, split). Rows are matched by item_id;
the sheet order does not matter. Practice items (P01-P10) are dropped before anything is
computed (they are excluded from kappa).

Measures, on the non-practice items (all of them, or only part 1 = test with --part 1):
  * first_wrong_step, exact match. Labels are nominal: a step number, "answer" (only the answer
    line is wrong) or "none" (the solution is correct).
  * error_type (planning / binding / arithmetic / other), on the items where both annotators
    gave the same first_wrong_step other than "none".
For each: Cohen's kappa, raw agreement, and PABAK = (k p_o - 1) / (k - 1) (Byrt et al. 1993),
with k = 4 for error_type and, for first_wrong_step, k = the number of distinct labels either
annotator used on the items analysed (stated in the output). 95% CIs: the pre-registered
template-clustered bootstrap (analysis.stats.cluster_bootstrap; experiment.yaml statistics:
10,000 resamples, seed 0, 95%), whole templates resampled. The pre-registered threshold is
kappa >= thresholds.kappa_min (0.6); it is reported for each kappa, on the point estimate.

Outputs: reports/m1/s5_annotation/agreement.json and agreement.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # works under `python -I`

import numpy as np                                                # noqa: E402

from analysis.stats import cluster_bootstrap                      # noqa: E402
from harness.config import REPO_ROOT, experiment_config           # noqa: E402

RECORD = REPO_ROOT / "reports" / "m1" / "s5_annotation" / "sample.json"
OUT = REPO_ROOT / "reports" / "m1" / "s5_annotation" / "agreement.json"
ERROR_TYPES = ("planning", "binding", "arithmetic", "other")
_STEP = re.compile(r"^\s*(?:step\s*)?\[?\s*(\d+)\s*\]?\s*$", re.IGNORECASE)


def n_steps(model_solution: str) -> int:
    return len(re.findall(r"^\[(\d+)\] ", model_solution, re.MULTILINE))


def norm_step(v: str, n: int) -> str | None:
    """'3', '[3]', 'step 3' -> '3' (if 1 <= 3 <= n); 'none'; 'answer'. None if blank or invalid."""
    v = (v or "").strip().lower()
    if v in ("none", "answer"):
        return v
    if v in ("answer line", "ans"):
        return "answer"
    m = _STEP.match(v)
    if m and 1 <= int(m.group(1)) <= n:
        return str(int(m.group(1)))
    return None


def norm_type(v: str) -> str | None:
    v = (v or "").strip().lower()
    return v if v in ERROR_TYPES else None


def read_sheet(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        iid = (r.get("item_id") or "").strip()
        if iid:
            if iid in out:
                raise SystemExit(f"{path}: item {iid} appears twice")
            out[iid] = r
    return out


def labels(sheet: dict[str, dict], ids: list[str], who: str) -> tuple[dict, list[str]]:
    """Normalised labels per item, and the problems found (blank / invalid entries)."""
    lab, bad = {}, []
    for iid in ids:
        r = sheet.get(iid)
        if r is None:
            bad.append(f"{who} {iid}: row missing")
            continue
        step = norm_step(r.get("first_wrong_step", ""), n_steps(r.get("model_solution", "")))
        typ = norm_type(r.get("error_type", ""))
        conf = (r.get("confidence") or "").strip()
        if step is None:
            bad.append(f"{who} {iid}: first_wrong_step {r.get('first_wrong_step', '')!r} is blank or not a valid step")
        elif step != "none" and typ is None:
            bad.append(f"{who} {iid}: error_type {r.get('error_type', '')!r} is blank or not one of {ERROR_TYPES}")
        if conf not in ("1", "2", "3"):
            bad.append(f"{who} {iid}: confidence {conf!r} is not 1, 2 or 3")
        lab[iid] = {"step": step, "type": typ if step != "none" else None, "confidence": conf}
    return lab, bad


def weighted_kappa_parts(a: list, b: list, w: np.ndarray) -> tuple[float, float]:
    """(p_o, p_e) of Cohen's kappa with item weights w (all ones = the usual kappa)."""
    W = w.sum()
    if W <= 0:
        return np.nan, np.nan
    a, b = np.asarray(a, object), np.asarray(b, object)
    po = float(w[a == b].sum() / W)
    pe = sum(float(w[a == c].sum() / W) * float(w[b == c].sum() / W) for c in set(a) | set(b))
    return po, pe


def stats(a: list, b: list, templates: list, k: int, boot: dict) -> dict:
    """Kappa, raw agreement and PABAK with template-clustered bootstrap CIs."""
    n = len(a)
    if n == 0:
        return {"n": 0}
    uniq, inv = np.unique(np.asarray(templates), return_inverse=True)

    def make(which):
        def f(cw):
            w = cw[inv]
            po, pe = weighted_kappa_parts(a, b, w)
            if which == "raw":
                return po
            if which == "pabak":
                return (k * po - 1) / (k - 1)
            return np.nan if pe >= 1 else (po - pe) / (1 - pe)
        return f

    out = {"n": n, "templates": int(len(uniq)), "k_categories": k}
    for which in ("kappa", "raw", "pabak"):
        r = cluster_bootstrap(np.asarray(templates), make(which), n_resamples=boot["n"], seed=boot["seed"], ci=boot["ci"])
        out[which] = {"estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                      "n_valid_resamples": r["n_valid_resamples"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_a")
    ap.add_argument("sheet_b")
    ap.add_argument("--part", type=int, choices=(1, 2), help="only part 1 (test) or part 2 (train and dev)")
    ap.add_argument("--record", default=str(RECORD))
    ap.add_argument("--out", help="default reports/m1/s5_annotation/agreement[_part<n>].json")
    a = ap.parse_args(argv)
    a.out = a.out or str(OUT.with_name(f"agreement_part{a.part}.json") if a.part else OUT)
    cfg = experiment_config()
    st = cfg["statistics"]
    boot = {"n": st["bootstrap_resamples"], "seed": 0, "ci": st["ci"]}
    kmin = cfg["thresholds"]["kappa_min"]
    rec = json.loads(Path(a.record).read_text(encoding="utf-8"))
    items = [it for it in rec["items"] if it["part"] != 0 and (a.part is None or it["part"] == a.part)]
    ids = [it["item_id"] for it in items]
    tmpl = {it["item_id"]: it["template_id"] for it in items}

    sa, sb = read_sheet(Path(a.sheet_a)), read_sheet(Path(a.sheet_b))
    known = {it["item_id"] for it in rec["items"]}
    extra = sorted((set(sa) | set(sb)) - known)
    if extra:
        raise SystemExit(f"items not in the sample record: {extra[:10]}")
    la, bad_a = labels(sa, ids, "A")
    lb, bad_b = labels(sb, ids, "B")
    if bad_a or bad_b:
        print("\n".join((bad_a + bad_b)[:40]))
        raise SystemExit(f"{len(bad_a)} problems in sheet A and {len(bad_b)} in sheet B: both sheets must be "
                         "complete before agreement is computed")

    sa_, sb_ = [la[i]["step"] for i in ids], [lb[i]["step"] for i in ids]
    k_step = len(set(sa_) | set(sb_))
    step = stats(sa_, sb_, [tmpl[i] for i in ids], max(k_step, 2), boot)
    same = [i for i in ids if la[i]["step"] == lb[i]["step"] and la[i]["step"] != "none"]
    typ = stats([la[i]["type"] for i in same], [lb[i]["type"] for i in same], [tmpl[i] for i in same],
                len(ERROR_TYPES), boot)
    for s in (step, typ):
        if s.get("n"):
            s["meets_kappa_min"] = bool(np.isfinite(s["kappa"]["estimate"]) and s["kappa"]["estimate"] >= kmin)
    conf = {w: {c: sum(l[i]["confidence"] == c for i in ids) for c in "123"} for w, l in (("A", la), ("B", lb))}
    out = {"note": "S5 two-annotator agreement; practice items excluded",
           "part": a.part or "all", "n_items": len(ids), "kappa_min": kmin,
           "bootstrap": {"resamples": boot["n"], "seed": boot["seed"], "ci": boot["ci"], "cluster": "template"},
           "first_wrong_step": step, "error_type_on_same_step": typ,
           "label_counts": {w: {"none": sum(l[i]["step"] == "none" for i in ids),
                                "answer": sum(l[i]["step"] == "answer" for i in ids),
                                **{t: sum(l[i]["type"] == t for i in ids) for t in ERROR_TYPES}}
                            for w, l in (("A", la), ("B", lb))},
           "confidence_counts": conf,
           "sheets": {"A": str(a.sheet_a), "B": str(a.sheet_b)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    def row(name, s):
        if not s.get("n"):
            return f"| {name} | 0 | – | – | – | – |"
        f = lambda m: f"{s[m]['estimate']:.3f} [{s[m]['ci_low']:.3f}, {s[m]['ci_high']:.3f}]"
        return (f"| {name} | {s['n']} | {f('kappa')} | {f('raw')} | {f('pabak')} (k = {s['k_categories']}) | "
                f"{'yes' if s['meets_kappa_min'] else 'no'} |")
    md = [f"# S5 annotation agreement ({'part ' + str(a.part) if a.part else 'all parts'})", "",
          f"{len(ids)} items, practice excluded. 95% CIs: template-clustered bootstrap, {boot['n']:,} resamples.", "",
          f"| Measure | n | Cohen's kappa | raw agreement | PABAK | kappa >= {kmin} |", "|---|---|---|---|---|---|",
          row("first_wrong_step (exact)", step), row("error_type (same step)", typ), ""]
    Path(a.out).with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
