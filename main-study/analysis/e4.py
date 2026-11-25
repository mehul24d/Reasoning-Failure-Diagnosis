"""E4 (RQ4) analysis: content attribution and failure typology (prereg/E4.md, DRAFT).

One row per failure with a first deviating step s0. Label: content-attributable at s0 (automatic,
parser v4). The classifier uses typology features only (no activations). It is fitted on the 33
train templates and scored on --split dev (a development check of this code), or on --split test
once prereg/UNLOCK_TEST.md exists AND the S5 agreement record meets kappa_min.

    python -m analysis.e4 --split dev [--n-shuffle 100]
    -> reports/m2/e4_dev.json, e4_dev.md
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from analysis.e1 import boot, boot_diff
from analysis.e3 import mean_ci, population
from harness.config import REPO_ROOT, experiment_config
from probes import fit
from probes.labels import ERROR_TYPES, _guard, _traces, step_rows, test_unlocked

CFG = experiment_config()
KAPPA_MIN = CFG["thresholds"]["kappa_min"]
OUT = REPO_ROOT / "reports" / "m2"
AGREEMENT = REPO_ROOT / "reports" / "m1" / "s5_annotation" / "agreement.json"
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
TYPOLOGY = ("distractor_condition", "n_steps", "s0_index", "s0_relative", "question_words",
            "question_numerals", "s0_compound")
S0_TYPES = tuple(t for t in ERROR_TYPES if t != "none") + ("non-computational",)


def kappa_status(path: Path = AGREEMENT) -> dict:
    """The S5 gate: both kappas (first wrong step; error type on the same step) >= kappa_min."""
    if not path.exists():
        return {"met": False, "status": "S5 kappa pending (no agreement record)"}
    a = json.loads(path.read_text())
    ks = {k: a.get(k, {}).get("kappa", {}).get("estimate") for k in ("first_wrong_step", "error_type_on_same_step")}
    met = all(a.get(k, {}).get("meets_kappa_min") is True for k in ks)
    return {"met": bool(met), "kappas": ks, "kappa_min": KAPPA_MIN,
            "status": f"S5 kappa {'meets' if met else 'BELOW'} {KAPPA_MIN}: {ks}"}


def guard(split: str, kappa: dict) -> None:
    if split == "test":
        if not test_unlocked():
            raise PermissionError("test templates are locked (prereg/UNLOCK_TEST.md is absent)")
        if not kappa["met"]:
            raise SystemExit(f"E4 on test needs the S5 agreement record with kappa >= {KAPPA_MIN}: {kappa['status']}")


def failure_rows(splits: tuple, allow_test: bool = False) -> list[dict]:
    _guard(splits, allow_test)
    question = {r["trace_id"]: p["question"] for r, p in _traces(tuple(sorted(splits)))}
    out = []
    for s in step_rows(splits, allow_test=allow_test):
        if s["correct"] or not s["is_s0"] or s["content_attributable"] is None:
            continue
        q = question[s["trace_id"]]
        et = s["error_type"] if s["error_type"] in S0_TYPES else "non-computational"
        out.append({
            "trace_id": s["trace_id"], "template_id": s["template_id"], "split": s["split"],
            "population": population(s), "y": int(bool(s["content_attributable"])),
            "s0_error_type": et,
            "x": {"distractor_condition": float(s["condition"] == "distractor"),
                  "n_steps": float(s["n_steps"]), "s0_index": float(s["step"]),
                  "s0_relative": float(s["step"]) / max(s["n_steps"], 1),
                  "question_words": float(len(q.split())),
                  "question_numerals": float(len(NUM.findall(q))),
                  "s0_compound": float(s["compound"])}})
    return out


def design(F: list[dict], with_type: bool) -> np.ndarray:
    X = [[f["x"][k] for k in TYPOLOGY] + ([float(f["s0_error_type"] == t) for t in S0_TYPES] if with_type else [])
         for f in F]
    return np.array(X, float)


def classifier(tr: list[dict], ev: list[dict], with_type: bool, n_shuffle: int) -> dict:
    classes = np.array([0, 1])
    Xtr, Xev = design(tr, with_type), design(ev, with_type)
    ytr, yev = np.array([f["y"] for f in tr]), np.array([f["y"] for f in ev])
    ttr, tev = np.array([f["template_id"] for f in tr]), np.array([f["template_id"] for f in ev])
    folds = fit.template_folds(ttr, seed=0)
    by_lam = {lam: fit.cv_score(Xtr, ytr, folds, lam) for lam in fit.LAMBDAS}
    lam = max(by_lam, key=lambda l: (np.nan_to_num(by_lam[l], nan=-1), l))
    probe = fit.fit(Xtr, ytr, lam, classes)
    p = probe.proba(Xev)
    out = {"features": list(TYPOLOGY) + ([f"s0_type={t}" for t in S0_TYPES] if with_type else []),
           "lambda": lam, "cv_by_lambda": {str(k): v for k, v in by_lam.items()},
           "n_train": int(len(tr)), "n_eval": int(len(ev)), "eval_templates": int(len(set(tev))),
           "eval_positive": int(yev.sum())}
    if len(np.unique(yev)) < 2:
        out["auroc"] = None
        return out
    out["auroc"] = boot(yev, p, 2, tev)
    rng = np.random.default_rng(0)
    nulls = [fit.fit(Xtr, fit.shuffle_within(ytr, ttr, rng), lam, classes).proba(Xev) for _ in range(n_shuffle)]
    null_m = np.array([fit.metric(yev, q, classes) for q in nulls])
    out["shuffled"] = {"n": n_shuffle, "mean": float(np.nanmean(null_m)),
                       "p_perm": float((1 + (null_m >= out["auroc"]["estimate"]).sum()) / (1 + n_shuffle))}
    out["minus_shuffled_ci"] = boot_diff(yev, p, nulls, 2, tev)
    out["criterion_met"] = bool(out["auroc"]["ci_low"] > 0.5)
    out["coef_standardised"] = dict(zip(out["features"], map(float, probe.model.coef_[0])))
    return out


def rates(F: list[dict]) -> dict:
    y = np.array([f["y"] for f in F], float)
    t = np.array([f["template_id"] for f in F])
    out = {"overall": mean_ci(y, t, unit=True)}
    for key in ("population", "s0_error_type"):
        for v in sorted({f[key] for f in F}):
            m = np.array([f[key] == v for f in F])
            out[f"{key}={v}"] = mean_ci(y[m], t[m])
    return out


def run(split="dev", n_shuffle=100, log=print) -> dict:
    t0 = time.time()
    kappa = kappa_status()
    guard(split, kappa)
    F = failure_rows(("train", split), allow_test=split == "test")
    tr = [f for f in F if f["split"] == "train"]
    ev = [f for f in F if f["split"] == split]
    log(f"[{time.time() - t0:.0f}s] {len(tr)} train / {len(ev)} {split} failures with s0")
    status = ("DEVELOPMENT CHECK of the E4 code on dev (prereg/E4.md is a DRAFT)" if split == "dev" else "TEST")
    out = {"split": split, "status": status, "labels": f"automatic (parser v4); {kappa['status']}",
           "kappa": kappa, "criterion": "template-bootstrap CI of the dev/test AUROC lies above 0.5",
           "attribution_rates": {"train": rates(tr), split: rates(ev)},
           "classifier": {"typology (primary)": classifier(tr, ev, False, n_shuffle),
                          "typology + s0 error type (partly definitional)": classifier(tr, ev, True, n_shuffle)},
           "seconds": None}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def markdown(r: dict) -> str:
    f = lambda c: "–" if not c or not np.isfinite(c["estimate"]) else (  # noqa: E731
        f"{c['estimate']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
    L = [f"# E4 ({r['split']}): content attribution and typology", "", f"**{r['status']}.** Labels: {r['labels']}.",
         "", f"Criterion (prereg/E4.md, DRAFT): {r['criterion']}.", "", "## Share of failures content-attributable at s0", "",
         "| set | " + " | ".join(r["attribution_rates"]) + " |", "|---|" + "---|" * len(r["attribution_rates"])]
    keys = sorted(set().union(*[v.keys() for v in r["attribution_rates"].values()]), key=lambda k: (k != "overall", k))
    for k in keys:
        row = []
        for v in r["attribution_rates"].values():
            c = v.get(k)
            row.append("" if c is None else f"{f(c)} (n {c['n']})")
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L += ["", "## Classifier", "", "| features | n train / eval (templates) | AUROC | shuffled mean | AUROC − shuffled | p_perm | CI > 0.5 |",
          "|---|---|---|---|---|---|---|"]
    for name, c in r["classifier"].items():
        if c.get("auroc") is None:
            L.append(f"| {name} | {c['n_train']} / {c['n_eval']} | one class only | | | | |")
            continue
        L.append(f"| {name} | {c['n_train']} / {c['n_eval']} ({c['eval_templates']}) | {f(c['auroc'])} | "
                 f"{c['shuffled']['mean']:.3f} | {f(c['minus_shuffled_ci'])} | {c['shuffled']['p_perm']:.3f} | "
                 f"{'yes' if c['criterion_met'] else 'no'} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n-shuffle", type=int, default=100)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = run(a.split, a.n_shuffle)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"e4_{a.split}.json").write_text(json.dumps(r, indent=1, default=float))
    (a.out / f"e4_{a.split}.md").write_text(markdown(r))
    print(markdown(r))


if __name__ == "__main__":
    main()
