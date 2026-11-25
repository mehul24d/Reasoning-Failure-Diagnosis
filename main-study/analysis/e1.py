"""E1 development run (prereg/E1.md): the 2 x 2 variables with the shared construction procedure.

Fit and select on the 33 train templates, report on the 16 dev templates. No test template is
read. Labels are automatic (parser v4); the S5 annotator kappa is pending.

    python -m analysis.e1 [--seeds 0,1,2,3,4] [--n-shuffle 100] [--jobs 4]
    -> reports/m2/e1_dev.json, reports/m2/e1_dev.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from analysis.stats import cluster_bootstrap
from harness.config import REPO_ROOT, experiment_config
from probes import fit
from probes.activations import load_layer, load_rows
from probes.labels import ERROR_TYPES, OPS, step_rows

OUT = REPO_ROOT / "reports" / "m2"
LAYERS = list(range(26))
VARIABLES = {
    "content/binary": ("distractor_used", (False, True)),
    "content/4-class": ("op", OPS),
    "evaluation/binary": ("step_correct", (False, True)),
    "evaluation/4-class": ("error_type", ERROR_TYPES),
}
THRESH = experiment_config()["thresholds"]["rq1_min_score"]
N_BOOT = experiment_config()["statistics"]["bootstrap_resamples"]


def variable_rows(steps: list[dict], var: str, kind: str = "step_first", rix=None) -> dict:
    """Rows of one variable with its label: X row index, y (class index), template, split."""
    key, classes = VARIABLES[var]
    out = {"ri": [], "y": [], "t": [], "split": [], "step": []}
    for s in steps:
        v = s[key]
        if v is None or (var.startswith("evaluation") and s["extra_valid"]):
            continue
        k = (s["trace_id"], s["step"], kind)
        if k not in rix:
            continue
        out["ri"].append(rix[k])
        out["y"].append(classes.index(v))
        out["t"].append(s["template_id"])
        out["split"].append(s["split"])
        out["step"].append(s)
    return {k: (np.array(v) if k != "step" else v) for k, v in out.items()}


def row_index(rows) -> dict:
    return {(r["trace_id"], r["step"], r["kind"]): r["i"] for r in rows if r["step"] is not None}


# ---------------------------------------------------------------- weighted metrics (bootstrap)
class Metric:
    """AUROC (binary) or balanced accuracy (4-class) with per-row weights, fast enough for
    10,000 template resamples: the sort is done once."""

    def __init__(self, y: np.ndarray, scores: np.ndarray, n_classes: int):
        self.y, self.k = y, n_classes
        if n_classes == 2:
            s = scores[:, 1] if scores.ndim == 2 else scores
            order = np.argsort(s, kind="mergesort")
            self.order, self.s, self.pos = order, s[order], (y[order] == 1)
            _, self.grp = np.unique(self.s, return_inverse=True)
        else:
            self.pred = scores.argmax(1)

    def __call__(self, w: np.ndarray) -> float:
        if self.k == 2:
            ww = w[self.order]
            neg = np.where(~self.pos, ww, 0.0)
            pw = np.where(self.pos, ww, 0.0)
            # negatives strictly below each score group, and in the same group (ties count 1/2)
            gneg = np.bincount(self.grp, weights=neg)
            below = np.concatenate([[0.0], np.cumsum(gneg)[:-1]])
            num = (pw * (below[self.grp] + 0.5 * gneg[self.grp])).sum()
            den = pw.sum() * neg.sum()
            return float(num / den) if den > 0 else float("nan")
        rec = []
        for c in range(self.k):
            m = self.y == c
            if w[m].sum() > 0:
                rec.append((w[m] * (self.pred[m] == c)).sum() / w[m].sum())
        return float(np.mean(rec)) if rec else float("nan")


def boot(y, scores, k, templates, n=N_BOOT, unit=None) -> dict:
    uniq, inv = np.unique(templates, return_inverse=True)
    m = Metric(y, scores, k)
    return cluster_bootstrap(templates, lambda w: m(w[inv]), n_resamples=n, seed=0, unit_values=unit)


def boot_diff(y, s_real, s_null: list, k, templates, n=N_BOOT) -> dict:
    """CI of (real metric - mean of the shuffled-probe metrics), same template resample."""
    uniq, inv = np.unique(templates, return_inverse=True)
    mr = Metric(y, s_real, k)
    mn = [Metric(y, s, k) for s in s_null]
    return cluster_bootstrap(templates, lambda w: mr(w[inv]) - float(np.mean([q(w[inv]) for q in mn])),
                             n_resamples=n, seed=0)


# ---------------------------------------------------------------- the procedure
def dim_cv(X, y, folds, classes) -> float:
    oof = np.zeros((len(y), len(classes)))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu = np.stack([X[tr][y[tr] == c].mean(0) for c in classes])
        p = fit.Probe(classes, None, None, mu, 0.0)
        oof[te] = p.dim_scores(X[te])
    return fit.metric(y, oof, classes)


def _null(Xtr, yy, lam, classes, Xdv):
    return fit.fit(Xtr, yy, lam, classes).proba(Xdv)


def select_parallel(rows, y, folds, classes, jobs: int) -> dict:
    def one(L):
        X = load_layer(L, rows)
        return L, fit.cv_score(X, y, folds, 1.0), dim_cv(X, y, folds, classes)
    res = Parallel(n_jobs=jobs)(delayed(one)(L) for L in LAYERS)
    by_layer = {L: a for L, a, _ in res}
    by_layer_dim = {L: b for L, _, b in res}
    best_L = max(by_layer, key=lambda L: (np.nan_to_num(by_layer[L], nan=-1), -L))
    X = load_layer(best_L, rows)
    lams = Parallel(n_jobs=jobs)(delayed(fit.cv_score)(X, y, folds, lam) for lam in fit.LAMBDAS)
    by_lam = dict(zip(fit.LAMBDAS, lams))
    best_lam = max(by_lam, key=lambda lam: (np.nan_to_num(by_lam[lam], nan=-1), lam))
    best_dim = max(by_layer_dim, key=lambda L: (np.nan_to_num(by_layer_dim[L], nan=-1), -L))
    return {"layer": int(best_L), "lambda": float(best_lam), "dim_layer": int(best_dim),
            "cv_by_layer": {int(k): v for k, v in by_layer.items()},
            "cv_by_lambda": {str(k): v for k, v in by_lam.items()},
            "dim_cv_by_layer": {int(k): v for k, v in by_layer_dim.items()}}


def run(seeds=(0, 1, 2, 3, 4), n_shuffle=100, jobs=4, log=print) -> dict:
    t0 = time.time()
    steps = step_rows()
    rix = row_index(load_rows())
    V = {v: variable_rows(steps, v, rix=rix) for v in VARIABLES}
    n_min = min(int((V[v]["split"] == "train").sum()) for v in V)
    res = {"n_train_equal": n_min, "variables": {}, "labels": "automatic (parser v4); S5 kappa pending",
           "scope": "development: fit/select on 33 train templates, scored on 16 dev templates; no test template read"}
    for var, d in V.items():
        classes = np.arange(len(VARIABLES[var][1]))
        k = len(classes)
        tr_all = np.flatnonzero(d["split"] == "train")
        dv = np.flatnonzero(d["split"] == "dev")
        out = {"classes": [str(c) for c in VARIABLES[var][1]],
               "n_train_rows_available": int(len(tr_all)), "n_dev": int(len(dv)),
               "dev_templates": int(len(set(d["t"][dv]))),
               "dev_class_counts": {str(VARIABLES[var][1][c]): int((d["y"][dv] == c).sum()) for c in classes},
               "seeds": {}}
        for seed in seeds:
            sub = tr_all[fit.subsample(d["y"][tr_all], n_min, seed)]
            folds = fit.template_folds(d["t"][sub], seed=seed)
            log(f"[{time.time() - t0:.0f}s] {var} seed {seed}: select on {len(sub)} train rows")
            sel = select_parallel(d["ri"][sub], d["y"][sub], folds, classes, jobs)
            L, lam = sel["layer"], sel["lambda"]
            Xtr, ytr = load_layer(L, d["ri"][sub]), d["y"][sub]
            probe = fit.fit(Xtr, ytr, lam, classes)
            Xdv = load_layer(L, d["ri"][dv])
            p = probe.proba(Xdv)
            ydv, tdv = d["y"][dv], d["t"][dv]
            e = {"selection": sel, "dev": fit.metric(ydv, p, classes)}
            # difference-in-means readout at its own CV layer
            Ld = sel["dim_layer"]
            Xd_tr = load_layer(Ld, d["ri"][sub])
            mu = np.stack([Xd_tr[ytr == c].mean(0) for c in classes])
            pd = fit.Probe(classes, None, None, mu, 0.0).dim_scores(load_layer(Ld, d["ri"][dv]))
            e["dev_dim"] = fit.metric(ydv, pd, classes)
            if seed == seeds[0]:
                correct = (p.argmax(1) == ydv).astype(float)
                e["dev_ci"] = boot(ydv, p, k, tdv, unit=correct)
                e["dev_dim_ci"] = boot(ydv, pd, k, tdv)
                rng = np.random.default_rng(seed)
                perms = [fit.shuffle_within(ytr, d["t"][sub], rng) for _ in range(n_shuffle)]
                log(f"[{time.time() - t0:.0f}s] {var}: {n_shuffle} shuffled-label probes")
                nulls = Parallel(n_jobs=jobs)(delayed(_null)(Xtr, yy, lam, classes, Xdv) for yy in perms)
                null_m = np.array([fit.metric(ydv, q, classes) for q in nulls])
                e["shuffled"] = {"n": n_shuffle, "mean": float(np.nanmean(null_m)),
                                 "q95": float(np.nanquantile(null_m, 0.95)),
                                 "p_perm": float((1 + (null_m >= e["dev"]).sum()) / (1 + n_shuffle))}
                e["minus_shuffled_ci"] = boot_diff(ydv, p, nulls, k, tdv, n=N_BOOT)
                e["readable_on_dev"] = bool(e["dev"] >= THRESH and e["minus_shuffled_ci"]["ci_low"] > 0)
            out["seeds"][str(seed)] = e
        devs = [out["seeds"][str(s)]["dev"] for s in seeds]
        out["dev_over_seeds"] = {"mean": float(np.mean(devs)), "sd": float(np.std(devs, ddof=1)) if len(devs) > 1 else 0.0,
                                 "min": float(np.min(devs)), "max": float(np.max(devs))}
        out["primary"] = out["seeds"][str(seeds[0])]
        res["variables"][var] = out
    # cosines between the binary content and evaluation difference-in-means directions, every layer
    cos = {}
    dc, de = V["content/binary"], V["evaluation/binary"]
    for L in LAYERS:
        tc = np.flatnonzero(dc["split"] == "train")
        te = np.flatnonzero(de["split"] == "train")
        Xc, Xe = load_layer(L, dc["ri"][tc]), load_layer(L, de["ri"][te])
        a = Xc[dc["y"][tc] == 1].mean(0) - Xc[dc["y"][tc] == 0].mean(0)
        b = Xe[de["y"][te] == 1].mean(0) - Xe[de["y"][te] == 0].mean(0)
        cos[L] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    res["cosine_content_vs_evaluation_binary_dim"] = cos
    res["seconds"] = round(time.time() - t0, 1)
    return res


def markdown(r: dict) -> str:
    L = ["# E1 development run (dev templates; automatic labels, S5 kappa pending)", "",
         f"Fit and selection on 33 train templates (template-grouped CV), scores on 16 dev templates; "
         f"equal train size {r['n_train_equal']} rows per variable. Not a test-set result.", "",
         "| variable | layer, λ (train CV) | dev metric [95% CI] | shuffled mean (p) | probe − shuffled [95% CI] | "
         "diff-in-means dev | seeds 0–4 mean ± sd | readable on dev |",
         "|---|---|---|---|---|---|---|---|"]
    for v, o in r["variables"].items():
        p = o["primary"]
        ci, dci = p["dev_ci"], p["minus_shuffled_ci"]
        L.append(f"| {v} | L{p['selection']['layer']}, {p['selection']['lambda']:g} | "
                 f"{p['dev']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] | "
                 f"{p['shuffled']['mean']:.3f} ({p['shuffled']['p_perm']:.3f}) | "
                 f"{dci['estimate']:+.3f} [{dci['ci_low']:+.3f}, {dci['ci_high']:+.3f}] | "
                 f"{p['dev_dim']:.3f} (L{p['selection']['dim_layer']}) | "
                 f"{o['dev_over_seeds']['mean']:.3f} ± {o['dev_over_seeds']['sd']:.3f} | "
                 f"{'yes' if p['readable_on_dev'] else 'no'} |")
    L += ["", "n and clustering of the dev rows (A16):", ""]
    for v, o in r["variables"].items():
        c = o["primary"]["dev_ci"].get("clustering") or {}
        L.append(f"- {v}: n = {o['n_dev']} steps, {o['dev_templates']} templates, classes {o['dev_class_counts']}; "
                 f"ICC {c.get('icc')}, deff {c.get('deff')}, n_eff {c.get('n_eff')}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n-shuffle", type=int, default=experiment_config()["probing"]["shuffled_label_permutations"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = run(tuple(int(s) for s in a.seeds.split(",")), a.n_shuffle, a.jobs, log=lambda m: print(m, flush=True))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e1_dev.json").write_text(json.dumps(r, indent=2))
    (a.out / "e1_dev.md").write_text(markdown(r))
    print(markdown(r))


if __name__ == "__main__":
    main()
