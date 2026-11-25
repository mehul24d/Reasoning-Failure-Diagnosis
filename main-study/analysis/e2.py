"""E2 development run (prereg/E2.md): readout map over layers x step positions, baselines, and
the RQ2 timing test. Train -> dev; no test template; automatic labels (S5 kappa pending).

Needs reports/m2/e1_dev.json (E1's selected layer and lambda per variable, seed 0).

    python -m analysis.e2 [--jobs 4]
    -> reports/m2/e2_dev.json, e2_dev.md, e2_map.png
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from analysis.e1 import LAYERS, VARIABLES, Metric, boot, row_index, variable_rows
from analysis.stats import cluster_bootstrap
from harness.config import REPO_ROOT, experiment_config
from probes import fit
from probes.activations import load_layer, load_rows
from probes.labels import TOP2, _traces, read_jsonl, step_rows

OUT = REPO_ROOT / "reports" / "m2"
BUCKETS = ["1", "2", "3", "4", "5", "6+"]
FPR = experiment_config()["thresholds"]["rq2_alarm_fpr"]
N_BOOT = experiment_config()["statistics"]["bootstrap_resamples"]


def _cell(L, kind_rows, sub, dv, y, t, lam, classes, buckets):
    Xtr = load_layer(L, kind_rows[sub])
    p = fit.fit(Xtr, y[sub], lam, classes).proba(load_layer(L, kind_rows[dv]))
    out = {"all": fit.metric(y[dv], p, classes)}
    for b in BUCKETS:
        m = buckets[dv] == b
        yy = y[dv][m]
        if m.sum() < 20 or len(np.unique(yy)) < 2:
            out[b] = None
            continue
        bt = boot(yy, p[m], len(classes), t[dv][m], n=2000)
        out[b] = {"metric": bt["estimate"], "ci_low": bt["ci_low"], "ci_high": bt["ci_high"],
                  "n": int(m.sum()), "templates": bt["n_clusters"]}
    return L, out, p


def readout_map(steps, rix, e1, jobs, log):
    res = {}
    for var in VARIABLES:
        sel = e1["variables"][var]["primary"]["selection"]
        lam = sel["lambda"]
        classes = np.arange(len(VARIABLES[var][1]))
        res[var] = {}
        for kind in ("step_first", "step_last"):
            d = variable_rows(steps, var, kind, rix)
            tr = np.flatnonzero(d["split"] == "train")
            sub = tr[fit.subsample(d["y"][tr], e1["n_train_equal"], 0)]
            dv = np.flatnonzero(d["split"] == "dev")
            buckets = np.array([s["step_pos"] for s in d["step"]])
            log(f"map {var} {kind}: {len(sub)} train, {len(dv)} dev")
            cells = Parallel(n_jobs=jobs)(delayed(_cell)(L, d["ri"], sub, dv, d["y"], d["t"], lam, classes,
                                                          buckets) for L in LAYERS)
            res[var][kind] = {str(L): c for L, c, _ in cells}
            best = max(LAYERS, key=lambda L: np.nan_to_num(res[var][kind][str(L)]["all"], nan=-1))
            res[var][kind + "_dev_tuned_layer"] = {"layer": int(best),
                                                   "dev": res[var][kind][str(best)]["all"],
                                                   "note": "selected on dev and reported on dev: optimistic"}
    return res


# ------------------------------------------------------------------------- baselines
def _texts(steps):
    """Text before each step's first token (problem + earlier steps), and the problem alone."""
    pos = {}
    with (REPO_ROOT / "data" / "generated" / "kaggle_outputs" / "btp-minor-s2" / "btp" / "data" / "generated"
          / "gemma-2-2b-it" / "s2_positions.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            pos[d["trace_id"]] = d
    tr = {r["trace_id"]: (r, p) for r, p in _traces(("dev", "train"))}
    before, question = [], []
    for s in steps:
        r, p = tr[s["trace_id"]]
        c = next(q["char"] for q in pos[s["trace_id"]]["positions"]
                 if q["kind"] == "step_first" and q["step"] == s["step"])
        question.append(p["question"])
        before.append(p["question"] + "\n" + r["text"][:c])
    return before, question


def _margins(steps, rix_tok):
    """Top-2 logit-margin features at each step_first read position."""
    need = {s["trace_id"] for s in steps}
    top = {}
    with TOP2.open() as f:
        for line in f:
            d = json.loads(line)
            if d["trace_id"] in need:
                top[d["trace_id"]] = np.array(d["l1"]) - np.array(d["l2"])
    tr = {r["trace_id"]: r for r, _ in _traces(("dev", "train"))}
    feats = []
    for s in steps:
        m = top[s["trace_id"]]
        P = len(tr[s["trace_id"]]["prompt_ids"])
        j = rix_tok[(s["trace_id"], s["step"])] - P            # output index of the read token
        now = m[j + 1] if j + 1 < len(m) else np.nan
        prev_first = rix_tok.get((s["trace_id"], s["step"] - 1))
        if prev_first is None:
            prev = np.array([])
        else:
            prev = m[prev_first - P: j]
        feats.append([now, prev.mean() if len(prev) else np.nan, prev.min() if len(prev) else np.nan,
                      float(s["step"] == 0)])
    return np.array(feats, dtype=float)


def _logit_grid(Xtr, ytr, folds, classes):
    from sklearn.linear_model import LogisticRegression

    def cv(lam):
        oof = np.zeros((len(ytr), len(classes)))
        for f in np.unique(folds):
            a, b = folds != f, folds == f
            m = LogisticRegression(C=1 / lam, max_iter=2000).fit(Xtr[a], ytr[a])
            pr = m.predict_proba(Xtr[b])
            for j, c in enumerate(m.classes_):
                oof[b, int(c)] = pr[:, j]
        return fit.metric(ytr, oof, classes)
    scores = {lam: cv(lam) for lam in fit.LAMBDAS}
    lam = max(scores, key=lambda x: (np.nan_to_num(scores[x], nan=-1), x))
    return LogisticRegression(C=1 / lam, max_iter=2000).fit(Xtr, ytr), lam


def baselines(steps, rix, e1, log):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    rix_tok = {}
    for r in load_rows():
        if r["kind"] == "step_first":
            rix_tok[(r["trace_id"], r["step"])] = r["tok"]
    out = {}
    for var in VARIABLES:
        sel = e1["variables"][var]["primary"]["selection"]
        classes = np.arange(len(VARIABLES[var][1]))
        d = variable_rows(steps, var, "step_first", rix)
        tr = np.flatnonzero(d["split"] == "train")
        sub = tr[fit.subsample(d["y"][tr], e1["n_train_equal"], 0)]
        dv = np.flatnonzero(d["split"] == "dev")
        folds = fit.template_folds(d["t"][sub], seed=0)
        y, t = d["y"], d["t"]
        probe = fit.fit(load_layer(sel["layer"], d["ri"][sub]), y[sub], sel["lambda"], classes)
        p_probe = probe.proba(load_layer(sel["layer"], d["ri"][dv]))
        S = d["step"]
        before, question = _texts(S)
        res = {"probe": fit.metric(y[dv], p_probe, classes)}
        log(f"baselines {var}")
        for name, docs in (("text_before", before), ("question_only", question)):
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([docs[i] for i in sub])
            Xs = vec.transform([docs[i] for i in sub])
            m, lam = _logit_grid(Xs, y[sub], folds, classes)
            pb = np.zeros((len(dv), len(classes)))
            pr = m.predict_proba(vec.transform([docs[i] for i in dv]))
            for j, c in enumerate(m.classes_):
                pb[:, int(c)] = pr[:, j]
            res[name] = _vs(y[dv], p_probe, pb, len(classes), t[dv], lam)
        F = _margins(S, rix_tok)
        mu = np.nanmean(F[sub], 0)
        F = np.where(np.isnan(F), mu, F)
        sc = StandardScaler().fit(F[sub])
        m, lam = _logit_grid(sc.transform(F[sub]), y[sub], folds, classes)
        pb = np.zeros((len(dv), len(classes)))
        pr = m.predict_proba(sc.transform(F[dv]))
        for j, c in enumerate(m.classes_):
            pb[:, int(c)] = pr[:, j]
        res["logit_margin"] = _vs(y[dv], p_probe, pb, len(classes), t[dv], lam)
        out[var] = res
    return out


def _vs(y, p_probe, p_base, k, templates, lam) -> dict:
    uniq, inv = np.unique(templates, return_inverse=True)
    a, b = Metric(y, p_probe, k), Metric(y, p_base, k)
    diff = cluster_bootstrap(templates, lambda w: a(w[inv]) - b(w[inv]), n_resamples=N_BOOT, seed=0)
    return {"baseline_dev": fit.metric(y, p_base, np.arange(k)), "lambda": lam,
            "probe_minus_baseline": diff}


# ------------------------------------------------------------------------- RQ2 timing
def timing(steps, rix, e1, log):
    var = "evaluation/binary"
    sel = e1["variables"][var]["primary"]["selection"]
    L, lam = sel["layer"], sel["lambda"]
    classes = np.arange(2)
    d = variable_rows(steps, var, "step_first", rix)
    tr = np.flatnonzero(d["split"] == "train")
    sub = tr[fit.subsample(d["y"][tr], e1["n_train_equal"], 0)]
    folds_sub = fit.template_folds(d["t"][sub], seed=0)
    probe = fit.fit(load_layer(L, d["ri"][sub]), d["y"][sub], lam, classes)
    # every step (any label) with a step_first row
    allsteps = [s for s in steps if (s["trace_id"], s["step"], "step_first") in rix]
    ri = np.array([rix[(s["trace_id"], s["step"], "step_first")] for s in allsteps])
    split = np.array([s["split"] for s in allsteps])
    tmpl = np.array([s["template_id"] for s in allsteps])
    # tau: out-of-fold P(incorrect) on the steps of CORRECT train traces (FPR 0.10)
    fold_of_t = {t: f for t, f in zip(d["t"][sub], folds_sub)}
    ctr = np.flatnonzero((split == "train") & np.array([s["correct"] for s in allsteps]))
    ctr = ctr[[tmpl[i] in fold_of_t for i in ctr]]
    oof = np.zeros(len(ctr))
    Xc = load_layer(L, ri[ctr])
    for f in range(fit.N_FOLDS):
        m = folds_sub != f
        pf = fit.fit(load_layer(L, d["ri"][sub][m]), d["y"][sub][m], lam, classes)
        sel_rows = np.array([fold_of_t[tmpl[i]] == f for i in ctr])
        if sel_rows.any():
            oof[sel_rows] = pf.proba(Xc[sel_rows])[:, 0]
    tau = float(np.quantile(oof, 1 - FPR))
    dv = np.flatnonzero(split == "dev")
    p_inc = probe.proba(load_layer(L, ri[dv]))[:, 0]
    alarm = p_inc >= tau
    S = [allsteps[i] for i in dv]
    bucket = np.array([s["step_pos"] for s in S])
    is_correct_trace = np.array([s["correct"] for s in S])
    is_fail = np.array([(not s["correct"]) and s["s0"] is not None and s["population"] in ("primary", "natural")
                        for s in S])
    s0 = np.array([s["s0"] if s["s0"] is not None else -1 for s in S])
    stp = np.array([s["step"] for s in S])
    pre = is_fail & (stp < s0)
    post = is_fail & (stp >= s0)
    templ = np.array([s["template_id"] for s in S])
    uniq, inv = np.unique(templ, return_inverse=True)

    def diff(mask, w):
        ww = w[inv]
        rate = {}
        for b in BUCKETS:
            m = is_correct_trace & (bucket == b)
            rate[b] = (ww[m] * alarm[m]).sum() / ww[m].sum() if ww[m].sum() > 0 else np.nan
        base = np.array([rate[b] for b in bucket])
        ok = mask & np.isfinite(base)
        return float((ww[ok] * (alarm[ok] - base[ok])).sum() / max(ww[ok].sum(), 1e-12))

    pre_ci = cluster_bootstrap(templ, lambda w: diff(pre, w), n_resamples=N_BOOT, seed=0)
    post_ci = cluster_bootstrap(templ, lambda w: diff(post, w), n_resamples=N_BOOT, seed=0)
    if pre_ci["ci_low"] > 0:
        verdict = "unsupported: evaluation alarms precede the first content deviation"
    elif post_ci["ci_low"] > 0:
        verdict = "supported (development): no excess alarm before s0, excess alarm at and after s0"
    else:
        verdict = "cannot be separated"
    # earliest alarm vs s0 per failure
    first = defaultdict(lambda: None)
    s0_of, pop_of = {}, {}
    for s, a in zip(S, alarm):
        if s["correct"] or s["s0"] is None or s["population"] not in ("primary", "natural"):
            continue
        s0_of[s["trace_id"]] = s["s0"]
        pop_of[s["trace_id"]] = s["population"]
        if a and (first[s["trace_id"]] is None or s["step"] < first[s["trace_id"]]):
            first[s["trace_id"]] = s["step"]
    dist = defaultdict(lambda: {"before": 0, "at": 0, "after": 0, "none": 0})
    for tid, z in s0_of.items():
        a = first[tid]
        k = "none" if a is None else ("before" if a < z else ("at" if a == z else "after"))
        dist[pop_of[tid]][k] += 1
    # content at the step where used: distractor-used steps vs earlier steps of the same traces
    dc = variable_rows(steps, "content/binary", "step_first", rix)
    selc = e1["variables"]["content/binary"]["primary"]["selection"]
    trc = np.flatnonzero(dc["split"] == "train")
    subc = trc[fit.subsample(dc["y"][trc], e1["n_train_equal"], 0)]
    pc = fit.fit(load_layer(selc["layer"], dc["ri"][subc]), dc["y"][subc], selc["lambda"], np.arange(2))
    first_use = {}
    for s in dc["step"]:
        if s["split"] == "dev" and s["distractor_used"]:
            first_use[s["trace_id"]] = min(first_use.get(s["trace_id"], 99), s["step"])
    idx = [i for i, s in enumerate(dc["step"]) if s["split"] == "dev" and s["trace_id"] in first_use
           and (s["distractor_used"] or s["step"] < first_use[s["trace_id"]])]
    yy = dc["y"][idx]
    pp = pc.proba(load_layer(selc["layer"], dc["ri"][idx]))
    used_vs_before = boot(yy, pp, 2, dc["t"][idx]) if len(np.unique(yy)) == 2 else None
    return {"tau": tau, "train_correct_steps_for_tau": int(len(ctr)), "fpr_target": FPR,
            "dev_correct_trace_alarm_rate": float(alarm[is_correct_trace].mean()),
            "pre_s0_excess_alarm": pre_ci, "at_after_s0_excess_alarm": post_ci,
            "n_pre_s0_steps": int(pre.sum()), "n_at_after_steps": int(post.sum()),
            "n_failures": len(s0_of), "earliest_alarm_vs_s0": {k: dict(v) for k, v in dist.items()},
            "content_used_vs_earlier_steps_auroc": used_vs_before, "verdict": verdict}


def plot(res, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for j, var in enumerate(VARIABLES):
        for i, kind in enumerate(("step_first", "step_last")):
            ax = axes[i, j]
            M = np.full((len(BUCKETS), len(LAYERS)), np.nan)
            for L in LAYERS:
                c = res["map"][var][kind][str(L)]
                for b, bk in enumerate(BUCKETS):
                    if c[bk] is not None:
                        M[b, L] = c[bk]["metric"]
            im = ax.imshow(M, aspect="auto", origin="lower", vmin=0.25 if "4-class" in var else 0.5, vmax=1.0,
                           cmap="viridis")
            ax.set_yticks(range(len(BUCKETS)), BUCKETS)
            ax.set_xlabel("layer")
            ax.set_ylabel("step")
            ax.set_title(f"{var}\n{kind} (dev)", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("E2 readout map on dev templates (AUROC / balanced accuracy); automatic labels, development run")
    fig.savefig(path, dpi=120)


def markdown(res) -> str:
    L = ["# E2 development run (dev templates; automatic labels, S5 kappa pending)", ""]
    L += ["## Best layer on the dev map (optimistic: selected and reported on dev)", "",
          "| variable | step_first: layer, dev | step_last: layer, dev |", "|---|---|---|"]
    for v, r in res["map"].items():
        a, b = r["step_first_dev_tuned_layer"], r["step_last_dev_tuned_layer"]
        L.append(f"| {v} | L{a['layer']}, {a['dev']:.3f} | L{b['layer']}, {b['dev']:.3f} |")
    L += ["", "## Probe against baselines (E1 layer, step_first, dev)", "",
          "| variable | probe | text before (Δ [95% CI]) | question only (Δ [95% CI]) | logit margin (Δ [95% CI]) |",
          "|---|---|---|---|---|"]
    for v, r in res["baselines"].items():
        cells = []
        for k in ("text_before", "question_only", "logit_margin"):
            d = r[k]["probe_minus_baseline"]
            cells.append(f"{r[k]['baseline_dev']:.3f} ({d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}])")
        L.append(f"| {v} | {r['probe']:.3f} | " + " | ".join(cells) + " |")
    t = res["timing"]
    L += ["", "## RQ2 timing (dev failures with s0)", "",
          f"- Threshold τ = {t['tau']:.3f} (FPR {t['fpr_target']} on out-of-fold correct train steps); "
          f"alarm rate on correct dev traces {t['dev_correct_trace_alarm_rate']:.3f}.",
          f"- Excess alarm before s0: {t['pre_s0_excess_alarm']['estimate']:+.3f} "
          f"[{t['pre_s0_excess_alarm']['ci_low']:+.3f}, {t['pre_s0_excess_alarm']['ci_high']:+.3f}] "
          f"({t['n_pre_s0_steps']} steps).",
          f"- Excess alarm at and after s0: {t['at_after_s0_excess_alarm']['estimate']:+.3f} "
          f"[{t['at_after_s0_excess_alarm']['ci_low']:+.3f}, {t['at_after_s0_excess_alarm']['ci_high']:+.3f}] "
          f"({t['n_at_after_steps']} steps).",
          f"- Earliest alarm vs s0: {t['earliest_alarm_vs_s0']}.",
          f"- **Development verdict: {t['verdict']}.**"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:.0f}s] {m}", flush=True)   # noqa: E731
    e1 = json.loads((a.out / "e1_dev.json").read_text())
    steps = step_rows()
    rix = row_index(load_rows())
    res = {"scope": "development: train -> dev; no test template; automatic labels, S5 kappa pending",
           "map": readout_map(steps, rix, e1, a.jobs, log)}
    res["baselines"] = baselines(steps, rix, e1, log)
    res["timing"] = timing(steps, rix, e1, log)
    res["seconds"] = round(time.time() - t0, 1)
    (a.out / "e2_dev.json").write_text(json.dumps(res, indent=2))
    (a.out / "e2_dev.md").write_text(markdown(res))
    plot(res, a.out / "e2_map.png")
    print(markdown(res))


if __name__ == "__main__":
    main()
