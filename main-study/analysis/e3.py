"""E3 (RQ3) analysis: self-sourced decodability of the gold content value (prereg/E3.md, DRAFT).

Population: failures whose first deviating step s0 is content-attributable. Read position: s0's
first token. Probes are fitted on the 33 train templates only, using the E1 procedure. Scored on
--split dev (a development check of this code), or --split test once prereg/UNLOCK_TEST.md
exists and prereg/E3.md has been signed off.

    python -m analysis.e3 --split dev [--jobs 4] [--n-shuffle 100] [--r1-out DIR]
    -> reports/m2/e3_dev.json, e3_dev.md

Needs reports/m2/e1_dev.json: the E1 seed-0 layer and lambda of each content variable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from analysis.e1 import VARIABLES, _null, row_index, select_parallel, variable_rows
from analysis.stats import cluster_bootstrap
from harness.config import REPO_ROOT, experiment_config
from probes import fit
from probes.activations import load_layer, load_rows
from probes.labels import OPS, step_rows, test_unlocked

CFG = experiment_config()
N_BOOT = CFG["statistics"]["bootstrap_resamples"]
MIN_TOP1 = CFG["thresholds"]["rq3_min_top1_gold"]
OUT = REPO_ROOT / "reports" / "m2"
E1_JSON = OUT / "e1_dev.json"
CONTENT = ("content/binary", "content/4-class")
POS = ("1", "2", "3", "4", "5", "6+")
EPS = 1e-9


def guard(split: str) -> None:
    if split == "test":
        if not test_unlocked():
            raise PermissionError("test templates are locked (prereg/UNLOCK_TEST.md is absent)")
        raise SystemExit("the S4 residual cache on this machine holds train/dev rows only "
                         "(probes/activations.py); build the test rows after the unlock")


# ---------------------------------------------------------------- population
def failures(steps: list[dict], split: str, rix: dict) -> list[dict]:
    """Content-attributable s0 rows of failed traces on `split`, with gold and bound class."""
    out = []
    for s in steps:
        if s["split"] != split or s["correct"] or not s["is_s0"] or not s["content_attributable"]:
            continue
        k = (s["trace_id"], s["step"], "step_first")
        if k not in rix:
            continue
        if s["distractor_used"]:
            var, gold, bound = "content/binary", 0, 1         # classes (False, True)
        else:
            var, gold, bound = "content/4-class", OPS.index(s["gold_op"]), OPS.index(s["op"])
        out.append({"trace_id": s["trace_id"], "template_id": s["template_id"],
                    "population": s["population"] or "none", "step": s["step"],
                    "step_pos": s["step_pos"], "variable": var, "gold": gold, "bound": bound,
                    "ri": rix[k]})
    return out


# ---------------------------------------------------------------- per-failure statistics
def per_failure(p: np.ndarray, gold: np.ndarray, bound: np.ndarray, pi: np.ndarray | None = None) -> dict:
    """top1 = [argmax p = gold]; gob = [p_gold > p_bound] (ties 1/2); `pi` = class prior for
    the prior-corrected gob (log p - log pi)."""
    i = np.arange(len(gold))
    pg, pb = p[i, gold], p[i, bound]
    out = {"top1": (p.argmax(1) == gold).astype(float),
           "gob": np.where(pg > pb, 1.0, np.where(pg == pb, 0.5, 0.0)),
           "bound_top1": (p.argmax(1) == bound).astype(float)}
    if pi is not None:
        lg = np.log(pg + EPS) - np.log(pi[gold] + EPS)
        lb = np.log(pb + EPS) - np.log(pi[bound] + EPS)
        out["gob_prior_corrected"] = np.where(lg > lb, 1.0, np.where(lg == lb, 0.5, 0.0))
    return out


def step_prior(d: dict, k: int) -> dict[str, np.ndarray]:
    """Train class frequencies of a variable at each step position (no activations)."""
    tr = d["split"] == "train"
    pos = np.array([s["step_pos"] for s in d["step"]])
    allc = np.bincount(d["y"][tr], minlength=k).astype(float)
    out = {}
    for q in POS:
        m = tr & (pos == q)
        c = np.bincount(d["y"][m], minlength=k).astype(float) if m.any() else allc
        out[q] = c / c.sum()
    return out


def mean_ci(x: np.ndarray, t: np.ndarray, unit: bool = False) -> dict:
    if len(x) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    uniq, inv = np.unique(t, return_inverse=True)

    def st(w):
        ww = w[inv]
        return float((ww * x).sum() / ww.sum()) if ww.sum() > 0 else float("nan")
    r = cluster_bootstrap(t, st, n_resamples=N_BOOT, seed=0, unit_values=x if unit else None)
    r["n"] = int(len(x))
    return r


# ---------------------------------------------------------------- probes
def content_probe(d: dict, var: str, sel: dict, n_min: int, n_shuffle: int, jobs: int, X_eval) -> dict:
    """E1's seed-0 probe for `var` (same subsample, layer, lambda, shuffles), scored on X_eval."""
    classes = np.arange(len(VARIABLES[var][1]))
    tr_all = np.flatnonzero(d["split"] == "train")
    sub = tr_all[fit.subsample(d["y"][tr_all], n_min, 0)]
    L, lam = sel["layer"], sel["lambda"]
    Xtr, ytr = load_layer(L, d["ri"][sub]), d["y"][sub]
    Xe = load_layer(L, X_eval)
    probe = fit.fit(Xtr, ytr, lam, classes)
    rng = np.random.default_rng(0)
    perms = [fit.shuffle_within(ytr, d["t"][sub], rng) for _ in range(n_shuffle)]
    nulls = Parallel(n_jobs=jobs)(delayed(_null)(Xtr, yy, lam, classes, Xe) for yy in perms)
    pi = np.bincount(ytr, minlength=len(classes)) / len(ytr)
    return {"p": probe.proba(Xe), "nulls": nulls, "pi": pi, "layer": L, "lambda": lam,
            "n_train": int(len(sub))}


def gold_label_rows(steps: list[dict], rix: dict) -> dict:
    """Simple steps labelled by the aligned gold step's operation (not the written one)."""
    out = {"ri": [], "y": [], "t": [], "split": [], "step": []}
    for s in steps:
        if s["op"] is None or s["gold_op"] is None:
            continue
        k = (s["trace_id"], s["step"], "step_first")
        if k not in rix:
            continue
        out["ri"].append(rix[k])
        out["y"].append(OPS.index(s["gold_op"]))
        out["t"].append(s["template_id"])
        out["split"].append(s["split"])
        out["step"].append(s)
    return {k: (np.array(v) if k != "step" else v) for k, v in out.items()}


def gold_probe(g: dict, n_min: int, n_shuffle: int, jobs: int, X_eval, log) -> dict:
    classes = np.arange(len(OPS))
    tr_all = np.flatnonzero(g["split"] == "train")
    sub = tr_all[fit.subsample(g["y"][tr_all], min(n_min, len(tr_all)), 0)]
    folds = fit.template_folds(g["t"][sub], seed=0)
    log(f"  gold-labelled op probe: select on {len(sub)} train rows")
    sel = select_parallel(g["ri"][sub], g["y"][sub], folds, classes, jobs)
    sel = {"layer": sel["layer"], "lambda": sel["lambda"], "cv_by_layer": sel["cv_by_layer"],
           "cv_by_lambda": sel["cv_by_lambda"]}
    L, lam = sel["layer"], sel["lambda"]
    Xtr, ytr = load_layer(L, g["ri"][sub]), g["y"][sub]
    Xe = load_layer(L, X_eval)
    probe = fit.fit(Xtr, ytr, lam, classes)
    rng = np.random.default_rng(0)
    perms = [fit.shuffle_within(ytr, g["t"][sub], rng) for _ in range(n_shuffle)]
    nulls = Parallel(n_jobs=jobs)(delayed(_null)(Xtr, yy, lam, classes, Xe) for yy in perms)
    pi = np.bincount(ytr, minlength=len(classes)) / len(ytr)
    return {"p": probe.proba(Xe), "nulls": nulls, "pi": pi, "selection": sel, "n_train": int(len(sub))}


# ---------------------------------------------------------------- stability (R1 samples)
def stability_by_trace(r1_out: Path | None, r1_inputs: Path) -> dict[str, str]:
    if r1_out is None or not r1_out.exists():
        return {}
    from repair.analyze import read_items, stability_labels
    items = read_items(r1_inputs)
    st = stability_labels(r1_out)
    return {items[i]["trace_id"]: lab for i, lab in st.items()
            if i in items and items[i]["role"] == "failure"}


# ---------------------------------------------------------------- assemble
def summarize(F: list[dict], S: dict) -> dict:
    """D_V, G_V, the baselines and the paired differences for one set of failures."""
    t = np.array([f["template_id"] for f in F])
    if len(F) == 0:
        return {"n": 0}
    pick = lambda key: np.array([S[f["trace_id"]][key] for f in F])  # noqa: E731
    D, G = pick("top1"), pick("gob")
    Gs, Gp = pick("gob_shuf"), pick("gob_prior")
    Ds, Dp = pick("top1_shuf"), pick("top1_prior")
    out = {"n": len(F), "templates": int(len(set(t))),
           "D_V": mean_ci(D, t, unit=True), "G_V": mean_ci(G, t, unit=True),
           "D_shuf": mean_ci(Ds, t), "D_prior": mean_ci(Dp, t),
           "G_shuf": mean_ci(Gs, t), "G_prior": mean_ci(Gp, t),
           "G_V_minus_G_shuf": mean_ci(G - Gs, t), "G_V_minus_G_prior": mean_ci(G - Gp, t),
           "bound_top1": float(pick("bound_top1").mean()),
           "gob_prior_corrected": mean_ci(pick("gob_pc"), t),
           "gob_prior_corrected_shuf": mean_ci(pick("gob_pc_shuf"), t),
           "gob_prior_corrected_minus_shuf": mean_ci(pick("gob_pc") - pick("gob_pc_shuf"), t)}
    out["criterion_met"] = bool(out["D_V"]["estimate"] >= MIN_TOP1
                                and out["G_V_minus_G_shuf"]["ci_low"] > 0
                                and out["G_V_minus_G_prior"]["ci_low"] > 0)
    return out


def score_set(F: list[dict], res: dict, priors: dict, key_p: str = "p") -> dict:
    """Per-failure statistics for the real, shuffled and prior readouts of one probe."""
    gold = np.array([f["gold"] for f in F])
    bound = np.array([f["bound"] for f in F])
    real = per_failure(res[key_p], gold, bound, res["pi"])
    sh = [per_failure(q, gold, bound, res["pi"]) for q in res["nulls"]]
    pr = per_failure(np.stack([priors[f["step_pos"]] for f in F]), gold, bound)
    out = {}
    for j, f in enumerate(F):
        out[f["trace_id"]] = {
            "top1": real["top1"][j], "gob": real["gob"][j], "bound_top1": real["bound_top1"][j],
            "gob_pc": real["gob_prior_corrected"][j],
            "top1_shuf": float(np.mean([s["top1"][j] for s in sh])) if sh else float("nan"),
            "gob_shuf": float(np.mean([s["gob"][j] for s in sh])) if sh else float("nan"),
            "gob_pc_shuf": float(np.mean([s["gob_prior_corrected"][j] for s in sh])) if sh else float("nan"),
            "top1_prior": pr["top1"][j], "gob_prior": pr["gob"][j]}
    return out


def run(split="dev", n_shuffle=100, jobs=4, r1_out: Path | None = None,
        r1_inputs: Path = REPO_ROOT / "data" / "generated" / "repair_r1_inputs", log=print) -> dict:
    guard(split)
    t0 = time.time()
    if not E1_JSON.exists():
        raise SystemExit(f"{E1_JSON} is missing: run python -m analysis.e1 first")
    e1 = json.loads(E1_JSON.read_text())
    n_min = int(e1["n_train_equal"])
    steps = step_rows(("train", split))
    rix = row_index(load_rows())
    F_all = failures(steps, split, rix)
    log(f"[{time.time() - t0:.0f}s] {len(F_all)} content-attributable failures on {split}")
    S, probes = {}, {}
    for var in CONTENT:
        F = [f for f in F_all if f["variable"] == var]
        if not F:
            continue
        d = variable_rows(steps, var, rix=rix)
        sel = e1["variables"][var]["primary"]["selection"]
        log(f"[{time.time() - t0:.0f}s] {var}: E1 probe at layer {sel['layer']}, lambda {sel['lambda']}; "
            f"{n_shuffle} shuffles; {len(F)} failures")
        res = content_probe(d, var, sel, n_min, n_shuffle, jobs, np.array([f["ri"] for f in F]))
        S.update(score_set(F, res, step_prior(d, len(VARIABLES[var][1]))))
        probes[var] = {"layer": res["layer"], "lambda": res["lambda"], "n_train": res["n_train"]}
    # secondary: gold-labelled operation probe on the 4-class failures
    F4 = [f for f in F_all if f["variable"] == "content/4-class"]
    S_gold = {}
    if F4:
        g = gold_label_rows(steps, rix)
        res = gold_probe(g, n_min, n_shuffle, jobs, np.array([f["ri"] for f in F4]), log)
        S_gold = score_set(F4, res, step_prior(g, len(OPS)))
        probes["gold-labelled op"] = {"selection": res["selection"], "n_train": res["n_train"]}
    stab = stability_by_trace(r1_out, r1_inputs)
    for f in F_all:
        f["stability"] = stab.get(f["trace_id"])

    pops = sorted({f["population"] for f in F_all})
    out = {"split": split, "status": ("DEVELOPMENT CHECK of the E3 code on dev (prereg/E3.md is a DRAFT); "
                                      "not the G2 gate" if split == "dev" else "TEST"),
           "labels": "automatic (parser v4); S5 kappa pending",
           "criterion": f"D_V >= {MIN_TOP1} and CI(G_V - G_shuf) > 0 and CI(G_V - G_prior) > 0",
           "probes": probes, "n_shuffle": n_shuffle, "counts": {}, "results": {}}
    for pop in pops:
        Fp = [f for f in F_all if f["population"] == pop]
        out["counts"][pop] = {v: sum(f["variable"] == v for f in Fp) for v in CONTENT}
        r = {"pooled": summarize(Fp, S)}
        for v in CONTENT:
            r[v] = summarize([f for f in Fp if f["variable"] == v], S)
        r["gold-labelled op (4-class failures)"] = summarize([f for f in Fp if f["variable"] == "content/4-class"], S_gold)
        if any(f["stability"] for f in Fp):
            for lab in ("stable", "borderline"):
                r[f"pooled, {lab}"] = summarize([f for f in Fp if f["stability"] == lab], S)
        out["results"][pop] = r
    out["stability_source"] = str(r1_out) if stab else "not available (R1 stability samples absent)"
    out["per_failure"] = [{**{k: f[k] for k in ("trace_id", "template_id", "population", "variable",
                                                 "gold", "bound", "step", "stability")},
                           **{k: float(v) for k, v in S[f["trace_id"]].items()}} for f in F_all]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def markdown(r: dict) -> str:
    f = lambda c: "–" if not c or c.get("n", 1) == 0 or not np.isfinite(c["estimate"]) else (  # noqa: E731
        f"{c['estimate']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
    L = [f"# E3 ({r['split']}): gold-value decodability at s0", "", f"**{r['status']}.** {r['labels']}.",
         "", f"Criterion (prereg/E3.md, DRAFT): {r['criterion']}.", "",
         "Probes (layer, lambda, train rows): " + "; ".join(
             f"{k}: L{v.get('layer', v.get('selection', {}).get('layer'))}, "
             f"{v.get('lambda', v.get('selection', {}).get('lambda'))}, {v['n_train']}"
             for k, v in r["probes"].items()),
         f"Stability: {r['stability_source']}.", ""]
    for pop, res in r["results"].items():
        L += [f"## Population: {pop} (counts {r['counts'][pop]})", "",
              "| set | n (templates) | D_V | G_V | G_shuf | G_prior | G_V − G_shuf | G_V − G_prior | bound top-1 | prior-corrected gob | met |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in res.items():
            if not s.get("n"):
                L.append(f"| {name} | 0 | | | | | | | | | |")
                continue
            L.append(f"| {name} | {s['n']} ({s['templates']}) | {f(s['D_V'])} | {f(s['G_V'])} | {f(s['G_shuf'])} | "
                     f"{f(s['G_prior'])} | {f(s['G_V_minus_G_shuf'])} | {f(s['G_V_minus_G_prior'])} | "
                     f"{s['bound_top1']:.3f} | {f(s['gob_prior_corrected'])} | {'yes' if s['criterion_met'] else 'no'} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n-shuffle", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--r1-out", type=Path, default=None, help="R1 output dir with stability_w*.jsonl")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = run(a.split, a.n_shuffle, a.jobs, a.r1_out)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"e3_{a.split}.json").write_text(json.dumps(r, indent=1, default=float))
    (a.out / f"e3_{a.split}.md").write_text(markdown(r))
    print(markdown(r))


if __name__ == "__main__":
    main()
