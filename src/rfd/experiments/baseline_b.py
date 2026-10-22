"""Baseline B: replicate Yuan et al. (2605.09502) on GSM-Symbolic.

  probe  (CPU) failure probes by layer at three positions (last trace token,
         first step token, question only), template-grouped CV on train
         templates, held-out test-template evaluation, shuffled-label control,
         token-uncertainty and TF-IDF baselines, calibration
  steer  (GPU) steering along the failure direction during generation:
         Yuan's projection removal and additive diff-in-means / probe-weight
         directions; strength chosen on train templates; matched-norm random
         directions on held-out test templates

Usage: python -m rfd.experiments.baseline_b {probe|steer} [--harness CFG] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rfd.probing.probes import fit_probe, grouped_oof_scores
from rfd.stats import (
    brier,
    cluster_bootstrap,
    expected_calibration_error,
    safe_auprc,
    safe_auroc,
)
from rfd.utils.config import load_run_config, record_config
from rfd.utils.io import read_jsonl
from rfd.utils.resume import (
    ResultCache,
    StageMarker,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    check_manifest,
    file_fingerprint,
    open_array,
)


def load_cfg(path: str, harness: str | None, out: str | None, traces: str | None) -> dict[str, Any]:
    return load_run_config(path, harness, out, traces)


def _load_traces(tdir: Path) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    traces = list(read_jsonl(tdir / "traces.jsonl"))
    steps = list(read_jsonl(tdir / "steps.jsonl"))
    acts = {
        "last": open_array(tdir, "acts_last"),
        "question": open_array(tdir, "acts_question"),
    }
    # first step token of each trace (traces without steps fall back to the last token)
    first = {}
    for k, s in enumerate(steps):
        first.setdefault(s["uid"], k)
    a_steps = open_array(tdir, "acts_steps")
    idx = [first.get(t["uid"], -1) for t in traces]
    fs = np.stack(
        [a_steps[k] if k >= 0 else acts["last"][i] for i, k in enumerate(idx)]
    )  # [n, L, d]
    acts["first_step"] = fs
    return traces, acts, steps


def _auroc_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray, nb: int, seed: int) -> dict:
    return cluster_bootstrap(lambda i: safe_auroc(y[i], s[i]), groups, nb, seed)


# --------------------------------------------------------------------------- probe


def _traces_inputs(tdir: Path) -> str:
    return file_fingerprint(tdir / "traces.jsonl", tdir / "steps.jsonl", tdir / "acts_index.json")


def cmd_probe(cfg: dict[str, Any], force: bool = False) -> None:
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = Path(cfg["out_dir"])
    tdir = Path(cfg["traces_dir"])
    inputs = _traces_inputs(tdir)
    stage = StageMarker(out, "probe", inputs)
    if stage.skip(force):
        return
    out.mkdir(parents=True, exist_ok=True)
    traces, acts, _ = _load_traces(tdir)
    y = np.array([not t["correct"] for t in traces], int)  # 1 = failure (Yuan: wrong = 1)
    groups = np.array([t["template_id"] for t in traces])
    split = np.array([t["split"] for t in traces])
    variant = np.array([t["variant"] for t in traces])
    tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    n_layers = acts["last"].shape[1]
    C, seed, nb = cfg["probe"]["C"], cfg["seed"], cfg["n_boot"]
    classes = np.array([0, 1])

    def layer_cv(pos: str, layer: int, shuffle: bool) -> float:
        X = np.asarray(acts[pos][tr, layer], np.float32)
        yy = y[tr].copy()
        if shuffle:
            yy = yy[np.random.default_rng(seed + layer).permutation(len(yy))]
        s = grouped_oof_scores(X, yy, groups[tr], "logistic", cfg["probe"]["cv_folds"], C, classes)
        ok = np.isfinite(s[:, 1])
        return safe_auroc(yy[ok], s[ok, 1])

    jobs = [(p, L, False) for p in cfg["probe"]["positions"] for L in range(n_layers)]
    jobs += [("last", L, True) for L in range(n_layers)]
    # each CV job is cached when its batch finishes, so a restart refits only the rest
    cache = ResultCache(out / "cache" / f"probe_cv-{inputs}")
    key = lambda j: f"{j[0]}_L{j[1]}_shuf{int(j[2])}"  # noqa: E731
    todo = [j for j in jobs if not cache.has(key(j))]
    step = max(1, os.cpu_count() or 1)
    for b0 in range(0, len(todo), step):
        batch = todo[b0 : b0 + step]
        for j, v in zip(batch, Parallel(n_jobs=-1)(delayed(layer_cv)(*j) for j in batch)):
            cache.put(key(j), v)
        print(f"probe CV {len(jobs) - len(todo) + b0 + len(batch)}/{len(jobs)}", flush=True)
    cv = {j: cache.get(key(j)) for j in jobs}
    cv = {j: (float("nan") if v is None else v) for j, v in cv.items()}
    curves = {p: [cv[(p, L, False)] for L in range(n_layers)] for p in cfg["probe"]["positions"]}
    shuffled_curve = [cv[("last", L, True)] for L in range(n_layers)]

    results: dict[str, Any] = {
        "n_traces": len(traces),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_train_templates": int(len(np.unique(groups[tr]))),
        "n_test_templates": int(len(np.unique(groups[te]))),
        "failure_rate_train": float(y[tr].mean()),
        "failure_rate_test": float(y[te].mean()),
        "accuracy_test_by_variant": {
            v: float(1 - y[te][variant[te] == v].mean()) for v in np.unique(variant)
        },
        "cv_auroc_by_layer": curves,
        "cv_auroc_by_layer_shuffled_last": shuffled_curve,
        "positions": {},
    }
    directions = {}
    for pos in cfg["probe"]["positions"]:
        if not np.isfinite(curves[pos]).any():
            results["positions"][pos] = {
                "error": "CV AUROC undefined at every layer (one class only in some folds)"
            }
            print(pos, "skipped: insufficient label variety on train templates", flush=True)
            continue
        L = int(np.nanargmax(curves[pos]))
        X_tr = np.asarray(acts[pos][tr, L], np.float32)
        X_te = np.asarray(acts[pos][te, L], np.float32)
        res_pos: dict[str, Any] = {"layer": L, "cv_auroc_train": curves[pos][L]}
        for kind in ("logistic", "diffmeans"):
            p = fit_probe(X_tr, y[tr], kind, C)
            s = p.scores(X_te)[:, 1]
            ys = y[tr][np.random.default_rng(seed + 1).permutation(len(tr))]
            ps = fit_probe(X_tr, ys, kind, C)
            ss = ps.scores(X_te)[:, 1]
            res_pos[kind] = {
                "test_auroc": _auroc_ci(y[te], s, groups[te], nb, seed),
                "test_auprc": cluster_bootstrap(
                    lambda i, s=s: safe_auprc(y[te][i], s[i]), groups[te], nb, seed
                ),
                "test_auroc_shuffled": safe_auroc(y[te], ss),
                "auroc_minus_shuffled": cluster_bootstrap(
                    lambda i, s=s, ss=ss: safe_auroc(y[te][i], s[i]) - safe_auroc(y[te][i], ss[i]),
                    groups[te],
                    nb,
                    seed,
                ),
                "auroc_by_variant": {
                    v: safe_auroc(y[te][variant[te] == v], s[variant[te] == v])
                    for v in np.unique(variant)
                },
            }
            if kind == "logistic":
                res_pos[kind]["ece"] = expected_calibration_error(y[te], s)
                res_pos[kind]["brier"] = brier(y[te], s)
                res_pos[kind]["brier_base_rate"] = brier(y[te], np.full(len(te), y[tr].mean()))
            if pos == "last":
                directions[kind] = p.direction()  # raw space, points toward FAILURE
        results["positions"][pos] = res_pos
        if pos == "last":
            mu_c = X_tr[y[tr] == 0].mean(0)
            mu_w = X_tr[y[tr] == 1].mean(0)
            directions["layer"] = L
            directions["mu_correct_minus_wrong"] = mu_c - mu_w
            directions["resid_norm_mean"] = float(np.linalg.norm(X_tr, axis=1).mean())

    # token-uncertainty baselines (sign oriented on train) and a TF-IDF text baseline
    def tok_stat(t: dict, name: str) -> float:
        lp, ent = np.array(t["token_logprobs"] or [0.0]), np.array(t["token_entropies"] or [0.0])
        return {
            "mean_logprob": -lp.mean(),
            "min_logprob": -lp.min(),
            "seq_logprob": -lp.sum(),
            "mean_entropy": ent.mean(),
            "max_entropy": ent.max(),
            "length": float(len(lp)),
        }[name]

    base = {}
    for name in (
        "mean_logprob",
        "min_logprob",
        "seq_logprob",
        "mean_entropy",
        "max_entropy",
        "length",
    ):
        s_all = np.array([tok_stat(t, name) for t in traces])
        sign = 1.0 if safe_auroc(y[tr], s_all[tr]) >= 0.5 else -1.0
        base[name] = {
            "sign": sign,
            "test_auroc": _auroc_ci(y[te], sign * s_all[te], groups[te], nb, seed),
        }
    texts = [t["text"] for t in traces]
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit([texts[i] for i in tr])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.transform([texts[i] for i in tr]), y[tr])
    s_txt = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
    base["tfidf_trace_text"] = {"test_auroc": _auroc_ci(y[te], s_txt, groups[te], nb, seed)}
    results["baselines"] = base

    atomic_write_json(out / "probe.json", results)
    _atomic_savez(out / "directions.npz", {k: np.asarray(v) for k, v in directions.items()})
    atomic_write_text(
        out / "directions_meta.json",
        json.dumps(
            {
                "layer": int(directions["layer"]),
                "norm_mu_diff": float(np.linalg.norm(directions["mu_correct_minus_wrong"])),
                "resid_norm_mean": directions["resid_norm_mean"],
                "cos_logistic_vs_diffmeans": float(
                    -np.dot(directions["logistic"], directions["mu_correct_minus_wrong"])
                    / np.linalg.norm(directions["logistic"])
                    / np.linalg.norm(directions["mu_correct_minus_wrong"])
                ),
            },
            indent=2,
        ),
    )
    from rfd.plotting import auroc_by_layer

    auroc_by_layer(
        {
            "last token": curves["last"],
            "first step": curves.get("first_step", []),
            "question only": curves.get("question", []),
        },
        shuffled_curve,
        out / "fig_b_probe_auroc_by_layer.png",
        "Failure probe, train-template CV AUROC by layer",
    )
    stage.mark()
    print(
        json.dumps(
            {k: results[k] for k in ("n_traces", "failure_rate_test", "accuracy_test_by_variant")}
        )
    )
    for pos, r in results["positions"].items():
        if "error" in r:
            continue
        print(
            pos,
            "layer",
            r["layer"],
            "test AUROC",
            round(r["logistic"]["test_auroc"]["estimate"], 3),
            "CI",
            round(r["logistic"]["test_auroc"]["lo"], 3),
            round(r["logistic"]["test_auroc"]["hi"], 3),
        )


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- steer


def _stratified(idx: np.ndarray, correct: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in (False, True):
        pool = idx[correct[idx] == c]
        out.append(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.sort(np.concatenate(out))


def natural_net(orig: np.ndarray, new: np.ndarray, p_wrong: float) -> float:
    """Accuracy change at the population failure rate: p_w * repair - (1 - p_w) * damage."""
    o, n = orig.astype(bool), new.astype(bool)
    rep = (~o & n).sum() / max((~o).sum(), 1)
    dmg = (o & ~n).sum() / max(o.sum(), 1)
    return float(p_wrong * rep - (1 - p_wrong) * dmg)


def cmd_steer(cfg: dict[str, Any], force: bool = False) -> None:
    import torch

    from rfd.harness.model import library_versions, load
    from rfd.steering.engine import SteerSpec, outcome_rates, random_directions, run_condition
    from rfd.utils.seed import set_seed

    out = Path(cfg["out_dir"])
    tdir = Path(cfg["traces_dir"])
    inputs = file_fingerprint(tdir / "traces.jsonl", out / "directions.npz")
    stage = StageMarker(out, "steer", inputs)
    if stage.skip(force):
        return
    set_seed(cfg["seed"])
    check_manifest(out, cfg["harness"]["model"])
    traces = list(read_jsonl(tdir / "traces.jsonl"))
    dirs = dict(np.load(out / "directions.npz"))
    L = int(dirs["layer"])
    mu_diff = torch.tensor(dirs["mu_correct_minus_wrong"], dtype=torch.float32)
    w_fail = torch.tensor(dirs["logistic"], dtype=torch.float32)  # toward failure
    norm = float(mu_diff.norm())
    correct = np.array([t["correct"] for t in traces])
    split = np.array([t["split"] for t in traces])
    groups = np.array([t["template_id"] for t in traces])
    sc = cfg["steer"]
    seed = cfg["seed"]
    e_train = _stratified(
        np.flatnonzero(split == "train"), correct, sc["n_eval_train_per_class"], seed
    )
    e_test = _stratified(
        np.flatnonzero(split == "test"), correct, sc["n_eval_test_per_class"], seed + 1
    )
    p_wrong_train = float(1 - correct[split == "train"].mean())
    p_wrong_test = float(1 - correct[split == "test"].mean())

    loaded = load(cfg["harness"]["model"])
    model, tok = loaded.model, loaded.tokenizer
    max_new = cfg["harness"]["generation"]["max_new_tokens"]
    bs = cfg["harness"]["forward_batch_size"]
    eos = {tok.eos_token_id}
    ge = getattr(model.generation_config, "eos_token_id", None)
    if ge is not None:
        eos |= set(ge if isinstance(ge, list) else [ge])

    def spec_for(kind: str, strength: float, vec: torch.Tensor | None = None) -> SteerSpec:
        if kind == "yuan_project":
            return SteerSpec(L, "project", w_fail if vec is None else vec, strength, name=kind)
        if kind == "add_diffmeans":
            v = (mu_diff / norm) if vec is None else vec / vec.norm()
            return SteerSpec(L, "add", v, strength * norm, name=kind)
        if kind == "add_probe":
            v = (-w_fail / w_fail.norm()) if vec is None else vec / vec.norm()
            return SteerSpec(L, "add", v, strength * norm, name=kind)
        raise ValueError(kind)

    # one cache file per evaluated condition: a restart re-runs only unfinished ones
    cache = ResultCache(out / "cache" / f"steer-{inputs}")

    def evaluate(
        idx: np.ndarray, spec: SteerSpec, key: str, keep_texts: bool = False
    ) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            sub = [traces[i] for i in idx]
            res = run_condition(model, tok, sub, spec, max_new, batch_size=bs, eos_ids=eos)
            return {
                "uids": [t["uid"] for t in sub],
                "new_correct": [bool(r.correct) for r in res],
                "n_diverged": int(sum(r.diverged for r in res)),
                "texts": [r.text for r in res] if keep_texts else None,
            }

        r = cache.get_or_compute(f"{key}|bs{bs}", compute)
        if r["uids"] != [traces[i]["uid"] for i in idx]:
            raise RuntimeError(f"cached condition {key} was run on other traces")
        return {**r, "new_correct": np.array(r["new_correct"], bool)}

    kinds = {
        "yuan_project": sc["project_alphas"],
        "add_diffmeans": sc["add_multiples"],
        "add_probe": sc["add_multiples"],
    }
    log: dict[str, Any] = {
        "layer": L,
        "norm_mu_diff": norm,
        "n_eval_train": int(len(e_train)),
        "n_eval_test": int(len(e_test)),
        "p_wrong_train": p_wrong_train,
        "p_wrong_test": p_wrong_test,
        "versions": library_versions(),
    }
    # 0. numerical-noise control: zero strength must not change any trace
    zero = evaluate(e_test, spec_for("add_diffmeans", 0.0), "zero")
    log["zero_strength_n_diverged"] = zero["n_diverged"]

    # 1. strength selection on TRAIN templates
    selection: dict[str, Any] = {}
    for kind, grid in kinds.items():
        rows = []
        for s in grid:
            r = evaluate(e_train, spec_for(kind, s), f"train|{kind}|{s}")
            net = natural_net(correct[e_train], r["new_correct"], p_wrong_train)
            rows.append(
                {
                    "strength": s,
                    "net_natural": net,
                    **outcome_rates(correct[e_train], r["new_correct"]),
                    "n_diverged": r["n_diverged"],
                }
            )
            print(kind, "train", s, rows[-1], flush=True)
        best = max(rows, key=lambda r: (round(r["net_natural"], 6), -r["strength"]))
        selection[kind] = {"grid": rows, "selected": best["strength"]}
    log["selection_train"] = selection

    # 2. held-out evaluation at the selected strength + matched-norm random directions
    d = mu_diff.shape[0]
    test: dict[str, Any] = {}
    per_trace = []
    for kind in kinds:
        s = selection[kind]["selected"]
        r = evaluate(e_test, spec_for(kind, s), f"test|{kind}|{s}", keep_texts=True)
        orig = correct[e_test]
        new = r["new_correct"]
        items_groups = groups[e_test]
        o_, n_ = orig.astype(bool), new.astype(bool)
        real = {
            "strength": s,
            **outcome_rates(orig, new),
            "net_natural": natural_net(orig, new, p_wrong_test),
            "n_diverged": r["n_diverged"],
            "repair_ci": cluster_bootstrap(
                lambda i: (~o_[i] & n_[i]).sum() / max((~o_[i]).sum(), 1),
                items_groups,
                cfg["n_boot"],
                seed,
            ),
            "damage_ci": cluster_bootstrap(
                lambda i: (o_[i] & ~n_[i]).sum() / max(o_[i].sum(), 1),
                items_groups,
                cfg["n_boot"],
                seed,
            ),
            "net_natural_ci": cluster_bootstrap(
                lambda i: natural_net(o_[i], n_[i], p_wrong_test), items_groups, cfg["n_boot"], seed
            ),
        }
        null_nets, null_rep, null_dmg = [], [], []
        if kind == "yuan_project":
            rand = random_directions(d, sc["n_random"], 1.0, seed + 100)
        else:
            rand = random_directions(d, sc["n_random"], 1.0, seed + 200 + len(kind))
        for k in range(sc["n_random"]):
            rr = evaluate(e_test, spec_for(kind, s, rand[k]), f"test|{kind}|{s}|random{k}")
            null_nets.append(natural_net(orig, rr["new_correct"], p_wrong_test))
            rates = outcome_rates(orig, rr["new_correct"])
            null_rep.append(rates["repair"])
            null_dmg.append(rates["damage"])
            if k % 10 == 0:
                print(kind, "random", k, null_nets[-1], flush=True)
        nn = np.array(null_nets)
        real["null"] = {
            "n": int(len(nn)),
            "net_mean": float(nn.mean()),
            "net_q025": float(np.quantile(nn, 0.025)),
            "net_q975": float(np.quantile(nn, 0.975)),
            "repair_mean": float(np.mean(null_rep)),
            "damage_mean": float(np.mean(null_dmg)),
            "p_value_net_ge": float((1 + (nn >= real["net_natural"]).sum()) / (1 + len(nn))),
        }
        test[kind] = real
        for i, t_new, txt in zip(e_test, new, r["texts"]):
            per_trace.append(
                {
                    "uid": traces[i]["uid"],
                    "kind": kind,
                    "strength": s,
                    "orig": bool(correct[i]),
                    "new": bool(t_new),
                    "text": txt,
                }
            )
        print(
            kind,
            "TEST",
            {k: v for k, v in real.items() if not isinstance(v, dict)},
            real["null"],
            flush=True,
        )
    log["test"] = test
    atomic_write_jsonl(out / "steer_test_traces.jsonl", per_trace)

    # 3. descriptive sweep on TEST with a random band (figure only)
    sweep = {}
    for kind, grid in kinds.items():
        rows = {
            "strengths": grid,
            "repair": [],
            "damage": [],
            "net": [],
            "null_lo": None,
            "null_hi": None,
        }
        band_lo, band_hi = [], []
        for s in grid:
            r = evaluate(e_test, spec_for(kind, s), f"test|{kind}|{s}")
            rates = outcome_rates(correct[e_test], r["new_correct"])
            rows["repair"].append(rates["repair"])
            rows["damage"].append(rates["damage"])
            rows["net"].append(natural_net(correct[e_test], r["new_correct"], p_wrong_test))
            if kind in sc["band_directions"]:
                rand = random_directions(d, sc["n_band"], 1.0, seed + 300 + int(s * 10))
                nets = [
                    natural_net(
                        correct[e_test],
                        evaluate(e_test, spec_for(kind, s, rand[k]), f"sweep|{kind}|{s}|band{k}")[
                            "new_correct"
                        ],
                        p_wrong_test,
                    )
                    for k in range(sc["n_band"])
                ]
                band_lo.append(100 * float(np.min(nets)))
                band_hi.append(100 * float(np.max(nets)))
        if band_lo:
            rows["null_lo"], rows["null_hi"] = band_lo, band_hi
        sweep[kind] = rows
    log["sweep_test_descriptive"] = sweep
    atomic_write_json(out / "steer.json", log)
    from rfd.plotting import strength_sweep

    strength_sweep(
        {k: {**v, "net": v["net"]} for k, v in sweep.items()},
        out / "fig_b_steering_vs_strength.png",
        f"Steering along the failure direction (layer {L}), held-out test templates; "
        "grey band = min-max over random directions",
    )
    stage.mark()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "steer"])
    ap.add_argument("--config", default="configs/baseline_b.yaml")
    ap.add_argument("--harness", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--n-random", type=int, default=None, help="debug: fewer random directions")
    ap.add_argument("--per-class", type=int, default=None, help="debug: smaller eval sets")
    ap.add_argument("--force", action="store_true", help="redo a stage that is already complete")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.harness, args.out, args.traces)
    record_config(cfg)
    if args.n_random is not None:
        cfg["steer"]["n_random"] = args.n_random
        cfg["steer"]["n_band"] = min(cfg["steer"]["n_band"], args.n_random)
    if args.per_class is not None:
        cfg["steer"]["n_eval_train_per_class"] = args.per_class
        cfg["steer"]["n_eval_test_per_class"] = args.per_class
    {
        "probe": lambda: cmd_probe(cfg, args.force),
        "steer": lambda: cmd_steer(cfg, args.force),
    }[args.cmd]()


if __name__ == "__main__":
    main()
