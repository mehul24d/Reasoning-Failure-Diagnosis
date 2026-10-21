"""Phase 3 PILOT: feasibility slice of self-sourced repair on distractor failures.

  readability  (CPU) four step-level variables (2x2: content/evaluation x
               binary/4-class) probed at the first token of each step with the
               same procedure; layer x step-position maps; held-out tests vs
               shuffled labels; RQ3 gold-value decodability in
               content-attributable failures
  sample       (GPU, optional) T=0.7 x k samples per failure -> stable/borderline
  repair       (GPU) edits at the first deviating step: oracle content,
               self-sourced gated, evaluation->correct, oracle residual patch,
               matched-norm random null; repair / damage / net gain

Usage: python -m rfd.experiments.pilot {readability|sample|repair} [--harness CFG] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rfd.probing.probes import fit_probe, grouped_oof_scores, metric
from rfd.stats import cluster_bootstrap
from rfd.utils.config import load_yaml
from rfd.utils.io import read_jsonl, write_jsonl

OPS = ["+", "-", "*", "/"]
ERRS = ["none", "planning", "binding", "arithmetic"]
VARIABLES = {
    # name: (family, arity)
    "content/binary: distractor used": ("content", 2),
    "content/4-class: operation": ("content", 4),
    "evaluation/binary: step correct": ("evaluation", 2),
    "evaluation/4-class: error type": ("evaluation", 4),
}


def load_cfg(path: str, harness: str | None, out: str | None, traces: str | None) -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["harness"] = load_yaml(harness or cfg["harness_config"])
    if out:
        cfg["out_dir"] = out
    if traces:
        cfg["traces_dir"] = traces
    return cfg


def _labels(steps: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """variable -> (row indices into steps, integer labels)."""
    out = {}
    parsed = [k for k, s in enumerate(steps) if s["error_type"] is not None]
    noop = [k for k in parsed if steps[k]["variant"] == "noop"]
    out["content/binary: distractor used"] = (
        np.array(noop),
        np.array([int(steps[k]["distractor_used"]) for k in noop]),
    )
    op_rows = [k for k in parsed if steps[k]["op"] in OPS]
    out["content/4-class: operation"] = (
        np.array(op_rows),
        np.array([OPS.index(steps[k]["op"]) for k in op_rows]),
    )
    out["evaluation/binary: step correct"] = (
        np.array(parsed),
        np.array([int(steps[k]["error_type"] == "none") for k in parsed]),
    )
    out["evaluation/4-class: error type"] = (
        np.array(parsed),
        np.array([ERRS.index(steps[k]["error_type"]) for k in parsed]),
    )
    return out


def _bucket(pos: np.ndarray, n: int) -> np.ndarray:
    return np.minimum(pos, n - 1)


# --------------------------------------------------------------------------- readability


def cmd_readability(cfg: dict[str, Any]) -> None:
    from joblib import Parallel, delayed

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["traces_dir"])
    steps = list(read_jsonl(tdir / "steps.jsonl"))
    traces = {t["uid"]: t for t in read_jsonl(tdir / "traces.jsonl")}
    acts = np.load(tdir / "acts_steps.npy", mmap_mode="r")
    n_layers = acts.shape[1]
    seed, C, nb = cfg["seed"], cfg["probe"]["C"], cfg["n_boot"]
    split = np.array([s["split"] for s in steps])
    groups = np.array([s["template_id"] for s in steps])
    spos = _bucket(np.array([s["step_pos"] for s in steps]), cfg["probe"]["step_buckets"])
    labels = _labels(steps)

    # same train size for every variable: subsample train rows to the smallest count
    tr_rows = {v: rows[split[rows] == "train"] for v, (rows, _) in labels.items()}
    n_eq = int(min(len(r) for r in tr_rows.values()))
    rng = np.random.default_rng(seed)
    tr_eq = {v: np.sort(rng.choice(r, n_eq, replace=False)) for v, r in tr_rows.items()}
    lab_of = {v: dict(zip(rows.tolist(), y.tolist())) for v, (rows, y) in labels.items()}

    def y_of(v: str, rows: np.ndarray) -> np.ndarray:
        return np.array([lab_of[v][int(r)] for r in rows])

    def job(v: str, kind: str, layer: int, shuffle: bool) -> dict[str, Any]:
        k = VARIABLES[v][1]
        classes = np.arange(k)
        rows = tr_eq[v]
        X = np.asarray(acts[rows, layer], np.float32)
        y = y_of(v, rows)
        if shuffle:
            y = y[np.random.default_rng(seed + layer).permutation(len(y))]
        oof = grouped_oof_scores(X, y, groups[rows], kind, cfg["probe"]["cv_folds"], C, classes)
        ok = np.all(np.isfinite(oof), 1)
        cv = metric(y[ok], oof[ok], classes) if ok.any() else float("nan")
        # held-out map: fit on train, score test templates by step-position bucket
        test_rows = labels[v][0][split[labels[v][0]] == "test"]
        cells = [float("nan")] * cfg["probe"]["step_buckets"]
        if len(np.unique(y)) == k and len(test_rows):
            p = fit_probe(X, y, kind, C)
            Xt = np.asarray(acts[test_rows, layer], np.float32)
            st, yt = p.scores(Xt), y_of(v, test_rows)
            full = np.zeros((len(test_rows), k))
            full[:, p.classes] = st
            for b in range(cfg["probe"]["step_buckets"]):
                m = spos[test_rows] == b
                if m.sum() >= 10:
                    cells[b] = metric(yt[m], full[m], classes)
        return {"var": v, "kind": kind, "layer": layer, "shuffle": shuffle, "cv": cv, "map": cells}

    jobs = [
        (v, kind, L, sh)
        for v in VARIABLES
        for kind in cfg["probe"]["kinds"]
        for L in range(n_layers)
        for sh in (False, True)
    ]
    res = Parallel(n_jobs=-1, verbose=2)(delayed(job)(*j) for j in jobs)
    by = {(r["var"], r["kind"], r["layer"], r["shuffle"]): r for r in res}

    summary: dict[str, Any] = {
        "n_steps": len(steps),
        "n_train_per_variable": n_eq,
        "label_counts_train": {
            v: np.bincount(y_of(v, tr_eq[v]), minlength=VARIABLES[v][1]).tolist() for v in VARIABLES
        },
        "variables": {},
    }
    maps = {}
    selected = {}
    for v in VARIABLES:
        k = VARIABLES[v][1]
        classes = np.arange(k)
        per_kind = {}
        for kind in cfg["probe"]["kinds"]:
            cvs = [by[(v, kind, L, False)]["cv"] for L in range(n_layers)]
            L = int(np.nanargmax(cvs))
            rows = tr_eq[v]
            X = np.asarray(acts[rows, L], np.float32)
            y = y_of(v, rows)
            p = fit_probe(X, y, kind, C)
            ys = y[np.random.default_rng(seed + 5).permutation(len(y))]
            ps = fit_probe(X, ys, kind, C)
            test_rows = labels[v][0][split[labels[v][0]] == "test"]
            Xt = np.asarray(acts[test_rows, L], np.float32)
            yt = y_of(v, test_rows)

            def full_scores(probe: Any) -> np.ndarray:
                s = np.zeros((len(test_rows), k))
                s[:, probe.classes] = probe.scores(Xt)
                return s

            st, ss = full_scores(p), full_scores(ps)
            g = groups[test_rows]
            per_kind[kind] = {
                "layer": L,
                "cv_by_layer": cvs,
                "cv_shuffled_by_layer": [by[(v, kind, LL, True)]["cv"] for LL in range(n_layers)],
                "test_metric": cluster_bootstrap(
                    lambda i, st=st: metric(yt[i], st[i], classes), g, nb, seed
                ),
                "test_metric_shuffled": metric(yt, ss, classes),
                "metric_minus_shuffled": cluster_bootstrap(
                    lambda i, st=st, ss=ss: metric(yt[i], st[i], classes)
                    - metric(yt[i], ss[i], classes),
                    g,
                    nb,
                    seed,
                ),
                "n_test": int(len(test_rows)),
                "n_test_templates": int(len(np.unique(g))),
                "metric_name": "AUROC" if k == 2 else "balanced accuracy",
            }
            if kind == "logistic":
                maps[v] = np.array(
                    [by[(v, kind, LL, False)]["map"] for LL in range(n_layers)], float
                )
                selected[v] = (L, p)
        summary["variables"][v] = per_kind

    # ---- RQ3: is the GOLD content value top-1 decodable at the first deviating step?
    first_dev: dict[str, int] = {}
    for k_, s in enumerate(steps):
        if s["error_type"] not in (None, "none") and s["uid"] not in first_dev:
            first_dev[s["uid"]] = k_
    pairs: dict[str, dict[str, dict]] = {}
    for t in traces.values():
        pairs.setdefault(t["pair_uid"], {})[t["variant"]] = t
    fail_uids = [
        p["noop"]["uid"]
        for p in pairs.values()
        if "noop" in p and "main" in p and p["main"]["correct"] and not p["noop"]["correct"]
    ]
    rq3_rows = []
    for uid in fail_uids:
        k_ = first_dev.get(uid)
        if k_ is None:
            continue
        s = steps[k_]
        if s["distractor_used"]:
            rq3_rows.append((k_, "content/binary: distractor used", 0, 1))  # gold: not used
        elif s["gold_op"] in OPS and s["op"] in OPS and s["op"] != s["gold_op"]:
            rq3_rows.append(
                (k_, "content/4-class: operation", OPS.index(s["gold_op"]), OPS.index(s["op"]))
            )
    rq3 = {
        "n_distractor_failures": len(fail_uids),
        "n_with_first_deviation": sum(first_dev.get(u) is not None for u in fail_uids),
        "n_content_attributable": len(rq3_rows),
        "by_split": {},
    }
    op_prior = {}
    for b in range(cfg["probe"]["step_buckets"]):
        rows = labels["content/4-class: operation"][0]
        rows = rows[(split[rows] == "train") & (spos[rows] == b)]
        if len(rows):
            op_prior[b] = int(
                np.bincount(y_of("content/4-class: operation", rows), minlength=4).argmax()
            )
    for sp in ("train", "test"):
        rr = [r for r in rq3_rows if steps[r[0]]["split"] == sp]
        if not rr:
            rq3["by_split"][sp] = {"n": 0}
            continue
        hit, hit_shuf, hit_bound, hit_prior, grp = [], [], [], [], []
        for k_, v, gold, bound in rr:
            L, p = selected[v]
            x = np.asarray(acts[k_, L], np.float32)[None]
            rows = tr_eq[v]
            ys = y_of(v, rows)[np.random.default_rng(seed + 5).permutation(len(rows))]
            ps = fit_probe(np.asarray(acts[rows, L], np.float32), ys, "logistic", cfg["probe"]["C"])
            top = int(p.predict(x)[0])
            hit.append(top == gold)
            hit_bound.append(top == bound)
            hit_shuf.append(int(ps.predict(x)[0]) == gold)
            if v == "content/4-class: operation":
                hit_prior.append(op_prior.get(int(spos[k_]), -1) == gold)
            grp.append(steps[k_]["template_id"])
        h, hs, g = np.array(hit, float), np.array(hit_shuf, float), np.array(grp)
        rq3["by_split"][sp] = {
            "n": len(rr),
            "n_templates": int(len(np.unique(g))),
            "by_variable": {
                v: sum(r[1] == v for r in rr) for v in VARIABLES if VARIABLES[v][0] == "content"
            },
            "gold_top1_rate": cluster_bootstrap(lambda i: h[i].mean(), g, nb, seed),
            "gold_top1_rate_shuffled": cluster_bootstrap(lambda i: hs[i].mean(), g, nb, seed),
            "difference": cluster_bootstrap(lambda i: h[i].mean() - hs[i].mean(), g, nb, seed),
            "bound_top1_rate": float(np.mean(hit_bound)),
            "op_step_position_prior_rate": float(np.mean(hit_prior)) if hit_prior else None,
        }
    summary["rq3"] = rq3
    (out / "readability.json").write_text(json.dumps(summary, indent=2, default=float))
    write_jsonl(
        out / "content_failures.jsonl",
        [
            {
                "step_row": int(k_),
                "uid": steps[k_]["uid"],
                "variable": v,
                "gold": int(gold),
                "bound": int(bound),
                "split": steps[k_]["split"],
            }
            for k_, v, gold, bound in rq3_rows
        ],
    )
    from rfd.plotting import readability_maps

    cols = [str(b + 1) for b in range(cfg["probe"]["step_buckets"] - 1)] + [
        f"{cfg['probe']['step_buckets']}+"
    ]
    readability_maps(
        maps,
        out / "fig_pilot_readability_maps.png",
        cols,
        chance={v: 0.5 if VARIABLES[v][1] == 2 else 0.25 for v in VARIABLES},
    )
    for v, r in summary["variables"].items():
        lg = r["logistic"]
        print(
            v,
            "layer",
            lg["layer"],
            lg["metric_name"],
            round(lg["test_metric"]["estimate"], 3),
            "shuffled",
            round(lg["test_metric_shuffled"], 3),
        )
    print(json.dumps(rq3, indent=2, default=float))


# --------------------------------------------------------------------------- sample


def cmd_sample(cfg: dict[str, Any]) -> None:
    import torch

    from rfd.generation.answer_extraction import extract_predicted_answer, is_correct
    from rfd.harness.model import load
    from rfd.utils.seed import set_seed

    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    tdir = Path(cfg["traces_dir"])
    traces = {t["uid"]: t for t in read_jsonl(tdir / "traces.jsonl")}
    fails = list(read_jsonl(out / "content_failures.jsonl"))
    loaded = load(cfg["harness"]["model"])
    model, tok = loaded.model, loaded.tokenizer
    k, T = cfg["sample"]["k"], cfg["sample"]["temperature"]
    rows = []
    for f in fails:
        t = traces[f["uid"]]
        ids = torch.tensor([t["prompt_ids"]] * k, device=model.device)
        with torch.no_grad():
            o = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=T,
                top_p=1.0,
                max_new_tokens=cfg["harness"]["generation"]["max_new_tokens"],
                pad_token_id=tok.pad_token_id,
            )
        texts = tok.batch_decode(o[:, ids.shape[1] :], skip_special_tokens=True)
        n_ok = sum(is_correct(extract_predicted_answer(x), str(t["gold_answer"])) for x in texts)
        rows.append({"uid": f["uid"], "n_correct_of_k": int(n_ok), "k": k, "stable": n_ok == 0})
    write_jsonl(out / "stability.jsonl", rows)
    print(json.dumps({"n": len(rows), "n_stable": sum(r["stable"] for r in rows)}))


# --------------------------------------------------------------------------- repair


def cmd_repair(cfg: dict[str, Any]) -> None:
    import torch

    from rfd.harness.model import library_versions, load
    from rfd.steering.engine import SteerSpec, random_directions, run_condition
    from rfd.utils.seed import set_seed

    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    tdir = Path(cfg["traces_dir"])
    steps = list(read_jsonl(tdir / "steps.jsonl"))
    traces = list(read_jsonl(tdir / "traces.jsonl"))
    tix = {t["uid"]: i for i, t in enumerate(traces)}
    acts = np.load(tdir / "acts_steps.npy", mmap_mode="r")
    summ = json.loads((out / "readability.json").read_text())
    fails = list(read_jsonl(out / "content_failures.jsonl"))
    labels = _labels(steps)
    split = np.array([s["split"] for s in steps])
    seed, nb = cfg["seed"], cfg["n_boot"]
    rng = np.random.default_rng(seed)

    # class means (raw space) per content variable at its selected layer, train templates only
    def class_means(v: str) -> tuple[int, np.ndarray]:
        L = summ["variables"][v]["logistic"]["layer"]
        rows, y = labels[v]
        m = split[rows] == "train"
        X = np.asarray(acts[rows[m], L], np.float32)
        # a class absent from train rows has no mean: edits involving it are skipped
        return L, np.stack(
            [
                X[y[m] == c].mean(0)
                if (y[m] == c).any()
                else np.full(X.shape[1], np.nan, np.float32)
                for c in range(VARIABLES[v][1])
            ]
        )

    means = {v: class_means(v) for v in VARIABLES}
    probes = {}
    for v in ("content/binary: distractor used", "content/4-class: operation"):
        L, _ = means[v]
        rows, y = labels[v]
        m = split[rows] == "train"
        probes[v] = fit_probe(
            np.asarray(acts[rows[m], L], np.float32), y[m], "logistic", cfg["probe"]["C"]
        )

    # matched solved set: correct noop traces from the same templates as the failures
    by_template: dict[int, list[dict]] = {}
    for f in fails:
        by_template.setdefault(steps[f["step_row"]]["template_id"], []).append(f)
    steps_of: dict[str, list[int]] = {}
    for k_, s in enumerate(steps):
        steps_of.setdefault(s["uid"], []).append(k_)
    solved = []
    for tmpl, fl in by_template.items():
        pool = [
            t
            for t in traces
            if t["template_id"] == tmpl and t["variant"] == "noop" and t["correct"]
        ]
        for t in rng.permutation(len(pool))[: len(fl)]:
            tr = pool[int(t)]
            f = fl[int(rng.integers(len(fl)))]
            cand = [k_ for k_ in steps_of.get(tr["uid"], []) if steps[k_]["op"] in OPS]
            if not cand:
                continue
            pos = min(steps[f["step_row"]]["step_pos"], len(cand) - 1)
            solved.append(
                {
                    "step_row": cand[pos],
                    "uid": tr["uid"],
                    "variable": f["variable"],
                    "split": tr["split"],
                    "solved": True,
                }
            )
    items = [dict(f, solved=False) for f in fails] + solved

    loaded = load(cfg["harness"]["model"])
    model, tok = loaded.model, loaded.tokenizer
    max_new = cfg["harness"]["generation"]["max_new_tokens"]
    bs = cfg["harness"]["forward_batch_size"]
    eos = {tok.eos_token_id}

    # oracle patch source: residual at the same step index in the matched MAIN (correct) run
    def patch_vec(item: dict, L: int) -> np.ndarray | None:
        s = steps[item["step_row"]]
        main_uid = s["pair_uid"] + "-main"
        cand = steps_of.get(main_uid, [])
        if not cand or main_uid not in tix:
            return None
        k_ = cand[min(s["step_pos"], len(cand) - 1)]
        return np.asarray(acts[k_, L], np.float32)

    def build(item: dict) -> dict[str, Any]:
        s = steps[item["step_row"]]
        v = item["variable"]
        L, mu = means[v]
        x = np.asarray(acts[item["step_row"], L], np.float32)
        if v == "content/binary: distractor used":
            bound = int(s["distractor_used"])
            gold = 0
        else:
            bound = OPS.index(s["op"])
            gold = OPS.index(s["gold_op"]) if s["gold_op"] in OPS else bound
        if item["solved"]:
            gold = bound  # a solved step's content already equals gold
        decoded = int(probes[v].predict(x[None])[0])
        oracle = mu[gold] - mu[bound]
        ref_norm = (
            float(np.linalg.norm(mu[0] - mu[1]))
            if v.startswith("content/binary")
            else float(
                np.mean(
                    [np.linalg.norm(mu[a] - mu[b]) for a in range(4) for b in range(4) if a != b]
                )
            )
        )
        self_src = mu[decoded] - mu[bound]
        Le, mue = means["evaluation/binary: step correct"]
        evald = mue[1] - mue[0]
        # NaN = a class without train examples -> that edit is skipped (zero vector)
        oracle, self_src, evald = (
            np.nan_to_num(v_, nan=0.0) if np.isnan(v_).any() else v_
            for v_ in (oracle, self_src, evald)
        )
        if np.isnan(ref_norm):
            ref_norm = float(
                np.nanmean(
                    [
                        np.linalg.norm(mu[a] - mu[b])
                        for a in range(len(mu))
                        for b in range(len(mu))
                        if a != b
                    ]
                )
            )
        return {
            "trace": traces[tix[s["uid"]]],
            "at": s["gen_token_index"],
            "L": L,
            "Le": Le,
            "ref_norm": ref_norm,
            "oracle": oracle,
            "self": self_src,
            "self_active": decoded != bound,
            "eval": evald,
            "patch": patch_vec(item, L),
            "solved": item["solved"],
            "split": item["split"],
            "template": s["template_id"],
            "variable": v,
        }

    built = [build(it) for it in items]
    changed = np.zeros(len(built), bool)  # filled by run(); prefix of length len(sub) is valid

    def run(
        sub: list[dict],
        vec_key: str,
        mult: float,
        layer_key: str = "L",
        rand: np.ndarray | None = None,
    ) -> np.ndarray:
        """Final correctness after the edit at each item's first deviating step;
        `changed` records which items' continuations the edit altered at all."""
        new = np.zeros(len(sub), bool)
        changed[:] = False
        by_layer: dict[int, list[int]] = {}
        for i, b in enumerate(sub):
            by_layer.setdefault(b[layer_key], []).append(i)
        for L, idx in by_layer.items():
            vecs, active, mode = [], [], "add"
            for i in idx:
                b = sub[i]
                norm = mult * b["ref_norm"]
                if vec_key == "patch":
                    mode = "set"
                    v = b["patch"] if b["patch"] is not None else np.zeros_like(b["oracle"])
                    vecs.append(v)
                    active.append(b["patch"] is not None)
                    continue
                v = rand if rand is not None else b[vec_key]
                nv = np.linalg.norm(v)
                vecs.append(v / nv * norm if nv > 0 else np.zeros_like(v))
                active.append(bool(nv > 0) and (b["self_active"] if vec_key == "self" else True))
            spec = SteerSpec(
                L,
                mode,
                torch.tensor(np.stack(vecs), dtype=torch.float32),
                1.0,
                where="at",
                at_gen_index=[sub[i]["at"] for i in idx],
                active=active,
            )
            res = run_condition(model, tok, [sub[i]["trace"] for i in idx], spec, max_new, bs, eos)
            for i, r in zip(idx, res):
                new[i] = r.correct
                changed[i] = r.diverged
        return new

    def rates(sub: list[dict], new: np.ndarray) -> dict[str, Any]:
        solved = np.array([b["solved"] for b in sub])
        g = np.array([b["template"] for b in sub])
        orig = solved.copy()  # failures start wrong, solved start right
        rep = lambda i: float(new[i][~solved[i]].mean()) if (~solved[i]).any() else np.nan  # noqa: E731
        dmg = lambda i: float((~new[i][solved[i]]).mean()) if solved[i].any() else np.nan  # noqa: E731
        net = lambda i: float(new[i].mean() - orig[i].mean())  # noqa: E731
        return {
            "n_failures": int((~solved).sum()),
            "n_solved": int(solved.sum()),
            "frac_items_changed": float(changed[: len(sub)].mean()) if len(sub) else float("nan"),
            "repair": cluster_bootstrap(rep, g, nb, seed),
            "damage": cluster_bootstrap(dmg, g, nb, seed),
            "net": cluster_bootstrap(net, g, nb, seed),
        }

    train_b = [b for b in built if b["split"] == "train"]
    test_b = [b for b in built if b["split"] == "test"]
    # strength selection on TRAIN templates, oracle content condition
    sel = []
    for m in cfg["repair"]["strength_multiples"]:
        new = run(train_b, "oracle", m)
        r = rates(train_b, new)
        sel.append(
            {
                "mult": m,
                "net": r["net"]["estimate"],
                "repair": r["repair"]["estimate"],
                "damage": r["damage"]["estimate"],
            }
        )
        print("train select", sel[-1], flush=True)
    best = (
        max(sel, key=lambda r: (round(r["net"], 6), -r["mult"]))["mult"]
        if sel
        else cfg["repair"]["strength_multiples"][0]
    )

    results: dict[str, Any] = {
        "selected_multiple": best,
        "selection_train": sel,
        "versions": library_versions(),
    }
    conds = {
        "oracle content steering": ("oracle", "L"),
        "self-sourced gated steering": ("self", "L"),
        "evaluation steering -> correct": ("eval", "Le"),
        "oracle residual patch (matched correct run)": ("patch", "L"),
    }
    per_item = []
    for name, (key, lk) in conds.items():
        new = run(test_b, key, best, lk)
        results[name] = rates(test_b, new)
        per_item += [
            {"cond": name, "uid": b["trace"]["uid"], "solved": b["solved"], "new_correct": bool(c)}
            for b, c in zip(test_b, new)
        ]
        print(
            name,
            {k: (v["estimate"] if isinstance(v, dict) else v) for k, v in results[name].items()},
            flush=True,
        )
    results["self_sourced_gate_open_rate_failures"] = (
        float(np.mean([b["self_active"] for b in test_b if not b["solved"]])) if test_b else None
    )
    results["self_sourced_gate_open_rate_solved"] = (
        float(np.mean([b["self_active"] for b in test_b if b["solved"]]))
        if any(b["solved"] for b in test_b)
        else None
    )
    # matched-norm random null (per-item norm = mult x the item's reference norm)
    d = acts.shape[-1]
    R = random_directions(d, cfg["repair"]["n_random"], 1.0, seed + 500).numpy()
    nulls = []
    for k in range(len(R)):
        new = run(test_b, "oracle", best, "L", rand=R[k])
        solved = np.array([b["solved"] for b in test_b])
        nulls.append(
            {
                "repair": float(new[~solved].mean()) if (~solved).any() else np.nan,
                "damage": float((~new[solved]).mean()) if solved.any() else np.nan,
                "net": float(new.mean() - solved.mean()),
            }
        )
    nn = np.array([x["net"] for x in nulls])
    results["random null"] = {
        "n": len(nulls),
        "net_mean": float(np.nanmean(nn)),
        "net_q025": float(np.nanquantile(nn, 0.025)),
        "net_q975": float(np.nanquantile(nn, 0.975)),
        "repair_mean": float(np.nanmean([x["repair"] for x in nulls])),
        "damage_mean": float(np.nanmean([x["damage"] for x in nulls])),
        "p_value_net_ge": {
            name: float((1 + (nn >= results[name]["net"]["estimate"]).sum()) / (1 + len(nn)))
            for name in conds
        },
    }
    (out / "repair.json").write_text(json.dumps(results, indent=2, default=float))
    write_jsonl(out / "repair_items.jsonl", per_item)
    from rfd.plotting import condition_bars

    rows = [
        {
            "name": n,
            "net": 100 * results[n]["net"]["estimate"],
            "lo": 100 * results[n]["net"]["lo"],
            "hi": 100 * results[n]["net"]["hi"],
        }
        for n in conds
    ]
    rows.append(
        {
            "name": f"random (mean of {len(nulls)}; bar = 2.5-97.5%)",
            "net": 100 * results["random null"]["net_mean"],
            "lo": 100 * results["random null"]["net_q025"],
            "hi": 100 * results["random null"]["net_q975"],
        }
    )
    condition_bars(
        rows,
        out / "fig_pilot_repair_conditions.png",
        "PILOT: net gain on held-out distractor failures + matched solved set",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["readability", "sample", "repair"])
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--harness", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--n-random", type=int, default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.harness, args.out, args.traces)
    if args.n_random is not None:
        cfg["repair"]["n_random"] = args.n_random
    {
        "readability": lambda: cmd_readability(cfg),
        "sample": lambda: cmd_sample(cfg),
        "repair": lambda: cmd_repair(cfg),
    }[args.cmd]()


if __name__ == "__main__":
    main()
