"""Inputs of the redesigned repair pilot (prereg/R1.md), built on CPU from S2 + the S4 cache.

Train and dev templates only. Output directory (default data/generated/repair_r1_inputs/, a
private Kaggle dataset for the GPU run):
  items.jsonl     one line per item (failure or matched solved trace): ids, split, template,
                  variable, layer, edit window, bound / gold / decoded values, gate, the recorded
                  prompt and output tokens, gold answer, original prediction
  vectors.npy     float32 [n_items, 3, d_model]: UNIT oracle, self-sourced and evaluation
                  directions (a zero row = no edit for that item and condition)
  meta.json       layer / lambda selection, reference norms r_v, counts, fallbacks, sha256s

    python -m repair.build_inputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness.config import REPO_ROOT
from probes import fit
from probes.activations import load_layer, load_rows
from probes.labels import OPS, positions_by_trace, step_rows, traces

SEED = 20261126
VARS = {"content/binary": "distractor_used", "content/4-class": "op"}
LAYERS = list(range(26))
OUT = REPO_ROOT / "data" / "generated" / "repair_r1_inputs"


def _label(s: dict, var: str):
    v = s[VARS[var]]
    if v is None:
        return None
    return int(v) if var == "content/binary" else OPS.index(v)


def main(out: Path = OUT) -> dict:
    steps = step_rows()
    rows = load_rows()
    rix = {(r["trace_id"], r["step"]): r["i"] for r in rows if r["kind"] == "step_first"}
    steps = [s for s in steps if (s["trace_id"], s["step"]) in rix]
    by_trace = defaultdict(list)
    for s in steps:
        by_trace[s["trace_id"]].append(s)
    recs = {r["trace_id"]: r for r in traces()}

    # ---- population: primary failures, content-attributable at s0
    fails = []
    for s in steps:
        if s["population"] == "primary" and s["is_s0"] and s["content_attributable"]:
            var = "content/binary" if s["distractor_used"] else "content/4-class"
            bound = _label(s, var)
            gold = 0 if var == "content/binary" else OPS.index(s["gold_op"])
            fails.append({"role": "failure", "step_row": s, "variable": var, "bound": bound, "gold": gold})
    fails.sort(key=lambda f: (f["step_row"]["trace_id"]))

    # ---- matched solved set (one per failure, same template; fallbacks pre-registered)
    rng = np.random.default_rng(SEED)
    pool = defaultdict(list)
    for tid, r in sorted(recs.items()):
        if r["condition"] == "distractor" and r["correct"]:
            pool[r["template_id"]].append(tid)
    split_pool = defaultdict(list)
    for t, lst in pool.items():
        split_pool[recs[lst[0]]["split"]].extend(lst)
    used = defaultdict(set)
    fallbacks = {"with_replacement": 0, "other_template": 0, "none": 0}
    solved = []

    def cands(tid, var):
        return [s for s in by_trace[tid] if _label(s, var) is not None]

    for f in fails:
        s0 = f["step_row"]
        var, tmpl = f["variable"], s0["template_id"]
        k = sum(1 for s in cands(s0["trace_id"], var) if s["step"] < s0["step"])
        order = [t for t in rng.permutation(pool[tmpl]).tolist() if cands(t, var)]
        fresh = [t for t in order if t not in used[tmpl]]
        if fresh:
            tid = fresh[0]
        elif order:
            tid = order[0]
            fallbacks["with_replacement"] += 1
        else:
            other = [t for t in rng.permutation(split_pool[s0["split"]]).tolist() if cands(t, var)]
            if not other:
                fallbacks["none"] += 1
                continue
            tid = other[0]
            fallbacks["other_template"] += 1
        used[tmpl].add(tid)
        c = cands(tid, var)
        st = c[min(k, len(c) - 1)]
        b = _label(st, var)
        solved.append({"role": "solved", "step_row": st, "variable": var, "bound": b, "gold": b,
                       "matched_failure": s0["trace_id"]})
    items = fails + solved

    # ---- per-variable selection on TRAIN rows (two-stage, prereg/E1.md procedure)
    train = [s for s in steps if s["split"] == "train"]
    tmpl_train = sorted({s["template_id"] for s in train})
    fold_of_t = dict(zip(np.array(tmpl_train), fit.template_folds(np.array(tmpl_train))))
    sel, models = {}, {}
    for var in VARS:
        tr = [s for s in train if _label(s, var) is not None]
        ri = np.array([rix[(s["trace_id"], s["step"])] for s in tr])
        y = np.array([_label(s, var) for s in tr])
        folds = np.array([fold_of_t[s["template_id"]] for s in tr])
        print(f"select {var}: {len(tr)} train rows")
        sel[var] = fit.select(load_layer, LAYERS, y, folds, ri)
        L, lam = sel[var]["layer"], sel[var]["lambda"]
        X = load_layer(L, ri)
        ev = [s for s in train if s["step_correct"] is not None]
        Xe = load_layer(L, np.array([rix[(s["trace_id"], s["step"])] for s in ev]))
        ye = np.array([int(s["step_correct"]) for s in ev])
        fe = np.array([fold_of_t[s["template_id"]] for s in ev])
        classes = np.arange(2 if var == "content/binary" else 4)
        full = fit.fit(X, y, lam, classes)
        mu = full.means
        sel[var]["r_v"] = float(np.linalg.norm(mu[1] - mu[0])) if len(classes) == 2 else float(
            np.mean([np.linalg.norm(mu[a] - mu[b]) for a in classes for b in classes if a != b]))
        sel[var]["n_train"] = int(len(y))
        sel[var]["class_counts"] = {int(c): int((y == c).sum()) for c in classes}
        models[var] = {"all": (full, Xe[ye == 1].mean(0) - Xe[ye == 0].mean(0))}
        for fo in range(fit.N_FOLDS):
            m = folds != fo
            me = fe != fo
            models[var][fo] = (fit.fit(X[m], y[m], lam, classes),
                               Xe[me & (ye == 1)].mean(0) - Xe[me & (ye == 0)].mean(0))

    # ---- edit windows and directions
    pos = positions_by_trace({it["step_row"]["trace_id"] for it in items})
    d = 2304
    vec = np.zeros((len(items), 3, d), np.float32)
    lines = []
    for i, it in enumerate(items):
        s = it["step_row"]
        var = it["variable"]
        L = sel[var]["layer"]
        key = "all" if s["split"] == "dev" else fold_of_t[s["template_id"]]
        probe, evald = models[var][key]
        x = load_layer(L, np.array([rix[(s["trace_id"], s["step"])]]))
        decoded = int(probe.predict(x)[0])
        mu = probe.means
        oracle = mu[it["gold"]] - mu[it["bound"]]
        self_src = mu[decoded] - mu[it["bound"]] if decoded != it["bound"] else np.zeros(d)
        for j, v in enumerate((oracle, self_src, evald)):
            n = float(np.linalg.norm(v))
            if np.isfinite(n) and n > 0:
                vec[i, j] = v / n
        p = pos[s["trace_id"]]["positions"]
        firsts = {q["step"]: q["tok"] for q in p if q["kind"] == "step_first"}
        lasts = {q["step"]: q["tok"] for q in p if q["kind"] == "step_last"}
        start = firsts[s["step"]]
        end = firsts.get(s["step"] + 1, lasts.get(s["step"], start) + 1)
        r = recs[s["trace_id"]]
        lines.append({
            "item": i, "role": it["role"], "trace_id": s["trace_id"], "split": s["split"],
            "template_id": s["template_id"], "variable": var, "layer": L, "r_v": sel[var]["r_v"],
            "step": s["step"], "start": start, "end": end, "bound": it["bound"], "gold": it["gold"],
            "decoded": decoded, "gate_open": decoded != it["bound"], "fit": str(key),
            "matched_failure": it.get("matched_failure"), "s0_type": s["error_type"],
            "prompt_ids": r["prompt_ids"], "output_ids": r["output_ids"],
            "gold_answer": r["gold_answer"], "orig_pred": r["pred_answer"],
            "orig_correct": bool(r["correct"]), "stop_reason": r["stop_reason"],
            "problem_id": r["problem_id"], "prompt": r["prompt"]})
        assert len(r["prompt_ids"]) <= start < end <= len(r["prompt_ids"]) + len(r["output_ids"])
        assert (it["role"] == "failure") != r["correct"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    np.save(out / "vectors.npy", vec)
    meta = {"seed": SEED, "selection": sel, "fallbacks": fallbacks,
            "n_items": len(lines), "n_failures": len(fails), "n_solved": len(solved),
            "failures_by_split": {k: sum(1 for f in fails if f["step_row"]["split"] == k) for k in ("train", "dev")},
            "failures_by_variable": {v: sum(1 for f in fails if f["variable"] == v) for v in VARS},
            "templates": len({f["step_row"]["template_id"] for f in fails}),
            "gate_open_failures": sum(ln["gate_open"] for ln in lines if ln["role"] == "failure"),
            "gate_open_solved": sum(ln["gate_open"] for ln in lines if ln["role"] == "solved"),
            "sha256": {n: hashlib.sha256((out / n).read_bytes()).hexdigest()
                       for n in ("items.jsonl", "vectors.npy")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    m = main(ap.parse_args().out)
    print(json.dumps({k: v for k, v in m.items() if k != "selection"}, indent=2))
    print(json.dumps({v: {k: m["selection"][v][k] for k in ("layer", "lambda", "r_v", "n_train")}
                      for v in m["selection"]}, indent=2))
