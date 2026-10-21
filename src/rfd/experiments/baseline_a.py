"""Baseline A: replicate Sun et al. (2507.12379), probing for arithmetic errors.

Subcommands (each writes under results/baseline_a/):
  pilot    error rate per digit length (choose the task size; pre-registered rule)
  collect  greedy answers + residual stream at the '=' token, every layer
  analyze  probes (CPU): correct-only protocol (brief) and mixed protocol (paper),
           shuffled-label / random-probe controls, layer curves, error recovery
  reprompt probe-guided selective re-prompting vs none / always / oracle

Usage: python -m rfd.experiments.baseline_a {pilot|collect|analyze|reprompt}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from rfd.data.arithmetic import ArithProblem, digits_by_place, sample_problems
from rfd.utils.config import load_yaml
from rfd.utils.io import read_jsonl, write_jsonl

_ANS_RE = re.compile(r"\s*(-?\d+)")


def _cfg(path: str) -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["harness"] = load_yaml(cfg["harness_config"])
    return cfg


def _parse_answer(text: str) -> int | None:
    m = _ANS_RE.match(text)
    return int(m.group(1)) if m else None


def _demos(cfg: dict[str, Any], n_digits: int) -> list[ArithProblem]:
    return sample_problems(cfg["n_demos"], n_digits, cfg["op"], seed=cfg["seed"] + 1000 + n_digits)


def _prefix(tok: Any, p: ArithProblem, demos: list[ArithProblem]) -> list[int]:
    from rfd.harness.prompts import arith_messages, chat_ids

    msgs = arith_messages(p.a, p.b, p.op, [(d.a, d.b) for d in demos])
    return chat_ids(tok, msgs, assistant_prefix=f"<<{p.a}{p.op}{p.b}=")


def _answer_batch(loaded: Any, problems: list[ArithProblem], demos, cfg, capture: bool):
    """Greedy answers; optionally the '=' residual at every layer [bs, L, d] (fp16)."""
    import torch

    from rfd.harness.generate import generate
    from rfd.harness.hooks import ResidCapture, decoder_layers

    tok, model = loaded.tokenizer, loaded.model
    prefixes = [_prefix(tok, p, demos) for p in problems]
    hooks: dict[int, Any] = {}
    if capture:
        n = max(len(s) for s in prefixes)
        pos = torch.full((len(prefixes), 1), n - 1, dtype=torch.long)  # left padding
        hooks = {layer: ResidCapture(pos) for layer in range(len(decoder_layers(model)))}
    outs = generate(
        model, tok, prefixes, cfg["max_new_tokens"], hooks=hooks, stop_at_final_answer=False
    )
    acts = None
    if capture:
        acts = torch.stack([hooks[k].result()[:, 0] for k in sorted(hooks)], 1).half().numpy()
    return prefixes, outs, acts


# --------------------------------------------------------------------------- pilot


def cmd_pilot(cfg: dict[str, Any]) -> None:
    from rfd.harness.model import load
    from rfd.utils.seed import set_seed

    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    loaded = load(cfg["harness"]["model"])
    rates = {}
    for nd in cfg["error_pilot"]["digit_lengths"]:
        demos = _demos(cfg, nd)
        probs = sample_problems(
            cfg["error_pilot"]["n_per_length"],
            nd,
            cfg["op"],
            seed=cfg["seed"] + 2000 + nd,
            exclude={(d.a, d.b) for d in demos},
        )
        n_err = n_unparsed = 0
        bs = cfg["gen_batch_size"]
        for i in range(0, len(probs), bs):
            _, outs, _ = _answer_batch(loaded, probs[i : i + bs], demos, cfg, capture=False)
            for p, o in zip(probs[i : i + bs], outs):
                pred = _parse_answer(o.text)
                n_unparsed += pred is None
                n_err += pred != p.answer
        rates[nd] = {"n": len(probs), "errors": n_err, "unparsed": n_unparsed}
        print(nd, rates[nd], flush=True)
    chosen = None
    for nd, r in rates.items():
        e = r["errors"] / r["n"]
        if e > 0:
            n_needed = max(cfg["n_min"], math.ceil(cfg["target_errors"] / e))
            if n_needed <= cfg["n_max"]:
                chosen = {"n_digits": nd, "error_rate": e, "n_problems": n_needed}
                break
    if chosen is None:  # fall back to the most error-prone length at n_max
        nd = max(rates, key=lambda k: rates[k]["errors"])
        chosen = {
            "n_digits": nd,
            "error_rate": rates[nd]["errors"] / rates[nd]["n"],
            "n_problems": cfg["n_max"],
        }
    (out / "error_pilot.json").write_text(json.dumps({"rates": rates, "chosen": chosen}, indent=2))
    print("chosen", chosen)


# --------------------------------------------------------------------------- collect


def cmd_collect(cfg: dict[str, Any], limit: int | None = None) -> None:
    from rfd.harness.model import library_versions, load
    from rfd.utils.seed import set_seed

    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    chosen = json.loads((out / "error_pilot.json").read_text())["chosen"]
    nd, n = chosen["n_digits"], chosen["n_problems"]
    if limit:
        n = limit
    demos = _demos(cfg, nd)
    probs = sample_problems(
        n, nd, cfg["op"], seed=cfg["seed"] + 3000, exclude={(d.a, d.b) for d in demos}
    )
    loaded = load(cfg["harness"]["model"])
    n_layers = len(loaded.model.model.layers)
    d_model = loaded.model.config.hidden_size
    acts = np.lib.format.open_memmap(
        out / "acts_eq.npy", mode="w+", dtype=np.float16, shape=(n, n_layers, d_model)
    )
    records = []
    bs = cfg["gen_batch_size"]
    for i in range(0, n, bs):
        batch = probs[i : i + bs]
        prefixes, outs, a = _answer_batch(loaded, batch, demos, cfg, capture=True)
        acts[i : i + len(batch)] = a
        for p, pre, o in zip(batch, prefixes, outs):
            pred = _parse_answer(o.text)
            records.append(
                {
                    "uid": p.uid,
                    "a": p.a,
                    "b": p.b,
                    "op": p.op,
                    "answer": p.answer,
                    "pred": pred,
                    "correct": pred == p.answer,
                    "text": o.text,
                    "prompt_len": len(pre),
                    "gen_ids": o.gen_ids,
                }
            )
        if (i // bs) % 10 == 0:
            n_err = sum(not r["correct"] for r in records)
            print(f"{len(records)}/{n} done, errors so far {n_err}", flush=True)
    acts.flush()
    write_jsonl(out / "collect.jsonl", records)
    meta = {
        "n_digits": nd,
        "n_problems": n,
        "demos": [(d.a, d.b) for d in demos],
        "n_layers": n_layers,
        "d_model": d_model,
        "n_errors": sum(not r["correct"] for r in records),
        "n_unparsed": sum(r["pred"] is None for r in records),
        "versions": library_versions(),
    }
    (out / "collect_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "versions"}))


# --------------------------------------------------------------------------- analyze


def _split(n: int, frac_train: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac_train * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


def _fit_eval(X_tr, y_tr, X_ev_list, C, seed, shuffle=False):
    from rfd.probing.probes import fit_probe

    y = np.asarray(y_tr)
    if shuffle:
        y = y[np.random.default_rng(seed).permutation(len(y))]
    if len(np.unique(y)) < 2:
        return [np.full(len(X), -1) for X in X_ev_list]
    p = fit_probe(X_tr, y, "logistic", C)
    return [p.predict(X) if len(X) else np.zeros(0, int) for X in X_ev_list]


def cmd_analyze(cfg: dict[str, Any]) -> None:
    from joblib import Parallel, delayed

    from rfd.stats import cluster_bootstrap, safe_balanced_accuracy

    out = Path(cfg["out_dir"])
    meta = json.loads((out / "collect_meta.json").read_text())
    recs = list(read_jsonl(out / "collect.jsonl"))
    acts = np.load(out / "acts_eq.npy", mmap_mode="r")
    n, n_layers = len(recs), meta["n_layers"]
    C, seed = cfg["probe"]["C"], cfg["seed"]

    answers = np.array([r["answer"] for r in recs])
    preds = np.array([r["pred"] if r["pred"] is not None else -1 for r in recs])
    parsed = preds >= 0
    correct = np.array([r["correct"] for r in recs])
    n_places = len(str(int(answers.max())))
    gold_d = np.array([digits_by_place(v, n_places) for v in answers])  # [n, P]
    model_d = np.array([digits_by_place(v, n_places) if v >= 0 else [-1] * n_places for v in preds])
    err = parsed & ~correct
    # most significant wrong place per error item (Sun: the hundreds digit, i.e. the leading one)
    err_place = np.full(n, -1)
    for i in np.flatnonzero(err):
        wrong = np.flatnonzero(gold_d[i] != model_d[i])
        if len(wrong):
            err_place[i] = wrong.max()
    tr, te = _split(n, cfg["split"]["train"], seed)
    tr_c, te_c = tr[correct[tr]], te[correct[te]]
    # layer selection data: correct TRAIN items only, 80/20 fit/val
    fit_idx, val_idx = (
        tr_c[_split(len(tr_c), 0.8, seed + 1)[0]],
        tr_c[_split(len(tr_c), 0.8, seed + 1)[1]],
    )
    err_idx = np.flatnonzero(err & (err_place >= 0))
    places = list(range(n_places))

    def job(layer: int, place: int) -> dict[str, Any]:
        X = np.asarray(acts[:, layer, :], dtype=np.float32)
        y = gold_d[:, place]
        # protocol P1 (brief): train on correct cases only
        (val_pred,) = _fit_eval(X[fit_idx], y[fit_idx], [X[val_idx]], C, seed)
        te_pred, err_pred = _fit_eval(X[tr_c], y[tr_c], [X[te_c], X[err_idx]], C, seed)
        (err_pred_shuf,) = _fit_eval(
            X[tr_c], y[tr_c], [X[err_idx]], C, seed + 7 + layer, shuffle=True
        )
        # random-weight linear readout (same standardization as the real probe)
        rng = np.random.default_rng(seed + 11 * layer + place)
        mu, sd = X[tr_c].mean(0), X[tr_c].std(0) + 1e-6
        W = rng.standard_normal((X.shape[1], 10))
        err_pred_rand = ((X[err_idx] - mu) / sd @ W).argmax(1)
        # protocol P2 (paper): mixed train split (errors included); output- and GT-digit probes
        tr_p = tr[parsed[tr]]
        te_p = te[parsed[te]]
        (gt_pred_te,) = _fit_eval(X[tr_p], y[tr_p], [X[te_p]], C, seed)
        (out_pred_te,) = _fit_eval(X[tr_p], model_d[tr_p, place], [X[te_p]], C, seed)
        return {
            "layer": layer,
            "place": place,
            "val_bacc": safe_balanced_accuracy(y[val_idx], val_pred),
            "te_correct_bacc": safe_balanced_accuracy(y[te_c], te_pred),
            "err_pred": err_pred.tolist(),
            "err_pred_shuf": err_pred_shuf.tolist(),
            "err_pred_rand": err_pred_rand.tolist(),
            "p2_gt_pred_te": gt_pred_te.tolist(),
            "p2_out_pred_te": out_pred_te.tolist(),
        }

    print(f"fitting {n_layers * len(places)} (layer, place) probe sets...", flush=True)
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(job)(layer, place) for layer in range(n_layers) for place in places
    )
    by = {(r["layer"], r["place"]): r for r in results}

    # error-place view: for each error item, its most significant wrong place
    ep = err_place[err_idx]
    gold_at = gold_d[err_idx, ep]
    model_at = model_d[err_idx, ep]

    def rec_rates(layer_of_place: dict[int, int], key: str) -> np.ndarray:
        """[n_err, 3] indicators: probe == gold digit, == model digit, other."""
        out_ = np.zeros((len(err_idx), 3))
        for j, (p_, g, m) in enumerate(zip(ep, gold_at, model_at)):
            pr = by[(layer_of_place[p_], p_)][key][j]
            out_[j] = [pr == g, pr == m, pr not in (g, m)]
        return out_

    sel_layer = {p: max(range(n_layers), key=lambda L, p=p: by[(L, p)]["val_bacc"]) for p in places}
    curves = {
        "per_place_test_bacc": {
            p: [by[(L, p)]["te_correct_bacc"] for L in range(n_layers)] for p in places
        },
        "per_place_val_bacc": {
            p: [by[(L, p)]["val_bacc"] for L in range(n_layers)] for p in places
        },
        "err_recovery_by_layer": {},
    }
    for key in ("err_pred", "err_pred_shuf", "err_pred_rand"):
        curves["err_recovery_by_layer"][key] = [
            rec_rates({p: L for p in places}, key).mean(0).tolist() for L in range(n_layers)
        ]

    real = rec_rates(sel_layer, "err_pred")
    shuf = rec_rates(sel_layer, "err_pred_shuf")
    rand = rec_rates(sel_layer, "err_pred_rand")
    items = np.arange(len(err_idx))
    nb = cfg["n_boot"]
    primary = {
        "n_error_items": int(len(err_idx)),
        "selected_layer_per_place": sel_layer,
        "recovery_gold": cluster_bootstrap(lambda i: real[i, 0].mean(), items, nb, seed),
        "tracks_model_output": cluster_bootstrap(lambda i: real[i, 1].mean(), items, nb, seed),
        "other_digit": cluster_bootstrap(lambda i: real[i, 2].mean(), items, nb, seed),
        "recovery_gold_shuffled": cluster_bootstrap(lambda i: shuf[i, 0].mean(), items, nb, seed),
        "recovery_gold_random_probe": cluster_bootstrap(
            lambda i: rand[i, 0].mean(), items, nb, seed
        ),
        "recovery_minus_shuffled": cluster_bootstrap(
            lambda i: real[i, 0].mean() - shuf[i, 0].mean(), items, nb, seed
        ),
        # chance if the probe ignored the input: the gold digit's train-set frequency
        "prior_baseline": float(
            np.mean([np.mean(gold_d[tr_c, p_] == g) for p_, g in zip(ep, gold_at)])
        ),
    }
    # decodability on held-out correct cases at the selected layers, vs shuffled control
    dec = {}
    for p in places:
        L = sel_layer[p]
        X = np.asarray(acts[:, L, :], dtype=np.float32)
        y = gold_d[:, p]
        (real_pred,) = _fit_eval(X[tr_c], y[tr_c], [X[te_c]], C, seed)
        (shuf_pred,) = _fit_eval(X[tr_c], y[tr_c], [X[te_c]], C, seed + 99, shuffle=True)
        hit_r, hit_s = (real_pred == y[te_c]), (shuf_pred == y[te_c])
        dec[p] = {
            "layer": L,
            "test_bacc": safe_balanced_accuracy(y[te_c], real_pred),
            "test_bacc_shuffled": safe_balanced_accuracy(y[te_c], shuf_pred),
            "acc_minus_shuffled": cluster_bootstrap(
                lambda i, a=hit_r, b=hit_s: a[i].mean() - b[i].mean(),
                np.arange(len(te_c)),
                nb,
                seed,
            ),
            "n_test_correct": int(len(te_c)),
        }
    primary["decodability_selected_layer"] = dec
    # paper protocol P2 on the mixed held-out split, incl. the error-only subset
    te_p = te[parsed[te]]
    te_err_mask = ~correct[te_p]
    p2 = {}
    for p in places:
        rows = []
        for L in range(n_layers):
            r = by[(L, p)]
            gt, op_ = np.array(r["p2_gt_pred_te"]), np.array(r["p2_out_pred_te"])
            rows.append(
                {
                    "gt_acc": float(np.mean(gt == gold_d[te_p, p])),
                    "out_acc": float(np.mean(op_ == model_d[te_p, p])),
                    "gt_acc_on_errors": float(
                        np.mean(gt[te_err_mask] == gold_d[te_p, p][te_err_mask])
                    )
                    if te_err_mask.any()
                    else float("nan"),
                    "detector_acc": float(np.mean((gt != op_) == te_err_mask)),
                    "detector_bacc": safe_balanced_accuracy(te_err_mask, gt != op_),
                }
            )
        p2[p] = rows
    summary = {
        "meta": {k: v for k, v in meta.items() if k != "versions"},
        "n": n,
        "n_parsed": int(parsed.sum()),
        "n_correct": int(correct.sum()),
        "n_errors_parsed": int(err.sum()),
        "error_place_counts": {int(k): int(v) for k, v in zip(*np.unique(ep, return_counts=True))},
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_test_errors": int(te_err_mask.sum()),
        "primary_P1_correct_only": primary,
        "paper_P2_mixed": p2,
        "curves": curves,
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=2, default=float))
    _plot(summary, out, n_layers, places)
    print(json.dumps({k: summary[k] for k in ("n", "n_errors_parsed", "error_place_counts")}))
    print(json.dumps(primary, indent=2, default=float))


def _plot(summary: dict[str, Any], out: Path, n_layers: int, places: list[int]) -> None:
    from rfd.plotting import baseline_a_figures

    baseline_a_figures(summary, out, n_layers, places)


# --------------------------------------------------------------------------- reprompt


def cmd_reprompt(cfg: dict[str, Any]) -> None:
    from rfd.harness.generate import generate
    from rfd.harness.model import load
    from rfd.probing.probes import fit_probe
    from rfd.stats import cluster_bootstrap, safe_auroc
    from rfd.utils.seed import set_seed

    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    recs = list(read_jsonl(out / "collect.jsonl"))
    acts = np.load(out / "acts_eq.npy", mmap_mode="r")
    n = len(recs)
    seed = cfg["seed"]
    correct = np.array([r["correct"] for r in recs])
    tr, te = _split(n, cfg["split"]["train"], seed)
    fit_idx, val_idx = (tr[s] for s in _split(len(tr), 0.8, seed + 2))
    # detector: binary logistic probe for "this answer is wrong", layer chosen on train val AUROC
    aucs = []
    for L in range(acts.shape[1]):
        X = np.asarray(acts[:, L, :], dtype=np.float32)
        p = fit_probe(X[fit_idx], ~correct[fit_idx], "logistic", cfg["probe"]["C"])
        aucs.append(safe_auroc(~correct[val_idx], p.scores(X[val_idx])[:, 1]))
    L_det = int(np.nanargmax(aucs))
    X = np.asarray(acts[:, L_det, :], dtype=np.float32)
    det = fit_probe(X[fit_idx], ~correct[fit_idx], "logistic", cfg["probe"]["C"])
    # threshold: maximize balanced accuracy on the train val part
    sv = det.scores(X[val_idx])[:, 1]
    grid = np.unique(np.quantile(sv, np.linspace(0.01, 0.99, 99)))
    from rfd.stats import safe_balanced_accuracy

    thr = float(max(grid, key=lambda t: safe_balanced_accuracy(~correct[val_idx], sv >= t)))
    det = fit_probe(X[tr], ~correct[tr], "logistic", cfg["probe"]["C"])
    flag = det.scores(X[te])[:, 1] >= thr

    loaded = load(cfg["harness"]["model"])
    tok = loaded.tokenizer
    msg = cfg["reprompt"]["message"]
    meta = json.loads((out / "collect_meta.json").read_text())
    demos = [ArithProblem("demo", a, b, cfg["op"]) for a, b in meta["demos"]]
    prefixes = []
    for i in te:
        r = recs[i]
        text = r["text"]
        cut = text.find(">>")
        first = text[: cut + 2] if cut >= 0 else text
        base = _prefix(tok, ArithProblem(r["uid"], r["a"], r["b"], r["op"]), demos)
        follow = f"{first} {msg} <<{r['a']}{r['op']}{r['b']}="
        prefixes.append(base + tok(follow, add_special_tokens=False)["input_ids"])
    new_pred = []
    bs = cfg["gen_batch_size"]
    for i in range(0, len(prefixes), bs):
        outs = generate(
            loaded.model,
            tok,
            prefixes[i : i + bs],
            cfg["max_new_tokens"],
            stop_at_final_answer=False,
        )
        # o.text is only the newly generated continuation after "<<a+b="
        new_pred += [_parse_answer(o.text) for o in outs]
    answers = np.array([recs[i]["answer"] for i in te])
    orig_ok = correct[te]
    re_ok = np.array([p == a for p, a in zip(new_pred, answers)])
    err_te = ~orig_ok
    conds = {
        "no_intervention": orig_ok,
        "always_reprompt": re_ok,
        "probe_selective": np.where(flag, re_ok, orig_ok),
        "oracle_selective": np.where(err_te, re_ok, orig_ok),
    }
    items = np.arange(len(te))
    nb = cfg["n_boot"]
    res = {
        "detector_layer": L_det,
        "detector_val_auroc_by_layer": aucs,
        "threshold": thr,
        "n_test": int(len(te)),
        "n_test_errors": int(err_te.sum()),
        "n_flagged": int(flag.sum()),
        "true_positives": int((flag & err_te).sum()),
        "false_positives": int((flag & ~err_te).sum()),
        "detector_test_auroc": safe_auroc(err_te, det.scores(X[te])[:, 1]),
        "tp_correction": float(re_ok[flag & err_te].mean()) if (flag & err_te).any() else None,
        "fp_preservation": float(re_ok[flag & ~err_te].mean()) if (flag & ~err_te).any() else None,
        "accuracy": {
            k: cluster_bootstrap(lambda i, v=v: v[i].mean(), items, nb, seed)
            for k, v in conds.items()
        },
        "selective_minus_none": cluster_bootstrap(
            lambda i: conds["probe_selective"][i].mean() - orig_ok[i].mean(), items, nb, seed
        ),
        "selective_minus_always": cluster_bootstrap(
            lambda i: conds["probe_selective"][i].mean() - re_ok[i].mean(), items, nb, seed
        ),
    }
    write_jsonl(
        out / "reprompt.jsonl",
        [
            {
                "uid": recs[i]["uid"],
                "flag": bool(f),
                "orig_ok": bool(o),
                "re_pred": p,
                "re_ok": bool(k),
            }
            for i, f, o, p, k in zip(te, flag, orig_ok, new_pred, re_ok)
        ],
    )
    (out / "reprompt.json").write_text(json.dumps(res, indent=2, default=float))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "detector_val_auroc_by_layer"},
            indent=2,
            default=float,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "collect", "analyze", "reprompt"])
    ap.add_argument("--config", default="configs/baseline_a.yaml")
    ap.add_argument("--limit", type=int, default=None, help="debug: collect only N problems")
    args = ap.parse_args()
    cfg = _cfg(args.config)
    if args.cmd == "pilot":
        cmd_pilot(cfg)
    elif args.cmd == "collect":
        cmd_collect(cfg, args.limit)
    elif args.cmd == "analyze":
        cmd_analyze(cfg)
    else:
        cmd_reprompt(cfg)


if __name__ == "__main__":
    main()
