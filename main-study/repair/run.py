"""GPU worker of the redesigned repair pilot (prereg/R1.md). One process per T4.

    python -m repair.run --inputs DIR --out DIR --stage STAGE --worker W --n-workers N \
        [--mults 0.25,0.5] [--conds oracle,self,eval,rand0-9] [--roles failure,solved] [--splits dev]
        [--deadline UNIX_TIME]

Stages:
  stability  k = 5 samples at T = 0.7 per failure (S3 settings and seeding), scored
  edit       edited greedy continuations for every (item, condition, multiple) asked for

Conditions: oracle | self | eval | randJ (J = 0..99; shared Gaussian directions, seed below).
Every non-zero edit has norm m x r_v (r_v of the item's variable). A zero edit (the oracle on a
solved item, the self-sourced edit when its gate is closed) is not run: the outcome is the
recorded trace, by construction.

Resumable: results are appended to <out>/<stage>_w<W>.jsonl and finished keys are skipped.
Timing per worker goes to <out>/<stage>_w<W>_timing.json (GPU seconds by phase).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

RANDOM_SEED = 20261126
N_RANDOM = 100
D_MODEL = 2304


def random_units(n: int = N_RANDOM, d: int = D_MODEL, seed: int = RANDOM_SEED) -> np.ndarray:
    g = np.random.default_rng(seed)
    v = g.standard_normal((n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def read_items(inputs: Path) -> tuple[list[dict], np.ndarray]:
    items = [json.loads(x) for x in (inputs / "items.jsonl").read_text().splitlines() if x.strip()]
    return items, np.load(inputs / "vectors.npy")


def parse_conds(s: str) -> list[str]:
    out = []
    for c in s.split(","):
        if c.startswith("rand") and "-" in c:
            a, b = c[4:].split("-")
            out += [f"rand{j}" for j in range(int(a), int(b) + 1)]
        elif c:
            out.append(c)
    return out


def unit_for(cond: str, i: int, vecs: np.ndarray, rnd: np.ndarray) -> np.ndarray:
    if cond.startswith("rand"):
        return rnd[int(cond[4:])]
    return vecs[i, {"oracle": 0, "self": 1, "eval": 2}[cond]]


def _done(path: Path) -> set:
    """Keys already in ANY worker file of this tag (e.g. calibration rows reused for the main
    run are copied into main_w9.jsonl before the main stage starts)."""
    tag = path.name.rsplit("_w", 1)[0]
    out = set()
    for p in path.parent.glob(f"{tag}_w*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.add((d["item"], d["cond"], d["m"]) if "cond" in d else (d["item"], d["sample"]))
    return out


def _load(device: str):
    from harness.config import model_config
    from harness.models import load_hf, load_tokenizer

    cfg = model_config("gemma-2-2b-it")
    return load_hf(cfg, device=device), load_tokenizer(cfg)


def stage_edit(a, items, vecs, model, tok) -> dict:
    from harness.generate import greedy_generation_config

    from .engine import Job, outcome, regenerate, scan

    rnd = random_units(d=vecs.shape[2])
    mults = [float(x) for x in a.mults.split(",")]
    conds = parse_conds(a.conds)
    out = Path(a.out) / f"{a.tag}_w{a.worker}.jsonl"
    done = _done(out)
    t = {"scan_s": 0.0, "regen_s": 0.0, "scan_rows": 0, "regen_rows": 0, "regen_new_tokens": 0,
         "trivial_rows": 0, "reference_reproduces_recorded": []}
    mine = [it for it in items if it["item"] % a.n_workers == a.worker
            and it["role"] in a.roles.split(",") and it["split"] in a.splits.split(",")]
    gen_cfg = lambda n: greedy_generation_config(tok, n)              # noqa: E731
    dev = next(model.parameters()).device
    with out.open("a") as f:
        for it in mine:
            if a.deadline and time.time() > a.deadline:
                print("deadline reached; stopping cleanly", flush=True)
                break
            i = it["item"]
            todo, trivial = [], []
            for m in mults:
                for c in conds:
                    if (i, c, m) in done:
                        continue
                    u = unit_for(c, i, vecs, rnd)
                    (todo if float(np.abs(u).sum()) > 0 else trivial).append((c, m, u))
            for c, m, _ in trivial:                                   # zero edit: recorded trace
                f.write(json.dumps({"item": i, "cond": c, "m": m, "edited": False, "diverged": False,
                                    "pred": it["orig_pred"], "correct": it["orig_correct"],
                                    "answer_changed": False, "broken": it["orig_pred"] is None,
                                    "stop_reason": it["stop_reason"]}) + "\n")
                t["trivial_rows"] += 1
            if not todo:
                f.flush()
                continue
            seq = it["prompt_ids"] + it["output_ids"]
            V = torch.tensor(np.stack([u * m * it["r_v"] for _, m, u in todo]), dtype=torch.float32,
                             device=dev)
            t0 = time.time()
            divs, info = scan(model, seq, it["start"], it["end"], it["layer"], V, batch=a.batch)
            torch.cuda.synchronize(dev) if dev.type == "cuda" else None
            t["scan_s"] += time.time() - t0
            t["scan_rows"] += len(todo)
            t["reference_reproduces_recorded"].append(info["reference_reproduces_recorded"])
            P = len(it["prompt_ids"])
            jobs, idx = [], []
            for k, ((c, m, _), dv) in enumerate(zip(todo, divs)):
                if dv.pos is None:
                    f.write(json.dumps({"item": i, "cond": c, "m": m, "edited": True, "diverged": False,
                                        "pred": it["orig_pred"], "correct": it["orig_correct"],
                                        "answer_changed": False, "broken": it["orig_pred"] is None,
                                        "stop_reason": it["stop_reason"],
                                        "ref_repro": info["reference_reproduces_recorded"]}) + "\n")
                    continue
                kk = dv.pos + 1 - P
                jobs.append(Job(it["prompt_ids"], it["output_ids"][:kk] + [dv.token], V[k],
                                it["start"], it["end"]))
                idx.append((c, m, dv.pos))
            for b0 in range(0, len(jobs), a.gen_batch):
                t0 = time.time()
                res = regenerate(model, tok, jobs[b0:b0 + a.gen_batch], it["layer"], gen_cfg, 512)
                torch.cuda.synchronize(dev) if dev.type == "cuda" else None
                t["regen_s"] += time.time() - t0
                t["regen_rows"] += len(res)
                for (c, m, p), r, j in zip(idx[b0:b0 + a.gen_batch], res, jobs[b0:b0 + a.gen_batch]):
                    t["regen_new_tokens"] += len(r["output_ids"]) - len(j.prefix_out) + 1
                    o = outcome(r["text"], r["stop_reason"], it["gold_answer"], it["orig_pred"])
                    f.write(json.dumps({"item": i, "cond": c, "m": m, "edited": True, "diverged": True,
                                        "div_pos": p, **o, "stop_reason": r["stop_reason"],
                                        "n_out": len(r["output_ids"]), "text": r["text"]}) + "\n")
            f.flush()
    t["reference_reproduces_recorded_min"] = min(t["reference_reproduces_recorded"], default=None)
    t["reference_reproduces_recorded_mean"] = float(np.mean(t["reference_reproduces_recorded"])) \
        if t["reference_reproduces_recorded"] else None
    del t["reference_reproduces_recorded"]
    return t


def stage_stability(a, items, vecs, model, tok) -> dict:
    from harness.batched import batch_seed, generate_batch
    from harness.config import experiment_config
    from harness.generate import sampled_generation_config
    from parsing.answers import extract_final_answer, is_correct

    ss = experiment_config()["generation"]["sampled_subset"]
    k = ss["k"]
    cfg = sampled_generation_config(tok, 512, ss["temperature"])
    out = Path(a.out) / f"stability_w{a.worker}.jsonl"
    done = _done(out)
    keys = [(it["item"], s) for it in items if it["role"] == "failure"
            and it["item"] % a.n_workers == a.worker for s in range(k)]
    keys = [x for x in keys if x not in done]
    by = {it["item"]: it for it in items}
    t = {"seconds": 0.0, "rows": 0, "new_tokens": 0}
    with out.open("a") as f:
        for b0 in range(0, len(keys), a.gen_batch):
            if a.deadline and time.time() > a.deadline:
                break
            chunk = keys[b0:b0 + a.gen_batch]
            ids = [(by[i]["trace_id"], s) for i, s in chunk]
            seed = batch_seed(ids)
            t0 = time.time()
            res = generate_batch(model, tok, [by[i]["prompt_ids"] for i, _ in chunk], cfg, seed=seed)
            t["seconds"] += time.time() - t0
            for (i, s), r in zip(chunk, res):
                text = tok.decode(r.output_ids, skip_special_tokens=True)
                pred = extract_final_answer(text)
                t["rows"] += 1
                t["new_tokens"] += len(r.output_ids)
                f.write(json.dumps({"item": i, "sample": s, "seed": seed, "stop_reason": r.stop_reason,
                                    "pred": None if pred is None else str(pred),
                                    "correct": bool(is_correct(pred, by[i]["gold_answer"])),
                                    "text": text}) + "\n")
            f.flush()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=["stability", "edit"], required=True)
    ap.add_argument("--tag", default="edit", help="result file prefix for the edit stage")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--mults", default="1")
    ap.add_argument("--conds", default="oracle,self,eval")
    ap.add_argument("--roles", default="failure,solved")
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, vecs = read_items(a.inputs)
    dev = a.device or (f"cuda:{a.worker}" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, tok = _load(dev)
    load_s = time.time() - t0
    fn = stage_stability if a.stage == "stability" else stage_edit
    t = fn(a, items, vecs, model, tok)
    t.update(load_s=load_s, wall_s=time.time() - t0, device=dev, finished=time.time())
    name = "stability" if a.stage == "stability" else a.tag
    p = a.out / f"{name}_w{a.worker}_timing.json"
    hist = json.loads(p.read_text()) if p.exists() else []
    hist.append(t)
    p.write_text(json.dumps(hist, indent=2))
    print(json.dumps(t), flush=True)


if __name__ == "__main__":
    main()
