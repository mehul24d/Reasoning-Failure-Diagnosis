"""Verified TransformerLens pass over finished traces: equivalence check + residual cache.

One forward pass per trace (prompt + output, batch 1, no padding: the configuration checked in
S1b) does three things (prereg/S2.md amendments A2 and A4):

1. Teacher-forced verification. The logits at every position that predicts an output token
   are computed and their argmax compared with the token HF generated. The logits are computed
   only at those positions, in chunks, from TL's own `ln_final` and `unembed` and Gemma's output
   soft-cap (the same operations as `HookedTransformer.forward`, which with return_type=None
   stops before the unembedding). So the full [T x 256k] float32 logits are never held, and the
   float32 TL copy (~11.9 GiB) fits on one 15 GB T4 without splitting it over two GPUs.
   At the HF token and the HF runner-up the TL logit is compared with the logit HF produced
   while generating (recorded by harness.batched.Top2Recorder): the max logit difference.
2. Residual cache. The residual stream h^(0) (`blocks.0.hook_resid_pre`) and h^(l)
   (`blocks.{l}.hook_resid_post`, l = 0..L-1) at the trace's pre-registered positions
   (cache/positions.py) are stored as float16 shards. Only storage is float16; every
   computation is float32. A cast that overflows stops the run.
3. A float16 storage check. For the first traces of the equivalence sample the float32
   residuals are also kept (`ref_fp32/`) and the float16 error is reported.

Each trace's verification record also holds its index entries (shard, row, kind, step, token),
and is appended in one line only after its shard is on disk, so a crash can never leave a trace
marked done without its residuals. A line cut by a crash is ignored and the trace redone.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from harness.config import experiment_config
from harness.pipeline import (PARTIAL_EXIT, Run, WorkerSpec, _Deadline, equivalence_sample,
                              read_jsonl, sha256, shard)


def hook_names(n_layers: int) -> list[str]:
    return ["blocks.0.hook_resid_pre"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]


@torch.no_grad()
def tl_pass(tl, ids: list[int], n_prompt: int, rows: list[int], hf_top2: dict | None = None,
            chunk: int = 128) -> tuple[np.ndarray | None, dict]:
    """Residuals [len(rows), L+1, d] (float32, CPU) and the verification of one trace."""
    L = tl.cfg.n_layers
    names = hook_names(L)
    last = names[-1]
    T = len(ids)
    saved: dict[str, torch.Tensor] = {}

    def fn(act, hook):
        if rows:
            saved[hook.name] = act[0, torch.tensor(rows, device=act.device)].float().cpu()
        if hook.name == last:
            saved["_final"] = act[0, n_prompt - 1: T - 1].clone()

    tl.run_with_hooks(torch.tensor([ids], device=tl.cfg.device), return_type=None,
                      fwd_hooks=[(h, fn) for h in names])
    final = saved.pop("_final")
    out = torch.tensor(ids[n_prompt:], device=final.device)
    cap = float(getattr(tl.cfg, "output_logits_soft_cap", 0.0) or 0.0)
    preds, t1, t2, at_out, at_id2 = [], [], [], [], []
    id2 = torch.tensor(hf_top2["id2"], device=final.device) if hf_top2 else None
    for i in range(0, final.shape[0], chunk):
        x = tl.ln_final(final[i: i + chunk][None])
        lg = tl.unembed(x)[0]
        if cap > 0.0:
            lg = cap * torch.tanh(lg / cap)
        lg = lg.float()
        preds.append(lg.argmax(-1))
        v, _ = lg.topk(2, dim=-1)
        t1.append(v[:, 0]), t2.append(v[:, 1])
        sl = torch.arange(lg.shape[0], device=lg.device)
        at_out.append(lg[sl, out[i: i + chunk]])
        if id2 is not None:
            at_id2.append(lg[sl, id2[i: i + chunk]])
    pred = torch.cat(preds).cpu()
    tl_margin = (torch.cat(t1) - torch.cat(t2)).cpu()
    outc = out.cpu()
    match = pred == outc
    rec = {"n_tokens": int(outc.numel()), "n_match": int(match.sum()),
           "first_mismatch": None if bool(match.all()) else int((~match).nonzero()[0])}
    hf_margin = None
    if hf_top2:
        l1 = torch.tensor(hf_top2["l1"])
        l2 = torch.tensor(hf_top2["l2"])
        d1 = (torch.cat(at_out).cpu() - l1).abs()
        d2 = (torch.cat(at_id2).cpu() - l2).abs()
        rec["max_abs_logit_diff"] = float(torch.maximum(d1, d2).max()) if d1.numel() else 0.0
        rec["sum_abs_logit_diff"] = float(d1.sum() + d2.sum())
        rec["n_logit_cmp"] = int(d1.numel() + d2.numel())
        rec["hf_top1_is_output"] = int((torch.tensor(hf_top2["id1"]) == outc).sum())
        hf_margin = l1 - l2
    mm = (~match).nonzero().flatten().tolist()[:20]
    rec["mismatches"] = [{"j": j, "hf_token": int(outc[j]), "tl_token": int(pred[j]),
                          "hf_margin": None if hf_margin is None else float(hf_margin[j]),
                          "tl_margin": float(tl_margin[j])} for j in mm]
    resid = None
    if rows:
        resid = torch.stack([saved[h] for h in names], 1).numpy()
    return resid, rec


class ShardWriter:
    """float16 shards `resid_w{worker}_{k:05d}.npy`; verification records written after them."""

    def __init__(self, store: Path, worker: int, rows_per_shard: int, verify_out: Path):
        self.store = store
        store.mkdir(parents=True, exist_ok=True)
        self.worker = worker
        self.rows_per_shard = rows_per_shard
        self.verify_out = verify_out
        ks = [int(p.stem.rsplit("_", 1)[1]) for p in store.glob(f"resid_w{worker}_*.npy")]
        self.k = max(ks, default=-1) + 1      # never overwrite a shard, even an orphan
        self.buf: list[np.ndarray] = []
        self.n = 0
        self.pending: list[dict] = []

    @property
    def name(self) -> str:
        return f"resid_w{self.worker}_{self.k:05d}.npy"

    def add(self, resid16: np.ndarray | None, rec: dict) -> None:
        if resid16 is not None and len(resid16):
            for e in rec["rows"]:
                e["shard"], e["row"] = self.name, e["row"] + self.n
            self.buf.append(resid16)
            self.n += len(resid16)
        self.pending.append(rec)
        if self.n >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if self.buf:
            tmp = self.store / (self.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, np.concatenate(self.buf))
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.store / self.name)
            self.k += 1
        with self.verify_out.open("a") as f:
            for r in self.pending:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.buf, self.n, self.pending = [], 0, []


def verify_worker(spec: WorkerSpec, loader) -> int:
    run: Run = spec.run_obj()
    cfg = experiment_config()
    vcfg, ccfg = cfg["verification"], cfg["cache"]
    sample = equivalence_sample(run)
    scored = {r["trace_id"]: r for r in read_jsonl(run.scored)}
    if spec.phase == "sample":
        items = sample
    else:
        s = set(sample)
        items = [t for t in scored if t not in s]
    items = shard(items, spec.worker, spec.n_workers)
    vout = run.part("verify", spec.worker)
    done = {r["trace_id"] for r in read_jsonl(vout)}
    todo = [t for t in items if t not in done]
    if not todo:
        return 0
    positions = {p["trace_id"]: p for p in read_jsonl(run.positions)}
    top2 = {r["trace_id"]: r for r in read_jsonl(run.top2)}
    fp16_ids = set(sample[: ccfg["fp16_check_traces"]])
    ref_dir = run.store / "ref_fp32"
    tl = loader.tl(run.base, n_devices=spec.tl_n_devices)
    writer = ShardWriter(run.store, spec.worker, ccfg["rows_per_shard"], vout)
    dl = _Deadline(spec.deadline)
    for tid in todo:
        if dl.near():
            writer.flush()
            return PARTIAL_EXIT
        t0 = time.time()
        r = scored[tid]
        pos = positions[tid]
        uniq = sorted({p["tok"] for p in pos["positions"]})
        row_of = {t: i for i, t in enumerate(uniq)}
        ids = r["prompt_ids"] + r["output_ids"]
        resid, rec = tl_pass(tl, ids, len(r["prompt_ids"]), uniq, top2.get(tid), vcfg["logit_chunk"])
        resid16 = None
        if resid is not None:
            resid16 = resid.astype(np.float16)
            if not np.isfinite(resid16).all():
                raise RuntimeError(f"float16 overflow in the residuals of {tid} "
                                   f"(max |x| = {np.abs(resid).max():.1f}): storage dtype needs a decision")
            if tid in fp16_ids:
                err = np.abs(resid16.astype(np.float32) - resid)
                rel = np.linalg.norm(resid16.astype(np.float32) - resid, axis=-1) / \
                    np.maximum(np.linalg.norm(resid, axis=-1), 1e-12)
                rec["fp16"] = {"max_abs_err": float(err.max()), "max_rel_err": float(rel.max()),
                               "max_abs_value": float(np.abs(resid).max())}
                ref_dir.mkdir(parents=True, exist_ok=True)
                fname = f"ref_w{spec.worker}_{len(list(ref_dir.glob('*.npy'))):05d}.npy"
                np.save(ref_dir / fname, resid)
                rec["fp16"]["ref_file"] = fname
        rec.update({"trace_id": tid, "in_eq_sample": spec.phase == "sample",
                    "template_id": r["template_id"], "worker": spec.worker,
                    "n_seq_tokens": len(ids), "seconds": round(time.time() - t0, 4),
                    "rows": [{"kind": p["kind"], "step": p["step"], "tok": p["tok"],
                              "token_id": p["token_id"], "char": p["char"], "row": row_of[p["tok"]],
                              "shard": None} for p in pos["positions"]]})
        writer.add(resid16, rec)
        dl.tick(time.time() - t0)
    writer.flush()
    return 0


def _verify_records(run: Run) -> dict[str, dict]:
    recs = {}
    for f in sorted((run.base / "parts").glob("verify.part*.jsonl")):
        for r in read_jsonl(f):
            recs.setdefault(r["trace_id"], r)
    return recs


def summarise(recs: list[dict]) -> dict:
    tok = sum(r["n_tokens"] for r in recs)
    match = sum(r["n_match"] for r in recs)
    with_hf = [r for r in recs if "max_abs_logit_diff" in r]
    mism = [dict(m, trace_id=r["trace_id"]) for r in recs for m in r["mismatches"]]
    fp = [r["fp16"] for r in recs if "fp16" in r]
    diffs = sorted(r["max_abs_logit_diff"] for r in with_hf)
    return {
        "traces": len(recs), "templates": len({r["template_id"] for r in recs}), "tokens": tok,
        "reproduction": match / max(tok, 1),
        "traces_with_any_mismatch": sum(r["first_mismatch"] is not None for r in recs),
        "max_abs_logit_diff": max(diffs, default=None),
        "p99_trace_max_abs_logit_diff": diffs[int(0.99 * (len(diffs) - 1))] if diffs else None,
        "mean_abs_logit_diff": (sum(r["sum_abs_logit_diff"] for r in with_hf)
                                / max(sum(r["n_logit_cmp"] for r in with_hf), 1)) if with_hf else None,
        "hf_top1_is_output": sum(r.get("hf_top1_is_output", 0) for r in with_hf),
        "mismatches": mism[:50],
        "fp16_check": None if not fp else {
            "traces": len(fp), "max_abs_err": max(x["max_abs_err"] for x in fp),
            "max_rel_err": max(x["max_rel_err"] for x in fp),
            "max_abs_value": max(x["max_abs_value"] for x in fp)},
        "seconds_per_trace": sum(r["seconds"] for r in recs) / max(len(recs), 1),
        "seconds_per_seq_token": sum(r["seconds"] for r in recs) / max(sum(r["n_seq_tokens"] for r in recs), 1),
    }


def gate_report(run: Run) -> dict:
    """The pre-registered equivalence check on the stratified sample (amendment A2)."""
    thr = experiment_config()["equivalence_checks"]["teacher_forced_reproduction_min"]
    recs = _verify_records(run)
    sample = equivalence_sample(run)
    got = [recs[t] for t in sample if t in recs]
    s = summarise(got)
    rep = {"check": "teacher-forced TL logits reproduce the HF batched greedy tokens",
           "threshold": f">= {thr}", "sample_size": len(sample), "verified": len(got),
           **s, "pass": len(got) == len(sample) and s["reproduction"] >= thr}
    run.reports.mkdir(parents=True, exist_ok=True)
    out = run.reports / f"{run.prefix}_equivalence_batched_{run.model}.json"
    out.write_text(json.dumps(rep, indent=2))
    return {"pass": rep["pass"], "reproduction": s["reproduction"], "tokens": s["tokens"],
            "traces": s["traces"], "max_abs_logit_diff": s["max_abs_logit_diff"],
            "traces_with_any_mismatch": s["traces_with_any_mismatch"],
            "fp16_check": s["fp16_check"], "report": str(out)}


def merge_store(run: Run) -> dict:
    """Index + manifest of the activation store, a check that every entry points into a shard,
    and the verification summary over ALL traces (secondary to the pre-registered sample)."""
    from harness.config import model_config

    recs = _verify_records(run)
    scored = [r["trace_id"] for r in read_jsonl(run.scored)]
    missing = [t for t in scored if t not in recs]
    if missing:
        raise RuntimeError(f"{len(missing)} traces have no verification record")
    shapes, used = {}, defaultdict(int)
    for p in sorted(run.store.glob("resid_w*.npy")):
        shapes[p.name] = np.load(p, mmap_mode="r").shape
    index = []
    for t in scored:
        for e in recs[t]["rows"]:
            if e["shard"] not in shapes or e["row"] >= shapes[e["shard"]][0]:
                raise RuntimeError(f"bad index entry for {t}: {e}")
            used[e["shard"]] += 1
            index.append({"trace_id": t, **e})
    idx_file = run.store / "index.jsonl"
    with idx_file.open("w") as f:
        for e in index:
            f.write(json.dumps(e) + "\n")
    mc = model_config(run.model)
    total = sum((run.store / s).stat().st_size for s in shapes)
    shape = next(iter(shapes.values()), (0, mc.n_layers + 1, mc.d_model))
    manifest = {"model": run.model, "hooks": hook_names(shape[1] - 1), "d_model": shape[2],
                "storage_dtype": "float16", "compute_dtype": mc.dtype,
                "positions": experiment_config()["probing"]["positions"],
                "shards": {s: {"rows": shapes[s][0], "bytes": (run.store / s).stat().st_size,
                               "index_entries": used.get(s, 0)} for s in shapes},
                "orphan_shards": [s for s in shapes if not used.get(s)],
                "n_rows": sum(v[0] for v in shapes.values()), "n_index_entries": len(index),
                "traces": len(scored), "bytes": total, "index_sha256": sha256(idx_file)}
    (run.store / "manifest.json").write_text(json.dumps(manifest, indent=1))
    allsum = summarise(list(recs.values()))
    run.reports.mkdir(parents=True, exist_ok=True)
    (run.reports / f"{run.prefix}_verification_all_{run.model}.json").write_text(
        json.dumps(allsum, indent=2))
    return {"store_bytes": total, "store_gb": round(total / 1e9, 3), "rows": manifest["n_rows"],
            "orphan_shards": len(manifest["orphan_shards"]),
            "all_traces_reproduction": allsum["reproduction"],
            "all_traces_max_abs_logit_diff": allsum["max_abs_logit_diff"]}
