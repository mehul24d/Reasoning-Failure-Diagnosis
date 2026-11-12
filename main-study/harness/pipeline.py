"""Minor-project data pipeline on Gemma-2-2B-IT (prereg/S2.md amendments of 2026-11-13).

Stages, in order (each is resumable, writes `status/<stage>.json`, and is skipped once complete):

  problems   CPU  GSM-Symbolic `main` problems for the study templates (needs the committed split)
  generate   GPU  greedy traces, batched HF generate, one worker per GPU      (amendments A2, A3)
  score      CPU  answer, correctness, split, s0 (parser v4), read-out positions
  verify     GPU  pre-registered TL equivalence check on the stratified sample: the GATE (A2)
  cache      GPU  verified TL pass + residual cache on all remaining traces                  (A4)
  sampled    GPU  T = 0.7, k = 5, test-split failures only                                    (A5)
  report     CPU  provenance, cap fraction, disk use

GPU stages run their workers as subprocesses (one per visible GPU, `CUDA_VISIBLE_DEVICES`), so
every stage releases its GPU memory when it ends. A worker stops starting new batches when the
session deadline is near, writes what it has, and exits with code 3; the stage is then
"partial" and the next session resumes it. No trace is ever regenerated.

Usage (repository root = main-study/):
    python -m harness.pipeline run --model gemma-2-2b-it --stages generate,score --deadline <unix>
    python -m harness.pipeline status --model gemma-2-2b-it
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import REPO_ROOT, experiment_config, model_config

PARTIAL_EXIT = 3
STAGES = ["problems", "generate", "score", "verify", "cache", "sampled", "report"]
GPU_STAGES = {"generate", "verify", "cache", "sampled"}
SPLIT_FILE = REPO_ROOT / "data" / "splits" / "gsm_symbolic_main_seed0.json"
S2_PREREG = REPO_ROOT / "prereg" / "S2.md"


# ----------------------------------------------------------------------------------- layout
@dataclass
class Run:
    """Where one run's files live. `study` = the pre-registered S2-S4 files; `smoke` = make estimate."""
    model: str
    base: Path                 # traces, parts, status
    store: Path                # activation shards
    reports: Path
    prefix: str = "s2"
    problems: Path | None = None
    split: Path | None = None

    @classmethod
    def study(cls, model: str, root: Path | None = None) -> "Run":
        root = Path(root or os.environ.get("BTP_DATA_ROOT", REPO_ROOT))
        inst = experiment_config()["data"]["instances_per_template"]
        return cls(model, root / "data" / "generated" / model, root / "cache" / "store" / model,
                   root / "reports" / "m1", "s2", root / "data" / f"s2_gsms_main_i{inst}.jsonl",
                   SPLIT_FILE)

    @classmethod
    def smoke(cls, model: str, root: Path | None = None) -> "Run":
        root = Path(root or os.environ.get("BTP_DATA_ROOT", REPO_ROOT))
        b = root / "data" / "smoke" / model
        return cls(model, b, b / "store", root / "reports" / "m1", "smoke", b / "problems.jsonl",
                   b / "split.json")

    # files
    def p(self, name: str) -> Path:
        self.base.mkdir(parents=True, exist_ok=True)
        return self.base / name

    @property
    def traces(self) -> Path: return self.p(f"{self.prefix}_traces.jsonl")
    @property
    def top2(self) -> Path: return self.p(f"{self.prefix}_top2.jsonl")
    @property
    def scored(self) -> Path: return self.p(f"{self.prefix}_traces_scored.jsonl")
    @property
    def positions(self) -> Path: return self.p(f"{self.prefix}_positions.jsonl")
    @property
    def sampled(self) -> Path: return self.p(f"{self.prefix}_sampled.jsonl")
    @property
    def eq_sample(self) -> Path: return self.p(f"{self.prefix}_eq_sample.json")

    def part(self, kind: str, w: int) -> Path:
        d = self.base / "parts"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{kind}.part{w}.jsonl"

    def status_file(self, stage: str) -> Path:
        d = self.base / "status"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stage}.json"

    def status(self, stage: str) -> dict:
        f = self.status_file(stage)
        return json.loads(f.read_text()) if f.exists() else {"status": "pending"}

    def set_status(self, stage: str, status: str, **info) -> dict:
        rec = {"stage": stage, "status": status, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        self.status_file(stage).write_text(json.dumps(rec, indent=2, default=str))
        return rec


# ----------------------------------------------------------------------------------- helpers
def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:   # a line cut by a crash: ignore, it is redone
                    pass
    return out


def write_jsonl(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trace_id(problem_id: str, prompt: str, sample: int) -> str:
    return f"{problem_id}|{prompt}|{sample}"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def n_gpus() -> int:
    """Visible GPUs, without creating a CUDA context in this (orchestrating) process."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return len([v for v in vis.split(",") if v.strip() != ""])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.SubprocessError):
        return 0


def s2_is_binding() -> bool:
    """prereg/S2.md carries a date on its 'Date committed (binding)' line."""
    for line in S2_PREREG.read_text().splitlines():
        if line.lstrip().startswith("* **Date committed (binding):**"):
            val = line.split(":**", 1)[1]
            return "_to be filled_" not in val and any(ch.isdigit() for ch in val)
    return False


@contextlib.contextmanager
def load_lock(directory: Path):
    """Serialise model loading across workers: each load briefly holds a float32 copy in CPU
    RAM (~10.5 GB HF + ~12 GB TL for Gemma-2-2B); two at once would exceed Kaggle's ~30 GB."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".load.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------------- work lists
def problems_of(run: Run) -> list[dict]:
    return read_jsonl(run.problems)


def greedy_items(run: Run) -> list[tuple]:
    from .prompts import all_prompt_specs

    return [(p["problem_id"], s.tag, 0) for p in problems_of(run) for s in all_prompt_specs()]


def sampled_items(run: Run) -> list[tuple]:
    """(problem, prompt, sample) for every greedy failure on a TEST template (amendment A5)."""
    k = experiment_config()["generation"]["sampled_subset"]["k"]
    fails = [r for r in read_jsonl(run.scored) if r.get("split") == "test" and not r["correct"]]
    fails.sort(key=lambda r: (r["problem_id"], r["prompt"]))
    return [(r["problem_id"], r["prompt"], s) for r in fails for s in range(k)]


def shard(items: list, w: int, n: int) -> list:
    return [it for i, it in enumerate(items) if i % n == w]


# ----------------------------------------------------------------------------------- workers
@dataclass
class WorkerSpec:
    kind: str                  # greedy | sampled | verify
    model: str
    run: dict                  # Run fields as strings
    worker: int = 0
    n_workers: int = 1
    deadline: float | None = None
    batch_size: int = 16
    engine: str = "hf_batched"     # or "tl" (fallback, amendment A2)
    phase: str = "all"            # verify: sample | rest
    tl_n_devices: int = 1
    limit: int | None = None
    extra: dict = field(default_factory=dict)

    def run_obj(self) -> Run:
        r = dict(self.run)
        for k in ("base", "store", "reports", "problems", "split"):
            if r.get(k) is not None:
                r[k] = Path(r[k])
        return Run(**r)


def _run_fields(run: Run) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(run).items()}


class RealLoader:
    """Loads the real models inside a worker process (one GPU, `CUDA_VISIBLE_DEVICES`)."""

    def __init__(self, model: str):
        self.cfg = model_config(model)

    def tokenizer(self):
        from .models import load_tokenizer

        return load_tokenizer(self.cfg)

    def hf(self, lock_dir: Path):
        import torch

        from .models import load_hf

        with load_lock(lock_dir):
            m = load_hf(self.cfg, device="cuda" if torch.cuda.is_available() else None)
        return m

    def tl(self, lock_dir: Path, n_devices: int = 1):
        import gc

        import torch

        from .models import load_hf, load_tl

        with load_lock(lock_dir):
            tok = self.tokenizer()
            hf = load_hf(self.cfg, device="cpu")
            tl = load_tl(self.cfg, hf_model=hf, tokenizer=tok,
                         device="cuda" if torch.cuda.is_available() else "cpu", n_devices=n_devices)
            del hf
            gc.collect()
        return tl


class _Deadline:
    def __init__(self, deadline: float | None):
        self.deadline = deadline
        self.worst = 0.0

    def tick(self, seconds: float) -> None:
        self.worst = max(self.worst, seconds)

    def near(self) -> bool:
        return self.deadline is not None and time.time() + 1.5 * self.worst + 60 > self.deadline


def generation_worker(spec: WorkerSpec, loader) -> int:
    """Greedy or sampled generation for this worker's share of the items. Returns an exit code."""
    import torch

    from .batched import Timer, batch_seed, eos_ids, generate_batch, tl_generate_one
    from .generate import greedy_generation_config, is_pure_greedy, sampled_generation_config
    from .prompts import PromptSpec, encode_chat

    run = spec.run_obj()
    exp = experiment_config()["generation"]
    max_new = exp["greedy"]["max_new_tokens"]
    tok = loader.tokenizer()
    greedy = spec.kind == "greedy"
    items = greedy_items(run) if greedy else sampled_items(run)
    items = shard(items, spec.worker, spec.n_workers)[: spec.limit]
    out = run.part("greedy" if greedy else "sampled", spec.worker)
    top_out = run.part("top2", spec.worker) if greedy else None
    done = {tuple(r["key"]) for r in read_jsonl(out)}
    todo = [it for it in items if tuple(it) not in done]
    if not todo:
        return 0
    probs = {p["problem_id"]: p for p in problems_of(run)}
    if greedy:
        gen_cfg = greedy_generation_config(tok, max_new)
        assert is_pure_greedy(gen_cfg)
    else:
        gen_cfg = sampled_generation_config(tok, max_new, exp["sampled_subset"]["temperature"])
    enc = {}
    for pid, tag, _ in todo:
        if (pid, tag) not in enc:
            style, para = tag.rsplit("_p", 1)
            enc[(pid, tag)] = encode_chat(tok, probs[pid]["question"], PromptSpec(style, int(para)))
    if greedy:   # longest prompts first: an out-of-memory shows up in the first batch
        todo.sort(key=lambda it: -len(enc[(it[0], it[1])]))
    model = loader.tl(run.base) if spec.engine == "tl" else loader.hf(run.base)
    dl = _Deadline(spec.deadline)
    bs = spec.batch_size if spec.engine != "tl" else 1
    pos = 0
    with out.open("a") as f, (top_out.open("a") if top_out else contextlib.nullcontext()) as ft:
        while pos < len(todo):
            if dl.near():
                return PARTIAL_EXIT
            batch = todo[pos: pos + bs]
            t = Timer()
            try:
                if spec.engine == "tl":
                    res = [tl_generate_one(model, tok, enc[(b[0], b[1])], max_new, eos_ids(tok))
                           for b in batch]
                    seed = None
                else:
                    seed = None if greedy else batch_seed(batch)
                    res = generate_batch(model, tok, [enc[(b[0], b[1])] for b in batch], gen_cfg,
                                         seed=seed, record_top2=greedy)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[worker {spec.worker}] out of memory: batch size -> {bs}", flush=True)
                continue
            secs = t.s
            dl.tick(secs)
            for (pid, tag, s), r in zip(batch, res):
                p = probs[pid]
                rec = {
                    "trace_id": trace_id(pid, tag, s), "key": [pid, tag, s],
                    "problem_id": pid, "template_id": p.get("template_id"),
                    "variant": p.get("variant"), "gold_answer": p.get("answer"),
                    "model": spec.model, "prompt": tag, "mode": spec.kind, "sample": s,
                    "seed": seed, "engine": spec.engine,
                    "prompt_ids": enc[(pid, tag)], "output_ids": r.output_ids,
                    "text": tok.decode(r.output_ids, skip_special_tokens=True),
                    "stop_reason": r.stop_reason, "hit_max_tokens": r.stop_reason == "max_new_tokens",
                    "n_new_tokens": len(r.output_ids), "worker": spec.worker,
                    "batch_size": len(batch), "batch_seconds": round(secs, 3),
                    "batch_keys_crc": batch_seed(batch),
                }
                f.write(json.dumps(rec) + "\n")
                if ft is not None and r.top2 is not None:
                    ft.write(json.dumps({"trace_id": rec["trace_id"], **r.top2}) + "\n")
            f.flush()
            if ft is not None:
                ft.flush()
            pos += len(batch)
    return 0


# ----------------------------------------------------------------------------------- orchestration
def launch(spec: WorkerSpec, n: int, loader=None) -> list[int]:
    """Run n workers. With a loader (tests) they run in this process, one after the other;
    otherwise each is a subprocess pinned to one GPU, so its memory is freed when it exits."""
    codes = []
    if loader is not None:
        for w in range(n):
            s = WorkerSpec(**{**asdict(spec), "worker": w, "n_workers": n})
            codes.append(_dispatch(s, loader))
        return codes
    procs = []
    for w in range(n):
        s = WorkerSpec(**{**asdict(spec), "worker": w, "n_workers": n})
        f = spec.run_obj().p(f"worker_{spec.kind}_{w}.json")
        f.write_text(json.dumps(asdict(s), default=str))
        env = dict(os.environ)
        if spec.tl_n_devices == 1 and n_gpus() > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(w) if "CUDA_VISIBLE_DEVICES" not in os.environ \
                else os.environ["CUDA_VISIBLE_DEVICES"].split(",")[w]
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        procs.append(subprocess.Popen([sys.executable, "-m", "harness.pipeline", "worker", str(f)],
                                      cwd=str(REPO_ROOT), env=env))
    for p in procs:
        codes.append(p.wait())
    return codes


def _dispatch(spec: WorkerSpec, loader) -> int:
    if spec.kind in ("greedy", "sampled"):
        return generation_worker(spec, loader)
    if spec.kind == "verify":
        from cache.verify_cache import verify_worker

        return verify_worker(spec, loader)
    raise ValueError(spec.kind)


def worker_main(spec_file: str) -> None:
    d = json.loads(Path(spec_file).read_text())
    spec = WorkerSpec(**d)
    code = _dispatch(spec, RealLoader(spec.model))
    raise SystemExit(code)


def _codes_status(codes: list[int]) -> str:
    if all(c == 0 for c in codes):
        return "complete"
    if all(c in (0, PARTIAL_EXIT) for c in codes):
        return "partial"
    return "failed"


def merge_parts(run: Run, kind: str, items: list[tuple], n_workers_max: int = 16) -> list[dict]:
    """All parts of `kind`, de-duplicated by key, in work-list order."""
    by_key = {}
    for w in range(n_workers_max):
        f = run.base / "parts" / f"{kind}.part{w}.jsonl"
        for r in read_jsonl(f):
            k = tuple(r["key"]) if "key" in r else r["trace_id"]
            by_key.setdefault(k, r)
    return [by_key[tuple(it)] for it in items if tuple(it) in by_key]


# ----------------------------------------------------------------------------------- stages
def stage_problems(run: Run, **_) -> dict:
    from .datasets import gsm_symbolic, write_jsonl as wj

    if run.prefix == "s2":
        if not run.split.exists():
            raise RuntimeError(f"{run.split} is missing: it is generated once and committed with the "
                               "binding prereg/S2.md (harness.splits), before any S2 data exist.")
    sp = json.loads(run.split.read_text())
    templates = sorted(sp["train"] + sp["dev"] + sp["test"])
    inst = experiment_config()["data"]["instances_per_template"]
    ids = [int(t.split("-")[-1]) for t in templates]
    rows = gsm_symbolic("main", templates=ids, instances=inst)
    wj(rows, run.problems)
    n_t = len({r["template_id"] for r in rows})
    if n_t != len(templates):
        raise RuntimeError(f"problems cover {n_t} templates, split has {len(templates)}")
    return {"problems": len(rows), "templates": n_t, "instances_per_template": inst,
            "sha256": sha256(run.problems)}


def stage_generate(run: Run, kind="greedy", loader=None, n_workers=None, deadline=None,
                   batch_size=None, engine=None, limit=None, **_) -> dict:
    exp = experiment_config()["generation"]
    n = n_workers or max(1, n_gpus())
    engine = engine or engine_of(run)
    spec = WorkerSpec(kind=kind, model=run.model, run=_run_fields(run), deadline=deadline,
                      batch_size=batch_size or exp["batched"]["batch_size"],
                      engine=engine, limit=limit)
    t0 = time.time()
    codes = launch(spec, n, loader)
    status = _codes_status(codes)
    info = {"worker_codes": codes, "n_workers": n, "seconds": round(time.time() - t0, 1)}
    if status != "complete":
        return {"_status": status, **info}
    items = greedy_items(run) if kind == "greedy" else sampled_items(run)
    if limit is not None:
        items = [it for w in range(n) for it in shard(items, w, n)[:limit]]
    recs = merge_parts(run, "greedy" if kind == "greedy" else "sampled", items)
    out = run.traces if kind == "greedy" else run.sampled
    write_jsonl(recs, out)
    info.update(lines=len(recs), sha256=sha256(out), file=str(out), commit=git_commit())
    if kind == "greedy":
        tops = {}
        for w in range(16):
            for r in read_jsonl(run.base / "parts" / f"top2.part{w}.jsonl"):
                tops.setdefault(r["trace_id"], r)
        write_jsonl([tops[r["trace_id"]] for r in recs if r["trace_id"] in tops], run.top2)
        info["top2_lines"] = len(tops)
    info.update(generation_summary(recs))
    return info


def engine_of(run: Run) -> str:
    """`tl` once the fallback has been invoked for this run (amendment A2), else the config."""
    if (run.base / "ENGINE_TL").exists():
        return "tl"
    return experiment_config()["generation"]["batched"]["engine"]


def fallback_to_tl(run: Run) -> dict:
    """The pre-registered fallback after a FAILED equivalence check (amendment A2): the HF
    batched traces and everything derived from them are moved (not deleted) to
    `hf_batched_rejected/`, and generation restarts with the TL engine. Only allowed when the
    verify stage has failed."""
    if run.status("verify").get("status") != "failed":
        raise RuntimeError("fallback is only for a failed equivalence check (verify stage)")
    dest = run.base / "hf_batched_rejected"
    dest.mkdir(exist_ok=True)
    moved = []
    for f in list(run.base.iterdir()):
        if f.name in ("hf_batched_rejected", "tl_fallback") or f.name.startswith("worker_"):
            continue
        if f.name in (run.problems.name if run.problems else "",):
            continue
        f.replace(dest / f.name)
        moved.append(f.name)
    if run.store.exists() and any(run.store.iterdir()):
        run.store.replace(dest / "store")
        moved.append("store")
    (run.base / "ENGINE_TL").write_text(time.strftime("%Y-%m-%d %H:%M:%S") +
                                        " equivalence check failed; TL generation (prereg S2 A2)\n")
    # the problems stage stays complete: its file is not derived from generation
    if (dest / "status" / "problems.json").exists():
        run.status_file("problems").write_text((dest / "status" / "problems.json").read_text())
    return {"moved": moved, "to": str(dest), "engine": "tl"}


def generation_summary(recs: list[dict]) -> dict:
    n = max(len(recs), 1)
    stops = Counter(r["stop_reason"] for r in recs)
    toks = [r["n_new_tokens"] for r in recs]
    return {"traces": len(recs), "stop_reasons": dict(stops),
            "hit_max_tokens_fraction": stops.get("max_new_tokens", 0) / n,
            "mean_new_tokens": sum(toks) / n, "max_new_tokens_seen": max(toks, default=0)}


def stage_score(run: Run, loader=None, **_) -> dict:
    """Answer, correctness, split, first error step (parser v4) and read-out positions."""
    from cache.positions import trace_positions
    from parsing.gold import gold_program
    from parsing.steps import parse_trace

    from .score import score_record

    tok = (loader or RealLoader(run.model)).tokenizer()
    sp = json.loads(run.split.read_text())
    split_of = {t: k for k in ("train", "dev", "test") for t in sp[k]}
    probs = {p["problem_id"]: p for p in problems_of(run)}
    scored, positions = [], []
    c = Counter()
    s0_test_templates = set()
    for r in read_jsonl(run.traces):
        r = score_record(r)
        r["split"] = split_of.get(r["template_id"])
        p = probs[r["problem_id"]]
        prog, sl = gold_program(p["question"], p["solution"], p["answer"], p["source"])
        r["gold_ok"] = prog.ok
        r["s0"] = r["s0_strict"] = None
        if prog.ok:
            pt = parse_trace(r["text"], r["prompt"].split("_")[0], sl, prog)
            r["s0"], r["s0_strict"] = pt.first_error, pt.first_error_strict
            r["s0_type"] = None if pt.first_error is None else pt.steps[pt.first_error].error
        pos = trace_positions(tok, r)
        positions.append(pos)
        c["traces"] += 1
        c["correct"] += r["correct"]
        c["no_answer"] += not r["answer_extracted"]
        c["hit_max_tokens"] += r["stop_reason"] == "max_new_tokens"
        c["char_map_failed"] += not pos["char_map_ok"]
        c["positions"] += len(pos["positions"])
        c["unique_rows"] += len({q["tok"] for q in pos["positions"]})
        if r["split"] == "test" and not r["correct"]:
            c["test_failures"] += 1
            if r["s0"] is not None:
                c["test_failures_with_s0"] += 1
                s0_test_templates.add(r["template_id"])
        scored.append(r)
    write_jsonl(scored, run.scored)
    write_jsonl(positions, run.positions)
    n = max(c["traces"], 1)
    return {"traces": c["traces"], "accuracy": c["correct"] / n, "no_answer": c["no_answer"] / n,
            "hit_max_tokens_fraction": c["hit_max_tokens"] / n,
            "char_map_failed": c["char_map_failed"], "positions": c["positions"],
            "unique_rows": c["unique_rows"], "rows_per_trace": c["unique_rows"] / n,
            "test_failures": c["test_failures"], "test_failures_with_s0": c["test_failures_with_s0"],
            "test_templates_with_s0_failure": len(s0_test_templates),
            "design_effect": design_effect(scored),
            "sha256_scored": sha256(run.scored), "sha256_positions": sha256(run.positions)}


def design_effect(scored: list[dict]) -> dict:
    """Kish design effect of the test failure set, clustered by template (amendment A6).

    DEFF = 1 + (m - 1) * rho, m = mean test failures (with s0) per test template and rho the
    one-way ANOVA intraclass correlation of the failure indicator across ALL non-test traces
    (train + dev templates, so the test split informs nothing but its own count).
    """
    groups = defaultdict(list)
    for r in scored:
        if r.get("split") in ("train", "dev"):
            groups[r["template_id"]].append(0 if r["correct"] else 1)
    k = len(groups)
    N = sum(len(g) for g in groups.values())
    rho = None
    if k > 1 and N > k:
        grand = sum(sum(g) for g in groups.values()) / N
        msb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values()) / (k - 1)
        msw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups.values()) / (N - k)
        n0 = (N - sum(len(g) ** 2 for g in groups.values()) / N) / (k - 1)
        rho = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)) if (msb + (n0 - 1) * msw) > 0 else 0.0
    test = defaultdict(int)
    for r in scored:
        if r.get("split") == "test" and not r["correct"] and r.get("s0") is not None:
            test[r["template_id"]] += 1
    m = sum(test.values()) / max(len(test), 1)
    deff = None if rho is None else 1 + (m - 1) * rho
    power = experiment_config()["data"]["power"]
    need = None if deff is None else max(power["min_test_failures"],
                                         math.ceil(power["n_after_holm"] * deff))
    return {"rho_failure_icc_train_dev": rho, "mean_test_failures_per_template": m,
            "deff": deff, "required_test_failures_with_s0": need,
            "observed_test_failures_with_s0": sum(test.values()),
            "top_up_needed": None if need is None else sum(test.values()) < need}


def equivalence_sample(run: Run) -> list[str]:
    """>= 200 greedy traces, stratified by template: the same number from every template,
    drawn with the pre-registered seed. Written once; later calls read the file."""
    if run.eq_sample.exists():
        return json.loads(run.eq_sample.read_text())["trace_ids"]
    v = experiment_config()["verification"]
    by_t = defaultdict(list)
    for r in read_jsonl(run.scored):
        by_t[r["template_id"]].append(r["trace_id"])
    per = max(v["per_template"], math.ceil(v["min_sample"] / max(len(by_t), 1)))
    rng = random.Random(v["seed"])
    ids = []
    for t in sorted(by_t):
        pool = sorted(by_t[t])
        ids += rng.sample(pool, min(per, len(pool)))
    rng.shuffle(ids)
    run.eq_sample.write_text(json.dumps({"seed": v["seed"], "per_template": per,
                                         "n": len(ids), "trace_ids": ids}, indent=1))
    return ids


def stage_verify(run: Run, phase="sample", loader=None, n_workers=None, deadline=None,
                 tl_n_devices=None, **_) -> dict:
    from cache.verify_cache import gate_report, merge_store

    exp = experiment_config()
    tl_n = tl_n_devices or exp["verification"]["tl_n_devices"]
    n = n_workers or (1 if tl_n > 1 else max(1, n_gpus()))
    equivalence_sample(run)
    spec = WorkerSpec(kind="verify", model=run.model, run=_run_fields(run), deadline=deadline,
                      phase=phase, tl_n_devices=tl_n)
    t0 = time.time()
    codes = launch(spec, n, loader)
    status = _codes_status(codes)
    info = {"worker_codes": codes, "n_workers": n, "seconds": round(time.time() - t0, 1),
            "tl_n_devices": tl_n}
    if status != "complete":
        return {"_status": status, **info}
    if phase == "sample":
        rep = gate_report(run)
        info.update(rep)
        if not rep["pass"]:
            return {"_status": "failed", **info,
                    "next": "equivalence check FAILED: fall back to TL generation "
                            "(engine: tl in configs/experiment.yaml), as pre-registered (A2)"}
        return info
    info.update(merge_store(run))
    return info


def stage_report(run: Run, **_) -> dict:
    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    files = {}
    for f in (run.problems, run.traces, run.top2, run.scored, run.positions, run.sampled):
        if f and f.exists():
            files[f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f),
                             "lines": sum(1 for _ in f.open())}
    rep = {"model": run.model, "commit": git_commit(), "files": files,
           "disk_bytes": {"traces_dir": du(run.base), "activation_store": du(run.store)},
           "stages": {s: run.status(s) for s in STAGES if s != "report"}}
    rep["disk_bytes"]["total"] = sum(rep["disk_bytes"].values())
    run.reports.mkdir(parents=True, exist_ok=True)
    out = run.reports / f"{run.prefix}_run_summary_{run.model}.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    return {"summary": str(out), "disk_gb": round(rep["disk_bytes"]["total"] / 1e9, 3)}


STAGE_FUNCS = {
    "problems": stage_problems,
    "generate": lambda run, **kw: stage_generate(run, "greedy", **kw),
    "score": stage_score,
    "verify": lambda run, **kw: stage_verify(run, "sample", **kw),
    "cache": lambda run, **kw: stage_verify(run, "rest", **kw),
    "sampled": lambda run, **kw: stage_generate(run, "sampled", **kw),
    "report": stage_report,
}


def run_stages(run: Run, stages: list[str], deadline: float | None = None, loader=None,
               require_binding: bool = True, **kw) -> list[dict]:
    """Run the stages in order; stop at the first that is not complete."""
    if require_binding and run.prefix == "s2" and not s2_is_binding():
        raise RuntimeError("prereg/S2.md is still a DRAFT: no study data may be produced until the "
                           "group signs it off and the binding date is filled in.")
    log = []
    for st in stages:
        cur = run.status(st)
        if cur["status"] == "complete":
            log.append({"stage": st, "status": "complete (skipped)"})
            continue
        if deadline is not None and time.time() > deadline - 300:
            log.append({"stage": st, "status": "not started (deadline)"})
            break
        print(f"=== stage {st}", flush=True)
        t0 = time.time()
        try:
            info = STAGE_FUNCS[st](run, deadline=deadline, loader=loader, **kw)
        except Exception as e:  # noqa: BLE001  (recorded, then re-raised for the caller)
            run.set_status(st, "failed", error=repr(e))
            raise
        status = info.pop("_status", "complete")
        info["stage_seconds"] = round(time.time() - t0, 1)
        rec = run.set_status(st, status, **info)
        log.append(rec)
        print(json.dumps(rec, indent=1, default=str)[:4000], flush=True)
        if status != "complete":
            break
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="gemma-2-2b-it")
    r.add_argument("--stages", default=",".join(STAGES))
    r.add_argument("--deadline", type=float, default=None, help="unix time; stop before it")
    r.add_argument("--hours", type=float, default=None, help="alternative to --deadline")
    r.add_argument("--engine", default=None, choices=[None, "hf_batched", "tl"])
    r.add_argument("--batch-size", type=int, default=None)
    s = sub.add_parser("status")
    s.add_argument("--model", default="gemma-2-2b-it")
    w = sub.add_parser("worker")
    w.add_argument("spec")
    fb = sub.add_parser("fallback", help="after a FAILED equivalence check: switch to TL generation")
    fb.add_argument("--model", default="gemma-2-2b-it")
    a = ap.parse_args()
    if a.cmd == "worker":
        worker_main(a.spec)
    run = Run.study(a.model)
    if a.cmd == "fallback":
        print(json.dumps(fallback_to_tl(run), indent=1))
        return
    if a.cmd == "status":
        print(json.dumps({st: run.status(st)["status"] for st in STAGES}, indent=1))
        return
    deadline = a.deadline or (time.time() + a.hours * 3600 if a.hours else None)
    log = run_stages(run, [x for x in a.stages.split(",") if x], deadline=deadline,
                     engine=a.engine, batch_size=a.batch_size)
    bad = [x for x in log if x["status"] not in ("complete", "complete (skipped)")]
    raise SystemExit(0 if not bad else (PARTIAL_EXIT if all(x["status"] in ("partial", "not started (deadline)")
                                                           for x in bad) else 1))


if __name__ == "__main__":
    main()
