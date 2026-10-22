"""GSM-Symbolic CoT traces (main + NoOp distractor variants) with step labels and activations.

Stages (each writes under <out_dir>/, skips itself once complete, and resumes
mid-way after a crash; see rfd.utils.resume):
  generate  greedy marked-CoT traces            -> gen_chunks/, traces.jsonl
  label     step parsing vs the gold program    -> steps.jsonl, parse_report.json, parse_sample.md
  extract   teacher-forced residuals, all layers -> shards/acts_{question,last,steps}/ + acts_index.json
            (read them with rfd.utils.resume.open_array)

Usage: python -m rfd.experiments.collect_traces {generate|label|extract} [--harness CFG] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from rfd.data.distractor import make_variant
from rfd.data.gsm_symbolic import load_gsm_symbolic, template_splits
from rfd.generation.answer_extraction import extract_predicted_answer, is_correct
from rfd.parsing.steps import char_to_token, label_steps
from rfd.utils.config import load_run_config, record_config
from rfd.utils.io import read_jsonl
from rfd.utils.resume import (
    ShardedArray,
    StageMarker,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    check_manifest,
    file_fingerprint,
)


def load_cfg(path: str, harness: str | None, out: str | None) -> dict[str, Any]:
    return load_run_config(path, harness, out)


def build_items(cfg: dict[str, Any], limit: int | None = None) -> tuple[list[dict], dict]:
    """The fixed trace set: K instances per usable template, each in every variant."""
    probs, report = load_gsm_symbolic(cfg["harness"]["cache_dir"])
    by_t: dict[int, list] = {}
    for p in probs:
        by_t.setdefault(p.template_id, []).append(p)
    splits = template_splits(sorted(by_t), cfg["split_fractions"], cfg["seed"])
    split_of = {t: s for s, ts in splits.items() for t in ts}
    rng = random.Random(cfg["seed"])
    items = []
    for t in sorted(by_t):
        chosen = sorted(
            rng.sample(by_t[t], min(cfg["instances_per_template"], len(by_t[t]))),
            key=lambda p: p.instance,
        )
        for p in chosen:
            forbidden = p.question_numbers + [v for s in p.gold_steps for v in (s.a, s.b, s.result)]
            for variant in cfg["variants"]:
                q, d, clause = p.question, None, None
                if variant == "noop":
                    v = make_variant(
                        t, p.question, forbidden, seed=cfg["seed"] * 1000 + t * 100 + p.instance
                    )
                    if v is None:
                        continue
                    q, d, clause = v.question, v.d, v.clause
                items.append(
                    {
                        "uid": f"{p.uid}-{variant}",
                        "pair_uid": p.uid,
                        "template_id": t,
                        "instance": p.instance,
                        "variant": variant,
                        "split": split_of[t],
                        "question": q,
                        "distractor": d,
                        "clause": clause,
                        "gold_answer": p.gold_answer,
                        "gold_steps": [[s.op, s.a, s.b, s.result] for s in p.gold_steps],
                        "question_numbers": p.question_numbers,
                    }
                )
    if limit:  # debug: the first `limit` traces of each split (whole templates first)
        items = [
            i
            for sp in ("train", "dev", "test")
            for i in [x for x in items if x["split"] == sp][:limit]
        ]
    return items, {"data_report": report, "splits": splits}


# --------------------------------------------------------------------------- generate


def cmd_generate(cfg: dict[str, Any], limit: int | None, force: bool = False) -> None:
    """Chunk k of the length-sorted trace list is written atomically to
    gen_chunks/<k>.jsonl, so a restart skips finished chunks and re-forms
    exactly the same batches; traces.jsonl is assembled in item order at the end."""
    from rfd.harness.generate import generate
    from rfd.harness.model import library_versions, load
    from rfd.harness.prompts import chat_ids, cot_messages
    from rfd.utils.seed import set_seed

    out = Path(cfg["out_dir"])
    stage = StageMarker(out, "generate")
    if stage.skip(force):
        return
    set_seed(cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    check_manifest(out, cfg["harness"]["model"])
    items, meta = build_items(cfg, limit)
    atomic_write_json(out / "splits.json", meta)
    loaded = load(cfg["harness"]["model"])
    tok = loaded.tokenizer
    gen_cfg = cfg["harness"]["generation"]
    for it in items:
        it["prompt_ids"] = chat_ids(tok, cot_messages(it["question"]))
    order = sorted(range(len(items)), key=lambda k: len(items[k]["prompt_ids"]))  # less padding
    cs = cfg["chunk_size"]
    chunks = [order[c0 : c0 + cs] for c0 in range(0, len(order), cs)]
    cdir = out / "gen_chunks"
    plan = {"chunk_size": cs, "uids": [[items[k]["uid"] for k in ch] for ch in chunks]}
    plan_path = cdir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError(f"{plan_path} was made for a different trace set; use --force")
    atomic_write_json(plan_path, plan)
    n_done = sum((cdir / f"{k:05d}.jsonl").exists() for k in range(len(chunks)))
    print(f"{len(items)} traces in {len(chunks)} chunks, {n_done} chunks done", flush=True)
    bs = gen_cfg["batch_size"]
    for k, ch in enumerate(chunks):
        path = cdir / f"{k:05d}.jsonl"
        if path.exists():
            continue
        chunk = [items[i] for i in ch]
        recs = []
        for b0 in range(0, len(chunk), bs):
            batch = chunk[b0 : b0 + bs]
            outs = generate(
                loaded.model, tok, [i["prompt_ids"] for i in batch], gen_cfg["max_new_tokens"]
            )
            for it, o in zip(batch, outs):
                pred = extract_predicted_answer(o.text)
                recs.append(
                    {
                        **it,
                        "gen_ids": o.gen_ids,
                        "text": o.text,
                        "pred": pred,
                        "correct": is_correct(pred, str(it["gold_answer"])),
                        "has_final_marker": "Final Answer:" in o.text,
                        "token_logprobs": o.token_logprobs,
                        "token_entropies": o.token_entropies,
                    }
                )
        atomic_write_jsonl(path, recs)
        print(
            f"chunk {k + 1}/{len(chunks)}; chunk acc {np.mean([r['correct'] for r in recs]):.2f}",
            flush=True,
        )
    by_uid = {r["uid"]: r for k in range(len(chunks)) for r in read_jsonl(cdir / f"{k:05d}.jsonl")}
    atomic_write_jsonl(out / "traces.jsonl", [by_uid[it["uid"]] for it in items])
    atomic_write_json(
        out / "generate_meta.json",
        {
            "model": cfg["harness"]["model"],
            "generation": gen_cfg,
            "versions": library_versions(),
        },
    )
    stage.mark({"n_traces": len(items)})


# --------------------------------------------------------------------------- label


def _token_texts(tok: Any, ids: list[int]) -> list[str]:
    """Decoded text of each generated token, such that ''.join == decode(ids)
    (cumulative decode handles multi-byte characters split across tokens)."""
    texts, prev = [], ""
    for k in range(1, len(ids) + 1):
        cur = tok.decode(ids[:k], skip_special_tokens=True)
        texts.append(cur[len(prev) :])
        prev = cur
    return texts


def cmd_label(cfg: dict[str, Any], force: bool = False) -> None:
    from transformers import AutoTokenizer

    from rfd.parsing.equations import BinStep

    out = Path(cfg["out_dir"])
    stage = StageMarker(out, "label", file_fingerprint(out / "traces.jsonl"))
    if stage.skip(force):
        return
    tok = AutoTokenizer.from_pretrained(cfg["harness"]["model"]["model_name"])
    traces = list(read_jsonl(out / "traces.jsonl"))
    step_rows = []
    for tr in traces:
        gold = [BinStep(op, a, b, r, r) for op, a, b, r in tr["gold_steps"]]
        labs = label_steps(tr["text"], gold, tr["question_numbers"], tr["distractor"])
        toks = _token_texts(tok, tr["gen_ids"])
        for j, lab in enumerate(labs):
            row = {k: v for k, v in lab.to_dict().items() if k != "leaves"}
            row.update(
                {
                    "uid": tr["uid"],
                    "pair_uid": tr["pair_uid"],
                    "template_id": tr["template_id"],
                    "variant": tr["variant"],
                    "split": tr["split"],
                    "trace_correct": tr["correct"],
                    "step_pos": j,
                    "n_steps": len(labs),
                    # first generated token of the step, relative to the generated text
                    "gen_token_index": char_to_token(toks, lab.char_start),
                    "step_text": tr["text"][lab.char_start : lab.char_end].strip(),
                }
            )
            step_rows.append(row)
    atomic_write_jsonl(out / "steps.jsonl", step_rows)
    parsed = [s for s in step_rows if s["error_type"] is not None]
    report = {
        "n_traces": len(traces),
        "accuracy_by_variant": {
            v: float(np.mean([t["correct"] for t in traces if t["variant"] == v]))
            for v in sorted({t["variant"] for t in traces})
        },
        "final_marker_rate": float(np.mean([t["has_final_marker"] for t in traces])),
        "traces_with_steps": float(
            np.mean([any(s["uid"] == t["uid"] for s in step_rows) for t in traces])
        )
        if len(traces) < 5000
        else None,
        "n_steps": len(step_rows),
        "step_parse_rate": len(parsed) / max(len(step_rows), 1),
        "error_type_counts": {
            e: sum(s["error_type"] == e for s in parsed)
            for e in ("none", "planning", "binding", "arithmetic")
        },
    }
    atomic_write_json(out / "parse_report.json", report)
    rng = random.Random(cfg["seed"])
    sample = rng.sample(parsed, min(cfg["parse_sample_size"], len(parsed)))
    lines = ["# Parsed-step sample for manual inspection (phase 0.3)\n"]
    for k, s in enumerate(sample, 1):
        lines.append(
            f"## {k}. {s['uid']} step {s['index']}\n\n```\n{s['step_text']}\n```\n"
            f"parsed: op `{s['op']}`, a={s['a']}, b={s['b']}, stated={s['stated']}; "
            f"error_type={s['error_type']}, distractor_used={s['distractor_used']}, "
            f"gold_op={s['gold_op']}\n\nmanual verdict: \n"
        )
    sample_path = out / "parse_sample.md"
    if sample_path.exists() and "manual verdict: \n" not in sample_path.read_text():
        print(f"[resume] keeping {sample_path}: it already holds manual verdicts")
    else:
        atomic_write_text(sample_path, "\n".join(lines))
    stage.mark({"n_traces": len(traces), "n_steps": len(step_rows)})
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------- extract


def cmd_extract(cfg: dict[str, Any], force: bool = False) -> None:
    """Residuals at every layer at three kinds of position, written as shards of
    `extract_chunk` traces (acts_question / acts_last: one row per trace;
    acts_steps: one row per labelled step, contiguous per trace)."""
    from rfd.harness.generate import forward
    from rfd.harness.model import load
    from rfd.harness.prompts import chat_ids, cot_messages

    out = Path(cfg["out_dir"])
    inputs = file_fingerprint(out / "traces.jsonl", out / "steps.jsonl")
    stage = StageMarker(out, "extract", inputs)
    if stage.skip(force):
        return
    check_manifest(out, cfg["harness"]["model"])
    traces = list(read_jsonl(out / "traces.jsonl"))
    steps = list(read_jsonl(out / "steps.jsonl"))
    steps_by_uid: dict[str, list[int]] = {}
    for k, s in enumerate(steps):
        steps_by_uid.setdefault(s["uid"], []).append(k)
    # step rows must be contiguous and in trace order for row-block shards
    flat = [k for t in traces for k in steps_by_uid.get(t["uid"], [])]
    if flat != list(range(len(steps))):
        raise RuntimeError("steps.jsonl is not in traces.jsonl order")
    loaded = load(cfg["harness"]["model"])
    model, tok = loaded.model, loaded.tokenizer
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    layers = list(range(n_layers))
    dtype = cfg["harness"]["acts_dtype"]
    arrays = {
        name: ShardedArray(out, name, (n_layers, d), dtype, inputs)
        for name in ("acts_question", "acts_last", "acts_steps")
    }
    ec = cfg.get("extract_chunk", 64)
    bs = cfg["harness"]["forward_batch_size"]
    n_shards = (len(traces) + ec - 1) // ec
    absmax = 0.0
    for c in range(n_shards):
        if all(a.has(c) for a in arrays.values()):
            continue
        chunk = traces[c * ec : (c + 1) * ec]
        q = np.zeros((len(chunk), n_layers, d), np.float32)
        last = np.zeros((len(chunk), n_layers, d), np.float32)
        srows = [k for t in chunk for k in steps_by_uid.get(t["uid"], [])]
        st = np.zeros((len(srows), n_layers, d), np.float32)
        s0 = (
            srows[0]
            if srows
            else sum(len(steps_by_uid.get(t["uid"], [])) for t in traces[: c * ec])
        )
        for b0 in range(0, len(chunk), bs):
            batch = chunk[b0 : b0 + bs]
            seqs, pos, slots = [], [], []
            for tr in batch:
                p = tr.get("prompt_ids") or chat_ids(tok, cot_messages(tr["question"]))
                seq = p + tr["gen_ids"]
                n_p = len(p)
                sidx = steps_by_uid.get(tr["uid"], [])
                positions = [n_p - 1, len(seq) - 1] + [
                    n_p + steps[k]["gen_token_index"] for k in sidx
                ]
                seqs.append(seq)
                pos.append(positions)
                slots.append(sidx)
            res = forward(model, tok, seqs, capture_layers=layers, capture_positions=pos)[
                "resid"
            ].numpy()  # [bs, L, k, d]
            for j in range(len(batch)):
                q[b0 + j] = res[j, :, 0]
                last[b0 + j] = res[j, :, 1]
                for m, k in enumerate(slots[j]):
                    st[k - s0] = res[j, :, 2 + m]
        absmax = max(absmax, float(np.abs(q).max()), float(np.abs(last).max()))
        arrays["acts_question"].write(c, c * ec, q)
        arrays["acts_last"].write(c, c * ec, last)
        arrays["acts_steps"].write(c, s0, st)
        print(f"extract shard {c + 1}/{n_shards}", flush=True)
    arrays["acts_question"].complete(len(traces), n_shards)
    arrays["acts_last"].complete(len(traces), n_shards)
    arrays["acts_steps"].complete(len(steps), n_shards)
    atomic_write_json(
        out / "acts_index.json",
        {
            "trace_uids": [t["uid"] for t in traces],
            "n_steps": len(steps),
            "n_layers": n_layers,
            "d_model": d,
            "dtype": dtype,
        },
    )
    stage.mark({"n_traces": len(traces), "n_steps": len(steps), "absmax_this_run": absmax})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "label", "extract"])
    ap.add_argument("--config", default="configs/traces.yaml")
    ap.add_argument(
        "--harness", default=None, help="override harness config (e.g. configs/debug.yaml)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N traces of each split")
    ap.add_argument("--force", action="store_true", help="redo a stage that is already complete")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.harness, args.out)
    record_config(cfg)
    {
        "generate": lambda: cmd_generate(cfg, args.limit, args.force),
        "label": lambda: cmd_label(cfg, args.force),
        "extract": lambda: cmd_extract(cfg, args.force),
    }[args.cmd]()


if __name__ == "__main__":
    main()
