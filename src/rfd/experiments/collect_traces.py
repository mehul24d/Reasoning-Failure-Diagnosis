"""GSM-Symbolic CoT traces (main + NoOp distractor variants) with step labels and activations.

Stages (resumable; each writes under <out_dir>/):
  generate  greedy marked-CoT traces            -> traces.jsonl
  label     step parsing vs the gold program    -> steps.jsonl, parse_report.json, parse_sample.md
  extract   teacher-forced residuals, all layers -> acts_{question,last,steps}.npy + index files

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
from rfd.utils.config import load_yaml
from rfd.utils.io import read_jsonl, write_jsonl


def load_cfg(path: str, harness: str | None, out: str | None) -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["harness"] = load_yaml(harness or cfg["harness_config"])
    if out:
        cfg["out_dir"] = out
    return cfg


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


def cmd_generate(cfg: dict[str, Any], limit: int | None) -> None:
    from rfd.harness.generate import generate
    from rfd.harness.model import library_versions, load
    from rfd.harness.prompts import chat_ids, cot_messages
    from rfd.utils.seed import set_seed

    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    items, meta = build_items(cfg, limit)
    (out / "splits.json").write_text(json.dumps(meta, indent=2))
    path = out / "traces.jsonl"
    done = {r["uid"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [i for i in items if i["uid"] not in done]
    print(f"{len(items)} traces, {len(done)} done, {len(todo)} to generate", flush=True)
    loaded = load(cfg["harness"]["model"])
    tok = loaded.tokenizer
    gen_cfg = cfg["harness"]["generation"]
    for it in todo:
        it["prompt_ids"] = chat_ids(tok, cot_messages(it["question"]))
    todo.sort(key=lambda i: len(i["prompt_ids"]))  # less padding per batch
    bs = gen_cfg["batch_size"]
    for c0 in range(0, len(todo), cfg["chunk_size"]):
        chunk = todo[c0 : c0 + cfg["chunk_size"]]
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
        with path.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n_done = len(done) + c0 + len(chunk)
        print(
            f"{n_done}/{len(items)} traces; chunk acc {np.mean([r['correct'] for r in recs]):.2f}",
            flush=True,
        )
    (out / "generate_meta.json").write_text(
        json.dumps(
            {
                "model": cfg["harness"]["model"],
                "generation": gen_cfg,
                "versions": library_versions(),
            },
            indent=2,
        )
    )


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


def cmd_label(cfg: dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    from rfd.parsing.equations import BinStep

    out = Path(cfg["out_dir"])
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
    write_jsonl(out / "steps.jsonl", step_rows)
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
    (out / "parse_report.json").write_text(json.dumps(report, indent=2))
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
    (out / "parse_sample.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------- extract


def cmd_extract(cfg: dict[str, Any]) -> None:
    from rfd.harness.generate import forward
    from rfd.harness.model import load

    out = Path(cfg["out_dir"])
    traces = list(read_jsonl(out / "traces.jsonl"))
    steps = list(read_jsonl(out / "steps.jsonl"))
    loaded = load(cfg["harness"]["model"])
    model, tok = loaded.model, loaded.tokenizer
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    layers = list(range(n_layers))
    steps_by_uid: dict[str, list[int]] = {}
    for k, s in enumerate(steps):
        steps_by_uid.setdefault(s["uid"], []).append(k)
    mm = lambda name, n: np.lib.format.open_memmap(out / name, "w+", np.float16, (n, n_layers, d))  # noqa: E731
    a_q, a_last, a_steps = (
        mm("acts_question.npy", len(traces)),
        mm("acts_last.npy", len(traces)),
        mm("acts_steps.npy", len(steps)),
    )
    bs = cfg["harness"]["forward_batch_size"]
    for b0 in range(0, len(traces), bs):
        batch = traces[b0 : b0 + bs]
        seqs, pos, slots = [], [], []
        for tr in batch:
            p = tr.get("prompt_ids") or None
            if p is None:
                from rfd.harness.prompts import chat_ids, cot_messages

                p = chat_ids(tok, cot_messages(tr["question"]))
            seq = p + tr["gen_ids"]
            n_p = len(p)
            sidx = steps_by_uid.get(tr["uid"], [])
            positions = [n_p - 1, len(seq) - 1] + [n_p + steps[k]["gen_token_index"] for k in sidx]
            seqs.append(seq)
            pos.append(positions)
            slots.append(sidx)
        res = forward(model, tok, seqs, capture_layers=layers, capture_positions=pos)[
            "resid"
        ]  # [bs, L, k, d]
        for j, tr in enumerate(batch):
            i = b0 + j
            a_q[i] = res[j, :, 0].half().numpy()
            a_last[i] = res[j, :, 1].half().numpy()
            for m, k in enumerate(slots[j]):
                a_steps[k] = res[j, :, 2 + m].half().numpy()
        if (b0 // bs) % 20 == 0:
            print(f"extract {b0 + len(batch)}/{len(traces)}", flush=True)
    for a in (a_q, a_last, a_steps):
        a.flush()
    (out / "acts_index.json").write_text(
        json.dumps(
            {
                "trace_uids": [t["uid"] for t in traces],
                "n_steps": len(steps),
                "n_layers": n_layers,
                "d_model": d,
            }
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "label", "extract"])
    ap.add_argument("--config", default="configs/traces.yaml")
    ap.add_argument(
        "--harness", default=None, help="override harness config (e.g. configs/debug.yaml)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N traces of each split")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.harness, args.out)
    {
        "generate": lambda: cmd_generate(cfg, args.limit),
        "label": lambda: cmd_label(cfg),
        "extract": lambda: cmd_extract(cfg),
    }[args.cmd]()


if __name__ == "__main__":
    main()
