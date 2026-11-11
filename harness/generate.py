"""Trace generation: greedy (T=0, batch size 1) and the sampled subset (T=0.7, k=5).

Input: a JSONL file of problems, one object per line with at least
    {"problem_id": str, "template_id": str, "variant": str, "question": str, "answer": number}
Output: a JSONL file of traces, one per (problem, prompt spec[, sample]).

Usage (on the GPU server):
    python -m harness.generate --model gemma-2-9b-it --problems data/problems.jsonl \
        --out data/generated/gemma-2-9b-it/greedy.jsonl --mode greedy
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import torch

from .config import experiment_config, model_config
from .models import load_hf, load_tokenizer, pick_device
from .prompts import PromptSpec, all_prompt_specs, encode_chat


def greedy_generation_config(tokenizer, max_new_tokens: int):
    """A pure greedy configuration that ignores every default in the model's generation_config.

    Built from scratch so that Qwen's default repetition penalty (and any default temperature,
    top-p or top-k) cannot leak in (Section 3.9.1).
    """
    from transformers import GenerationConfig

    eos = _eos_ids(tokenizer)
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0],
    )


def sampled_generation_config(tokenizer, max_new_tokens: int, temperature: float):
    from transformers import GenerationConfig

    eos = _eos_ids(tokenizer)
    return GenerationConfig(
        do_sample=True,
        num_beams=1,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0],
    )


def is_pure_greedy(gen_cfg) -> bool:
    """Check 5 of Table 5: no penalty, temperature, top-p or top-k sampling."""
    return (
        not gen_cfg.do_sample
        and gen_cfg.num_beams == 1
        and (gen_cfg.repetition_penalty in (None, 1.0))
        and getattr(gen_cfg, "no_repeat_ngram_size", 0) in (None, 0)
    )


def _eos_ids(tokenizer) -> list[int]:
    ids = [tokenizer.eos_token_id]
    for tok in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return [i for i in ids if i is not None]


@torch.no_grad()
def generate_one(model, tokenizer, prompt_ids: list[int], gen_cfg, seed: int | None = None) -> dict:
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device)
    if seed is not None:
        torch.manual_seed(seed)
    # use_model_defaults=False: since transformers 4.50, fields of an explicit config that sit at
    # the global default (e.g. repetition_penalty=1.0) are otherwise REPLACED by the model's
    # shipped defaults (Qwen ships repetition_penalty=1.05). Caught by tests/test_harness_offline.py.
    out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                         generation_config=gen_cfg, use_model_defaults=False)
    new_ids = out[0, len(prompt_ids):].tolist()
    return {
        "prompt_ids": prompt_ids,
        "output_ids": new_ids,
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "hit_max_tokens": len(new_ids) >= gen_cfg.max_new_tokens,
    }


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(model_key: str, problems_path: str, out_path: str, mode: str = "greedy",
        specs: list[PromptSpec] | None = None, device: str | None = None,
        limit: int | None = None) -> Path:
    exp = experiment_config()["generation"]
    cfg = model_config(model_key)
    tok = load_tokenizer(cfg)
    model = load_hf(cfg, device=pick_device(device))
    max_new = exp["greedy"]["max_new_tokens"]
    if mode == "greedy":
        gen_cfg = greedy_generation_config(tok, max_new)
        assert is_pure_greedy(gen_cfg)
        k = 1
    elif mode == "sampled":
        gen_cfg = sampled_generation_config(tok, max_new, exp["sampled_subset"]["temperature"])
        k = exp["sampled_subset"]["k"]
    else:
        raise ValueError(mode)

    problems = read_jsonl(problems_path)[:limit]
    specs = specs or all_prompt_specs()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resume
        for r in read_jsonl(out):
            done.add((r["problem_id"], r["prompt"], r["sample"]))
    with out.open("a") as f:
        for p in problems:
            for spec in specs:
                ids = encode_chat(tok, p["question"], spec)
                for s in range(k):
                    key = (p["problem_id"], spec.tag, s)
                    if key in done:
                        continue
                    t0 = time.time()
                    # Stable across processes (Python's hash() is salted per run).
                    seed = None if mode == "greedy" else zlib.crc32("|".join(map(str, key)).encode())
                    rec = generate_one(model, tok, ids, gen_cfg, seed=seed)
                    rec.update({
                        "problem_id": p["problem_id"], "template_id": p.get("template_id"),
                        "variant": p.get("variant"), "gold_answer": p.get("answer"),
                        "model": model_key, "prompt": spec.tag, "mode": mode, "sample": s,
                        "seed": seed, "seconds": round(time.time() - t0, 3),
                    })
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["greedy", "sampled"], default="greedy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.problems, a.out, a.mode, device=a.device, limit=a.limit)


if __name__ == "__main__":
    main()
