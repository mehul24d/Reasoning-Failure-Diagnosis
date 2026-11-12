"""Problem sources -> problems JSONL (the input format of harness.generate).

At M0 only GSM8K *train* is used, for SAE validation and the E0 sparsity check, so no
GSM-Symbolic template (the study population) is touched before its splits are pre-registered.

`gsm_symbolic` reads apple/GSM-Symbolic (configs main, p1, p2; template = its `id`). That
release has no no-op distractor variants: those come from the no-op recreation the plan uses
(Qu & Gomes), which is not public yet, so `variant` is always "base" here. Every row keeps
its solution text so parsing.gold can build the gold program.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from parsing.answers import gsm8k_gold


def gsm8k(split: str = "train", n: int | None = None, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    if n is not None:
        idx = sorted(random.Random(seed).sample(idx, n))
    out = []
    for i in idx:
        row = ds[i]
        gold = gsm8k_gold(row["answer"])
        out.append({"problem_id": f"gsm8k-{split}-{i}", "template_id": f"gsm8k-{split}-{i}",
                    "variant": "base", "source": "gsm8k", "question": row["question"],
                    "answer": str(gold), "solution": row["answer"]})
    return out


def gsm_symbolic(config: str = "main", templates: list[int] | None = None,
                 instances: int | None = None) -> list[dict]:
    """apple/GSM-Symbolic rows. `templates` restricts to those template ids; `instances` keeps the
    first n instances (by instance number) of each template."""
    from datasets import load_dataset

    ds = load_dataset("apple/GSM-Symbolic", config, split="test")
    keep = None if templates is None else set(templates)
    rows = [r for r in ds if keep is None or r["id"] in keep]
    rows.sort(key=lambda r: (r["id"], r["instance"]))
    out, seen = [], {}
    for r in rows:
        seen[r["id"]] = seen.get(r["id"], 0) + 1
        if instances is not None and seen[r["id"]] > instances:
            continue
        out.append({"problem_id": f"gsms-{config}-{r['id']}-{r['instance']}",
                    "template_id": f"gsms-{r['id']}", "variant": "base",
                    "source": "gsm_symbolic", "config": config, "question": r["question"],
                    "answer": str(gsm8k_gold(r["answer"])), "solution": r["answer"]})
    return out


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gsm8k", "gsm_symbolic"], required=True)
    ap.add_argument("--config", default="main", help="gsm_symbolic: main, p1 or p2")
    ap.add_argument("--instances", type=int, default=None, help="gsm_symbolic: per template")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.source == "gsm8k":
        rows = gsm8k(a.split, a.n, a.seed)
    else:
        rows = gsm_symbolic(a.config, instances=a.instances)
    write_jsonl(rows, a.out)
    print(f"wrote {len(rows)} problems to {a.out}")


if __name__ == "__main__":
    main()
