"""Problem sources -> problems JSONL (the input format of harness.generate).

At M0 only GSM8K *train* is used, for SAE validation and the E0 sparsity check, so no
GSM-Symbolic template (the study population) is touched before its splits are pre-registered.
GSM-Symbolic, GSM8K test and MATH-500 loaders are added at S2.
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
                    "answer": str(gold)})
    return out


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gsm8k"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = gsm8k(a.split, a.n, a.seed)
    write_jsonl(rows, a.out)
    print(f"wrote {len(rows)} problems to {a.out}")


if __name__ == "__main__":
    main()
