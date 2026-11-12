"""Template split (plan Table 25: by template; test >= 40 templates, e.g. 40/20/40, provisional).

Deterministic: the sorted unique template ids are shuffled with random.Random(seed) and cut
into train/dev/test in the configured proportions. The split file is written once and
committed with the pre-registration that fixes it; later steps read it, never recompute it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_templates(template_ids, n_train: int = 40, n_dev: int = 20, n_test: int = 40,
                    seed: int = 0, min_test: int = 40) -> dict[str, list[str]]:
    ids = sorted(set(template_ids))
    total = n_train + n_dev + n_test
    if len(ids) > total:   # scale proportionally, test gets the rounding remainder
        n_train = round(len(ids) * n_train / total)
        n_dev = round(len(ids) * n_dev / total)
        n_test = len(ids) - n_train - n_dev
    elif len(ids) < total:
        # Fewer templates than the nominal split (97 study templates, amendment A6): the test
        # split keeps its size (>= min_test), and train and dev share the rest in their ratio.
        rest = len(ids) - n_test
        if rest < (n_train + n_dev) / 2:
            raise ValueError(f"{len(ids)} templates, but the split needs {total}: too few to keep "
                             f"{n_test} test templates and half of train + dev")
        n_train = round(rest * n_train / (n_train + n_dev))
        n_dev = rest - n_train
    if n_test < min_test:
        raise ValueError(f"test split has {n_test} templates; at least {min_test} required")
    random.Random(seed).shuffle(ids)
    return {"train": sorted(ids[:n_train]), "dev": sorted(ids[n_train:n_train + n_dev]),
            "test": sorted(ids[n_train + n_dev:])}


def main() -> None:
    from harness.config import experiment_config
    from harness.generate import read_jsonl

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = experiment_config()["data"]
    t = c["template_split"]
    rows = read_jsonl(a.problems)
    sp = split_templates([r["template_id"] for r in rows], t["train"], t["dev"], t["test"],
                         a.seed, c["min_test_templates"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"seed": a.seed, **sp}, indent=2))
    print({k: len(v) for k, v in sp.items()})


if __name__ == "__main__":
    main()
