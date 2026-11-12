"""Read the residual cache on CPU (no GPU needed): index, residuals, SAE features.

SAE activations are computed later from the stored float16 residuals (prereg/S2.md amendment
A4): each row is cast back to float32 and encoded by the Gemma Scope SAE of its layer on CPU.
`fp16_sae_check` compares those features with the ones from the float32 reference copies kept
for the first traces of the equivalence sample, so the effect of float16 storage on the SAE
read-out is reported, not assumed.

    from cache import store
    idx = store.load_index("cache/store/gemma-2-2b-it")
    x = store.residuals("cache/store/gemma-2-2b-it", idx[:100], "blocks.12.hook_resid_post")
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def manifest(store) -> dict:
    return json.loads((Path(store) / "manifest.json").read_text())


def load_index(store) -> list[dict]:
    with (Path(store) / "index.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=8)
def _shard(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def residuals(store, entries: list[dict], hook: str) -> np.ndarray:
    """float32 [len(entries), d_model] at `hook` (a name from the manifest's `hooks`)."""
    store = Path(store)
    j = manifest(store)["hooks"].index(hook)
    return np.stack([_shard(str(store / e["shard"]))[e["row"], j] for e in entries]).astype(np.float32)


def fp16_sae_check(store, verify_records: list[dict], hook: str, encode) -> dict:
    """SAE features from stored float16 rows vs the float32 references (`encode`: x -> features)."""
    store = Path(store)
    j = manifest(store)["hooks"].index(hook)
    diffs, l0_changes, n = [], 0, 0
    for r in verify_records:
        if "fp16" not in r:
            continue
        ref = np.load(store / "ref_fp32" / r["fp16"]["ref_file"])[:, j]
        first = {}
        for e in r["rows"]:
            first.setdefault(e["tok"], e)
        stored = np.stack([_shard(str(store / first[t]["shard"]))[first[t]["row"], j]
                           for t in sorted(first)]).astype(np.float32)
        a, b = np.asarray(encode(ref)), np.asarray(encode(stored))
        diffs.append(float(np.abs(a - b).max()))
        l0_changes += int(((a > 0) != (b > 0)).sum())
        n += a.size
    return {"traces": len(diffs), "max_abs_feature_diff": max(diffs, default=None),
            "active_flag_changes": l0_changes, "feature_values": n}
