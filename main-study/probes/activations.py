"""Train/dev residuals from the S4 cache, one float16 array per hook (CPU, one pass over the shards).

Only index entries of TRAIN and DEV traces are read; test rows are skipped by the index before
any shard row is touched (hard rule: the held-out test templates stay locked).

Output (gitignored), under data/generated/traindev_resid/:
  rows.jsonl              one line per stored row: trace_id, kind, step, tok, split, template_id
  L{hook_index:02d}.npy   float16 [n_rows, d_model]; hook names in meta.json (index 0 = resid_pre 0,
                          index l+1 = blocks.l.hook_resid_post)
  meta.json               hooks, n_rows, the S4 manifest index_sha256, counts by split and kind

    python -m probes.activations            # ~10 min, ~7 GB
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harness.config import REPO_ROOT

from .labels import STORE, read_jsonl, SCORED, check_inputs

OUT = REPO_ROOT / "data" / "generated" / "traindev_resid"


def hook_index(layer: int | None) -> int:
    """Hook index in the store: None -> blocks.0.hook_resid_pre, l -> blocks.l.hook_resid_post."""
    return 0 if layer is None else layer + 1


def load_rows() -> list[dict]:
    return read_jsonl(OUT / "rows.jsonl")


def load_layer(layer: int, rows: np.ndarray | None = None) -> np.ndarray:
    """float32 [n, d] at blocks.{layer}.hook_resid_post (rows: optional index array)."""
    a = np.load(OUT / f"L{hook_index(layer):02d}.npy", mmap_mode="r")
    a = a if rows is None else a[rows]
    return np.asarray(a, dtype=np.float32)


def meta() -> dict:
    return json.loads((OUT / "meta.json").read_text())


def extract() -> dict:
    from .labels import _sha

    check_inputs()
    man = json.loads((STORE / "manifest.json").read_text())
    committed = json.loads((REPO_ROOT / "reports" / "m1" / "s4" / "manifest.json").read_text())
    if man != committed or _sha(STORE / "index.jsonl") != committed["index_sha256"]:
        raise SystemExit("the local S4 store does not match the committed S4 manifest")
    split_of, tmpl_of = {}, {}
    for r in read_jsonl(SCORED):
        split_of[r["trace_id"]] = r["split"]
        tmpl_of[r["trace_id"]] = r["template_id"]
    entries = []
    with (STORE / "index.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            if split_of.get(e["trace_id"]) in ("train", "dev"):
                entries.append(e)
    # one stored row per (shard, row); several index entries may share a row
    uniq: dict[tuple, int] = {}
    rows = []
    for e in entries:
        key = (e["shard"], e["row"])
        if key in uniq:
            continue
        uniq[key] = len(rows)
        rows.append(e)
    by_shard = defaultdict(list)
    for i, e in enumerate(rows):
        by_shard[e["shard"]].append((e["row"], i))
    OUT.mkdir(parents=True, exist_ok=True)
    hooks, d = man["hooks"], man["d_model"]
    outs = [np.lib.format.open_memmap(OUT / f"L{j:02d}.npy", mode="w+", dtype=np.float16,
                                      shape=(len(rows), d)) for j in range(len(hooks))]
    t0 = time.time()
    for k, (sh, lst) in enumerate(sorted(by_shard.items())):
        src = np.load(STORE / sh, mmap_mode="r")
        lst.sort()
        src_rows = np.array([r for r, _ in lst])
        dst_rows = np.array([i for _, i in lst])
        block = np.asarray(src[src_rows])                    # [m, hooks, d] float16
        for j in range(len(hooks)):
            outs[j][dst_rows] = block[:, j]
        print(f"shard {k + 1}/{len(by_shard)} {sh}: {len(lst)} rows ({time.time() - t0:.0f}s)",
              flush=True)
    for o in outs:
        o.flush()
    # every entry (including the ones sharing a row) points at its stored row
    with (OUT / "rows.jsonl").open("w") as f:
        for e in entries:
            i = uniq[(e["shard"], e["row"])]
            f.write(json.dumps({"i": i, "trace_id": e["trace_id"], "kind": e["kind"],
                                "step": e["step"], "tok": e["tok"],
                                "split": split_of[e["trace_id"]],
                                "template_id": tmpl_of[e["trace_id"]]}) + "\n")
    m = {"hooks": hooks, "d_model": d, "n_rows": len(rows), "n_entries": len(entries),
         "store_index_sha256": man["index_sha256"],
         "by_split": dict(Counter(split_of[e["trace_id"]] for e in entries)),
         "by_kind": dict(Counter(e["kind"] for e in entries)),
         "test_rows_read": 0, "seconds": round(time.time() - t0, 1)}
    (OUT / "meta.json").write_text(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    print(json.dumps(extract(), indent=2))
