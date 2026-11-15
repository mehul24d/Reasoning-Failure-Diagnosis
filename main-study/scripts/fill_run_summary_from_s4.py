"""Fill the cache and store fields of the S3 run summary from the committed S4 records (D45).

S3 ran in a kernel restored from the S2 output, so its `report` stage saw no activation cache:
the summary written on Kaggle has `stages.cache` pending, and `disk_bytes.activation_store`
counts only the partial store restored from S2 (the verify stage's shards).
This fills those fields on CPU from reports/m1/s4/ (cache.json and the store manifest.json),
records where each value came from (with the source files' sha256), and keeps the Kaggle
summary unchanged next to it as `<name>_kaggle.json`.

    python scripts/fill_run_summary_from_s4.py reports/m1/s3/s2_run_summary_gemma-2-2b-it.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from harness.config import REPO_ROOT
from harness.pipeline import sha256


def fill(summary: dict, s4: Path) -> dict:
    cache = json.loads((s4 / "cache.json").read_text())
    man = json.loads((s4 / "manifest.json").read_text())
    if cache.get("status") != "complete" or cache.get("store_bytes") != man["bytes"]:
        raise ValueError("S4 records are inconsistent: cache.json is not complete or its store size "
                         "differs from the manifest")
    if man["n_rows"] != cache["rows"]:
        raise ValueError("S4 records are inconsistent: row counts differ")
    out = json.loads(json.dumps(summary))
    kaggle_cache = out["stages"].get("cache", {})
    out["stages"]["cache"] = cache
    disk = out["disk_bytes"]
    kaggle_store = disk.get("activation_store", 0)
    disk["activation_store"] = man["bytes"]
    disk["total"] = sum(v for k, v in disk.items() if k != "total")
    out["activation_store"] = {k: man[k] for k in ("model", "n_rows", "n_index_entries", "traces",
                                                   "d_model", "storage_dtype", "compute_dtype",
                                                   "positions", "index_sha256", "bytes")}
    out["activation_store"].update(n_hooks=len(man["hooks"]), n_shards=len(man["shards"]),
                                   orphan_shards=len(man["orphan_shards"]))
    out["cpu_fill"] = {
        "decision": "D45: S3 restored from the S2 output, so the S4 cache was not attached",
        "fields": ["stages.cache", "disk_bytes.activation_store", "disk_bytes.total", "activation_store"],
        "kaggle_values": {"stages.cache": kaggle_cache, "disk_bytes.activation_store": kaggle_store},
        "activation_store_bytes": "manifest.json `bytes`: all 55 shards of the S4 store, which "
                                  "include the shards S2's verify stage wrote. The index, the manifest "
                                  "and the fp32 reference arrays of the fp16 check are not included. "
                                  "The Kaggle value is the size of the partial store restored from S2 "
                                  "(verify-stage shards, reference arrays, index), not of the full store.",
        "sources": {f"reports/m1/s4/{f}": sha256(s4 / f) for f in ("cache.json", "manifest.json")},
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--s4", default=str(REPO_ROOT / "reports" / "m1" / "s4"))
    a = ap.parse_args(argv)
    p = Path(a.summary)
    keep = p.with_name(p.stem + "_kaggle.json")
    if not keep.exists():
        shutil.copy2(p, keep)
    filled = fill(json.loads(keep.read_text()), Path(a.s4))
    p.write_text(json.dumps(filled, indent=2, default=str))
    print(json.dumps({"stages.cache": filled["stages"]["cache"]["status"],
                      "disk_bytes": filled["disk_bytes"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
