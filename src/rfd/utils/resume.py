"""Results-directory resolution and restart-safe outputs.

Every phase writes under a results root (default `results/`; set
RFD_RESULTS_DIR, e.g. to a Google Drive folder on Colab) and can be killed at
any point: on restart it skips whatever already completed.

  results_path   map a config path "results/<x>" into the results root
  atomic_*       write-to-temp then rename, so a file is either whole or absent
  StageMarker    "stage X finished on these inputs" markers (skip whole stages)
  ResultCache    one JSON file per finished unit of work (a condition, a probe
                 job, a generation chunk)
  ShardedArray   row-sharded activation arrays: each shard is written once,
                 atomically; `open_array` assembles them into one .npy (in
                 RFD_SCRATCH_DIR when set, so the Drive copy stays shard-only)
  check_manifest refuse to mix outputs of different models in one directory
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

RESULTS_ENV = "RFD_RESULTS_DIR"
SCRATCH_ENV = "RFD_SCRATCH_DIR"


def results_root() -> Path:
    return Path(os.environ.get(RESULTS_ENV) or "results")


def results_path(path: str | Path) -> Path:
    """`results/<rest>` -> `<results root>/<rest>`; other paths are unchanged."""
    p = Path(path)
    if not p.is_absolute() and p.parts[:1] == ("results",):
        return results_root().joinpath(*p.parts[1:])
    return p


# --------------------------------------------------------------------------- atomic writes


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=float))


def atomic_write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in records))


def atomic_save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with tmp.open("wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- stages


def file_fingerprint(*paths: str | Path) -> str:
    """Short content hash of input files: caches and stage markers keyed on it
    go stale automatically when an upstream stage is redone."""
    h = hashlib.sha1()
    for p in paths:
        with Path(p).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()[:12]


class StageMarker:
    """`<out>/.done/<stage>.json` exists iff the stage finished; it records the
    fingerprint of the stage's inputs, so a stage whose inputs changed reruns."""

    def __init__(self, out: str | Path, stage: str, inputs: str | None = None) -> None:
        self.path = Path(out) / ".done" / f"{stage}.json"
        self.stage, self.inputs = stage, inputs

    def done(self) -> bool:
        if not self.path.exists():
            return False
        return json.loads(self.path.read_text()).get("inputs") == self.inputs

    def mark(self, info: dict[str, Any] | None = None) -> None:
        atomic_write_json(self.path, {"inputs": self.inputs, **(info or {})})

    def skip(self, force: bool) -> bool:
        """True (and say so) if the stage is complete for these inputs and not forced."""
        if self.done() and not force:
            print(f"[resume] {self.stage}: already complete ({self.path}); --force to redo")
            return True
        if self.path.exists() and not force:
            print(f"[resume] {self.stage}: inputs changed since it completed; redoing")
        return False


# --------------------------------------------------------------------------- per-unit cache


def _safe_key(key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._=-]+", "_", key)[:80]
    return f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"


class ResultCache:
    """JSON results keyed by a string, one file each, written atomically."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self.dir / f"{_safe_key(key)}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Any:
        return json.loads(self._path(key).read_text())["value"]

    def put(self, key: str, value: Any) -> None:
        atomic_write_json(self._path(key), {"key": key, "value": value})

    def get_or_compute(self, key: str, fn: Callable[[], Any]) -> Any:
        if self.has(key):
            return self.get(key)
        value = fn()
        self.put(key, value)
        return value

    def n_cached(self) -> int:
        return len(list(self.dir.glob("*.json"))) if self.dir.exists() else 0


# --------------------------------------------------------------------------- sharded arrays

FP16_MAX = float(np.finfo(np.float16).max)


def check_fp16_range(arr: np.ndarray, what: str) -> float:
    """Largest |value|; raise before a float16 cast would turn it into inf."""
    a = float(np.max(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(a) or a >= FP16_MAX:
        raise ValueError(
            f"{what}: |activation| reaches {a:.3g}, beyond float16 range; "
            "set `acts_dtype: float32` in the harness config"
        )
    return a


class ShardedArray:
    """A [n_rows, *row_shape] array stored as shards under <out>/shards/<name>/.

    Shard k holds a contiguous block of rows; `write(k, rows_start, arr)` is
    atomic and idempotent, `has(k)` tells a restart what to skip, and
    `complete(n_rows, n_shards)` records the full layout for `open_array`.
    """

    def __init__(
        self,
        out: str | Path,
        name: str,
        row_shape: tuple[int, ...],
        dtype: str,
        inputs: str | None = None,
    ) -> None:
        self.out, self.name = Path(out), name
        self.dir = self.out / "shards" / name
        self.row_shape, self.dtype = tuple(row_shape), np.dtype(dtype)
        # shards made from different inputs (an upstream stage was redone) are stale
        tag = self.dir / "inputs.json"
        want = {"inputs": inputs, "row_shape": list(self.row_shape), "dtype": self.dtype.str}
        if tag.exists() and json.loads(tag.read_text()) != want:
            import shutil

            print(f"[resume] {self.dir}: made from other inputs; discarding its shards")
            shutil.rmtree(self.dir)
        if not tag.exists():
            atomic_write_json(tag, want)

    def _shard(self, k: int) -> Path:
        return self.dir / f"{k:05d}.npy"

    def has(self, k: int) -> bool:
        return (self.dir / f"{k:05d}.json").exists()

    def write(self, k: int, rows_start: int, arr: np.ndarray) -> None:
        arr = np.asarray(arr)
        if arr.shape[1:] != self.row_shape:
            raise ValueError(f"{self.name} shard {k}: shape {arr.shape} vs rows {self.row_shape}")
        if self.dtype == np.float16:
            check_fp16_range(arr, f"{self.name} shard {k}")
        atomic_save_npy(self._shard(k), arr.astype(self.dtype))
        atomic_write_json(self.dir / f"{k:05d}.json", {"start": rows_start, "n": len(arr)})

    def complete(self, n_rows: int, n_shards: int) -> None:
        layout = []
        for k in range(n_shards):
            if not self.has(k):
                raise RuntimeError(f"{self.name}: shard {k} missing")
            layout.append(json.loads((self.dir / f"{k:05d}.json").read_text()))
        pos = 0
        for k, s in enumerate(layout):
            if s["start"] != pos:
                raise RuntimeError(f"{self.name}: shard {k} starts at {s['start']}, expected {pos}")
            pos += s["n"]
        if pos != n_rows:
            raise RuntimeError(f"{self.name}: shards cover {pos} rows, expected {n_rows}")
        atomic_write_json(
            self.dir / "complete.json",
            {
                "n_rows": n_rows,
                "n_shards": n_shards,
                "row_shape": list(self.row_shape),
                "dtype": self.dtype.str,
            },
        )


def _assembled_path(out: Path, name: str) -> Path:
    scratch = os.environ.get(SCRATCH_ENV)
    if not scratch:
        return out / f"{name}.npy"
    tag = hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:10]
    return Path(scratch) / tag / f"{name}.npy"


def open_array(out: str | Path, name: str) -> np.ndarray:
    """Read-only memmap of <out>/<name>: a plain .npy if one exists, otherwise the
    assembled shards (assembled once, streamed shard by shard, then reused)."""
    out = Path(out)
    plain = out / f"{name}.npy"
    if plain.exists() and not (out / "shards" / name).exists():
        return np.load(plain, mmap_mode="r")
    meta_path = out / "shards" / name / "complete.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{name}: no {plain} and no completed shards in {meta_path.parent}")
    meta = json.loads(meta_path.read_text())
    shape = (meta["n_rows"], *meta["row_shape"])
    dtype = np.dtype(meta["dtype"])
    target = _assembled_path(out, name)
    if target.exists():
        arr = np.load(target, mmap_mode="r")
        if arr.shape == shape and arr.dtype == dtype:
            return arr
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(target)
    with tmp.open("wb") as f:
        np.lib.format.write_array_header_2_0(
            f,
            {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape},
        )
        for k in range(meta["n_shards"]):
            f.write(np.ascontiguousarray(np.load(out / "shards" / name / f"{k:05d}.npy")).tobytes())
    os.replace(tmp, target)
    return np.load(target, mmap_mode="r")


# --------------------------------------------------------------------------- model manifest


def check_manifest(out: str | Path, model_cfg: dict[str, Any]) -> None:
    """Pin an output directory to one model configuration. Restarting with a
    different model (or dtype / attention) into the same directory is an error,
    so partial outputs of two models can never be merged."""
    keys = ("model_name", "dtype", "attn_implementation")
    want = {k: model_cfg.get(k) for k in keys}
    path = Path(out) / "run_manifest.json"
    if path.exists():
        have = json.loads(path.read_text())
        if {k: have.get(k) for k in keys} != want:
            raise RuntimeError(
                f"{out} holds outputs of {have}, not {want}. Use a fresh results directory "
                f"(set {RESULTS_ENV}) or delete it deliberately."
            )
        return
    atomic_write_json(path, want)
