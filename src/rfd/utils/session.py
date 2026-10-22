"""Helpers for time-limited, non-interactive notebook sessions (Kaggle "Save &
Run All"; see notebooks/run_on_kaggle.ipynb and DECISIONS D16).

  restore_results  seed the results directory from a previous session's output
                   (an attached dataset / notebook output: a results/ folder or
                   a results zip), never overwriting files already present
  run_targets      run make targets in order under a wall-clock deadline; a
                   target still running at the deadline is killed (every
                   phase is resumable, so the next session continues it)
  zip_results      zip the results directory for the session's output
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any


def _markers(root: Path) -> int:
    return sum(1 for _ in root.rglob(".done/*.json"))


def _is_results_dir(p: Path) -> bool:
    return p.is_dir() and (
        any(p.rglob("run_manifest.json")) or any(p.rglob(".done")) or (p / "phase0").is_dir()
    )


def find_results_sources(inputs_root: str | Path, max_depth: int = 4) -> list[Path]:
    """results/ folders and results*.zip files under inputs_root (depth-limited)."""
    root = Path(inputs_root)
    if not root.exists():
        return []
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        here = Path(dirpath)
        if here.name == "results" and _is_results_dir(here):
            found.append(here)
            dirnames[:] = []  # do not descend into a results tree
            continue
        found += [here / f for f in filenames if f.startswith("results") and f.endswith(".zip")]
        if depth >= max_depth:
            dirnames[:] = []
    return sorted(found)


def _copy_missing(src: Path, dest: Path) -> tuple[int, int]:
    copied = kept = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        target = dest / f.relative_to(src)
        if target.exists():
            kept += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
    return copied, kept


def restore_results(inputs_root: str | Path, dest: str | Path) -> dict[str, Any]:
    """Copy the most complete previous results (most stage markers) into dest.
    Files already in dest win, so re-running this cell is harmless."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    candidates = []
    sources = find_results_sources(inputs_root)
    if any(s.is_dir() for s in sources):  # a folder needs no extraction; its zip is a copy
        sources = [s for s in sources if s.is_dir()]
    with tempfile.TemporaryDirectory() as tmp:
        for k, src in enumerate(sources):
            if src.suffix == ".zip":
                out = Path(tmp) / f"zip{k}"
                with zipfile.ZipFile(src) as z:
                    z.extractall(out)
                inner = [p for p in out.rglob("results") if _is_results_dir(p)]
                root = inner[0] if inner else out
            else:
                root = src
            candidates.append((_markers(root), str(src), root))
        if not candidates:
            return {"restored_from": None, "candidates": [], "copied": 0, "kept": 0}
        candidates.sort(key=lambda c: (c[0], c[1]))
        n_markers, name, root = candidates[-1]
        copied, kept = _copy_missing(root, dest)
    return {
        "restored_from": name,
        "stage_markers": n_markers,
        "candidates": [c[1] for c in candidates],
        "copied": copied,
        "kept": kept,
    }


def _stream(proc: subprocess.Popen, log) -> None:
    for line in iter(proc.stdout.readline, ""):
        log.write(line)
        log.flush()


def run_targets(
    targets: list[str],
    deadline: float,
    cmd: list[str] | None = None,
    min_seconds: float = 120.0,
    grace: float = 30.0,
    log=None,
) -> list[dict[str, Any]]:
    """Run `cmd + [target]` for each target in order until the deadline (epoch
    seconds). Stops at the first failure (later phases depend on earlier ones)
    or when time runs out; the killed target resumes in the next session."""
    import sys

    cmd = cmd or ["make"]
    log = log or sys.stdout
    status = []
    for k, t in enumerate(targets):
        remaining = deadline - time.time()
        if remaining < min_seconds:
            status += [{"target": x, "status": "not started (session time)"} for x in targets[k:]]
            break
        t0 = time.time()
        proc = subprocess.Popen(
            [*cmd, t],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # its own process group: the deadline kills all children
        )
        reader = threading.Thread(target=_stream, args=(proc, log), daemon=True)
        reader.start()
        try:
            rc = proc.wait(timeout=remaining)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = proc.returncode
        reader.join(timeout=5)
        row = {"target": t, "seconds": round(time.time() - t0, 1), "returncode": rc}
        if timed_out:
            row["status"] = "stopped at session deadline (resumes next session)"
        else:
            row["status"] = "complete" if rc == 0 else "FAILED"
        status.append(row)
        log.write(f"[session] {t}: {row['status']} ({row['seconds']} s)\n")
        if row["status"] != "complete":
            status += [{"target": x, "status": "not started"} for x in targets[k + 1 :]]
            break
    return status


def dir_size_gb(path: str | Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**30


def zip_results(
    results_dir: str | Path, zip_path: str | Path, max_full_gb: float = 8.0
) -> dict[str, Any]:
    """Zip results_dir (archive root: results/). Above max_full_gb the activation
    shards and caches are left out of the zip (they remain in results_dir)."""
    results_dir, zip_path = Path(results_dir), Path(zip_path)
    size = dir_size_gb(results_dir)
    light = size > max_full_gb
    n = 0
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(f".{zip_path.name}.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in sorted(results_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(results_dir)
            if light and (rel.parts[1:2] in (("shards",), ("cache",)) or f.suffix == ".npy"):
                continue
            z.write(f, Path("results") / rel)
            n += 1
    os.replace(tmp, zip_path)
    return {"zip": str(zip_path), "files": n, "results_gb": round(size, 2), "light": light}


def write_status(path: str | Path, status: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(status, indent=2, default=str))
