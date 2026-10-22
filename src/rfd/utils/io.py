"""JSONL / path helpers used across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = ensure_parent(path)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Records of a JSONL file. A truncated LAST line (a write killed mid-line)
    is dropped with a warning; a malformed line anywhere else is an error."""
    with Path(path).open("r") as f:
        lines = [ln.strip() for ln in f]
    lines = [ln for ln in lines if ln]
    for k, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise
            print(f"[read_jsonl] {path}: dropping truncated last line")


def device_for(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
