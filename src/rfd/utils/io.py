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
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def device_for(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
