"""Load the YAML configurations in configs/ (plan Appendix E)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open() as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_id: str
    tl_name: str
    n_layers: int
    d_model: int
    dtype: str
    role: str
    sae: dict[str, Any] | None
    gated: bool = False


def model_config(key: str) -> ModelConfig:
    models = load_yaml("models.yaml")["models"]
    if key not in models:
        raise KeyError(f"Unknown model {key!r}; known: {sorted(models)}")
    m = models[key]
    return ModelConfig(
        key=key,
        hf_id=m["hf_id"],
        tl_name=m["tl_name"],
        n_layers=int(m["n_layers"]),
        d_model=int(m["d_model"]),
        dtype=m["dtype"],
        role=m["role"],
        sae=m.get("sae"),
        gated=bool(m.get("gated", False)),
    )


def experiment_config() -> dict[str, Any]:
    return load_yaml("experiment.yaml")
