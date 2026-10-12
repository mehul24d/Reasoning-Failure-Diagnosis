"""YAML config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_experiment(experiment_config_path: str | Path) -> dict[str, Any]:
    """Load the top-level experiment config and inline the sub-configs it
    references (model/dataset/sae/segmentation), keyed by their file stem.
    """
    exp = load_yaml(experiment_config_path)
    base = Path(experiment_config_path).parent.parent  # repo root
    for key in ("model_config", "dataset_config", "sae_config", "segmentation_config"):
        if key in exp:
            sub_path = Path(exp[key])
            if not sub_path.is_absolute():
                sub_path = base / sub_path
            exp[key.replace("_config", "")] = load_yaml(sub_path)
    return exp
