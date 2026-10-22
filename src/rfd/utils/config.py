"""YAML config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def repo_root(start: str | Path | None = None) -> Path:
    """Nearest ancestor of `start` (default: cwd) that contains pyproject.toml.
    Config paths inside YAML files are relative to this directory."""
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def load_experiment(experiment_config_path: str | Path) -> dict[str, Any]:
    """Load the top-level experiment config and inline the sub-configs it
    references (model/dataset/sae/segmentation), keyed by their file stem.
    """
    exp = load_yaml(experiment_config_path)
    base = repo_root(experiment_config_path)
    for key in ("model_config", "dataset_config", "sae_config", "segmentation_config"):
        if key in exp:
            sub_path = Path(exp[key])
            if not sub_path.is_absolute():
                sub_path = base / sub_path
            exp[key.replace("_config", "")] = load_yaml(sub_path)
    return exp


def load_run_config(
    path: str | Path,
    harness: str | None = None,
    out: str | None = None,
    traces: str | None = None,
) -> dict[str, Any]:
    """An experiment config with its harness (model) config inlined under
    "harness", CLI overrides applied, and every results path mapped into the
    results root (RFD_RESULTS_DIR, default results/; see rfd.utils.resume)."""
    from rfd.utils.resume import results_path

    cfg = load_yaml(path)
    cfg["harness"] = load_yaml(harness or cfg["harness_config"])
    cfg["harness"].setdefault("acts_dtype", "float16")
    if out:
        cfg["out_dir"] = out
    if traces:
        cfg["traces_dir"] = traces
    for key in ("out_dir", "traces_dir"):
        if key in cfg:
            cfg[key] = str(results_path(cfg[key]))
    return cfg
