"""Build the failure taxonomy CSV + per-error-type metrics for the pilot.

Usage: python -m rfd.taxonomy.run_taxonomy --experiment configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from rfd.metrics.taxonomy_metrics import compute_per_error_type_metrics
from rfd.taxonomy.classifier import build_taxonomy_rows
from rfd.taxonomy.csv_template import write_taxonomy_csv
from rfd.utils.config import load_experiment
from rfd.utils.io import ensure_parent, read_jsonl


def run_taxonomy(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    traces = list(read_jsonl(experiment_cfg["traces_file"]))
    segmentations = list(read_jsonl(experiment_cfg["segmentation_output"]))
    segmentations_by_id = {s["id"]: s for s in segmentations}

    feature_summaries = list(read_jsonl(f"{experiment_cfg['output_dir']}/sae_feature_summary.jsonl"))
    feature_summaries_by_id = {s["id"]: s for s in feature_summaries}

    rows = build_taxonomy_rows(traces, segmentations_by_id)
    write_taxonomy_csv(
        rows,
        experiment_cfg["taxonomy_output"],
        fraction=experiment_cfg["annotation"]["taxonomy_manual_check_fraction"],
        seed=experiment_cfg["seed"],
    )

    per_error_metrics = compute_per_error_type_metrics(rows, feature_summaries_by_id)
    metrics_path = ensure_parent(experiment_cfg["metrics_output"])
    metrics_path.write_text(json.dumps(per_error_metrics, indent=2))

    return per_error_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    args = parser.parse_args()

    experiment_cfg = load_experiment(args.experiment)
    metrics = run_taxonomy(experiment_cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
