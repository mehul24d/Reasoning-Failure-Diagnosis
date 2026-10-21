"""Run phase segmentation over every pilot trace's cached SAE features.

Usage: python -m rfd.legacy.segmentation.run_segmentation --experiment configs/legacy/experiment.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from tqdm import tqdm

from rfd.legacy.segmentation.phase_segmenter import segment_trace
from rfd.utils.config import load_experiment
from rfd.utils.io import read_jsonl, write_jsonl


def run_segmentation(experiment_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    segmentation_cfg = experiment_cfg["segmentation"]
    summaries = list(read_jsonl(f"{experiment_cfg['output_dir']}/sae_feature_summary.jsonl"))

    results = []
    for summary in tqdm(summaries, desc="segmenting traces"):
        npz = np.load(summary["feature_file"])
        feature_acts = npz["feature_acts"]
        result = segment_trace(feature_acts, segmentation_cfg)
        result["id"] = summary["id"]
        results.append(result)

    write_jsonl(experiment_cfg["segmentation_output"], results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/legacy/experiment.yaml")
    args = parser.parse_args()

    experiment_cfg = load_experiment(args.experiment)
    results = run_segmentation(experiment_cfg)
    avg_segments = sum(len(r["segments"]) for r in results) / max(1, len(results))
    print(f"Segmented {len(results)} traces. Avg segments per trace: {avg_segments:.2f}")


if __name__ == "__main__":
    main()
