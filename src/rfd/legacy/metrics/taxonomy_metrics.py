"""Per-error-type metrics: accuracy, mean L0, mean feature entropy, and a
'sparsity collapse delta' proxy.

sparsity_collapse_delta := mean_L0(first half of trace) - mean_L0(second
half of trace). A positive delta means active-feature count drops in the
back half of the trace -- a candidate signal for "giving up" / collapse
near a failure. This is a simple proxy, not a validated collapse
detector -- TODO for Semester 7 to validate against manual annotations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sparsity_collapse_delta(l0_array: np.ndarray) -> float:
    n = len(l0_array)
    if n < 2:
        return 0.0
    mid = n // 2
    first_half, second_half = l0_array[:mid], l0_array[mid:]
    return float(first_half.mean() - second_half.mean())


def compute_per_error_type_metrics(
    taxonomy_rows: list[dict[str, Any]], feature_summaries_by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    df = pd.DataFrame(taxonomy_rows)
    results: dict[str, dict[str, float]] = {}

    for error_type, group in df.groupby("error_type"):
        l0s, entropies, deltas = [], [], []
        for trace_id in group["id"]:
            summary = feature_summaries_by_id.get(trace_id)
            if summary is None:
                continue
            l0s.append(summary["mean_l0"])
            entropies.append(summary["mean_entropy"])
            npz = np.load(summary["feature_file"])
            deltas.append(_sparsity_collapse_delta(npz["l0"]))

        results[error_type] = {
            "count": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "mean_l0": float(np.mean(l0s)) if l0s else float("nan"),
            "mean_entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "mean_sparsity_collapse_delta": float(np.mean(deltas)) if deltas else float("nan"),
        }
    return results
