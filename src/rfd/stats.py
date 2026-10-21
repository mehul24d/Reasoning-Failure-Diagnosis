"""Cluster (template) bootstrap and small metric helpers.

The independent unit is the template: every CI resamples templates with
replacement and keeps all of a resampled template's items.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

N_BOOT = 10_000


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Point estimate + percentile CI of `stat(item_indices)` under template resampling.

    Resamples where `stat` is undefined (returns nan, e.g. one class only) are
    dropped; `n_valid` reports how many were kept.
    """
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(clusters)))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[c] for c in pick])
        vals[i] = stat(idx)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"estimate": float(point), "lo": float("nan"), "hi": float("nan"), "n_valid": 0}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_valid": int(len(vals)),
        "n_clusters": int(len(uniq)),
    }


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def safe_balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(balanced_accuracy_score(y, pred))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
