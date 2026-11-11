"""Shared statistics (plan Section 3.14.2): template-clustered bootstrap, Holm, Benjamini–Hochberg."""
from __future__ import annotations

from typing import Callable

import numpy as np


def cluster_bootstrap(cluster_ids: np.ndarray, statistic: Callable[[np.ndarray], float],
                      n_resamples: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict:
    """Resample whole clusters (templates) with replacement.

    `statistic(cluster_weights)` receives a weight per UNIQUE cluster (how many times it was
    drawn; the observed estimate uses all-ones) and returns a float. Passing weights instead of
    re-indexed data keeps every resample vectorised.
    """
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    k = len(uniq)
    rng = np.random.default_rng(seed)
    est = statistic(np.ones(k))
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        boots[b] = statistic(np.bincount(rng.integers(0, k, k), minlength=k).astype(float))
    boots = boots[np.isfinite(boots)]
    a = (1 - ci) / 2
    lo, hi = np.quantile(boots, [a, 1 - a]) if len(boots) else (np.nan, np.nan)
    # two-sided bootstrap p-value for H0: statistic == 0
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))) if len(boots) else np.nan
    return {"estimate": float(est), "ci_low": float(lo), "ci_high": float(hi), "p": float(p),
            "n_clusters": int(k), "n_valid_resamples": int(len(boots))}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        adj[i] = running
    return adj.tolist()


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / sp)
