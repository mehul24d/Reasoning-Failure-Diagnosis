"""Changepoint detection over the community-activation time series."""

from __future__ import annotations

from typing import Any

import numpy as np


def _ruptures_changepoints(series: np.ndarray, cfg: dict[str, Any]) -> list[int]:
    import ruptures as rpt

    n, d = series.shape
    algo = rpt.Pelt(model=cfg.get("model", "l2"), min_size=cfg.get("min_size", 2)).fit(series)

    # Adaptive, BIC-style penalty (standard practice for Pelt's l2 cost
    # when the data's own scale is not known ahead of time: pen ~=
    # scale * sigma^2 * d * log(n)). A fixed absolute `penalty` was found
    # during the pilot run to be badly miscalibrated for SAE-feature-scale
    # signals (mean activation ~0.1) -- it collapsed nearly every trace
    # into a single segment. `penalty_scale` is tuned once, uniformly
    # across all traces (not per-trace), targeting a handful of
    # boundaries per trace on the pilot data; see docs/PROGRESS.md.
    variance_estimate = max(series.var(axis=0).mean(), 1e-6)
    penalty = cfg.get("penalty_scale", 1.5) * variance_estimate * d * np.log(max(n, 2))

    breakpoints = algo.predict(pen=penalty)
    return breakpoints[:-1]  # ruptures appends len(series) as the final boundary


def _hmm_changepoints(series: np.ndarray, cfg: dict[str, Any]) -> list[int]:
    from hmmlearn.hmm import GaussianHMM

    n_states = min(cfg.get("n_states", 4), series.shape[0])
    model = GaussianHMM(
        n_components=n_states,
        n_iter=cfg.get("n_iter", 100),
        random_state=cfg.get("seed", 42),
    )
    model.fit(series)
    states = model.predict(series)
    boundaries = [i for i in range(1, len(states)) if states[i] != states[i - 1]]
    return boundaries


def detect_changepoints(series: np.ndarray, changepoint_cfg: dict[str, Any]) -> list[int]:
    """series: [n_tokens, n_communities]. Returns sorted token indices
    where a new phase is predicted to start (excludes index 0)."""
    if series.shape[0] < 2 or series.shape[1] == 0:
        return []

    method = changepoint_cfg.get("method", "ruptures")
    if method == "ruptures":
        return _ruptures_changepoints(series, changepoint_cfg.get("ruptures", {}))
    if method == "hmm":
        return _hmm_changepoints(series, changepoint_cfg.get("hmm", {}))
    raise ValueError(f"Unknown changepoint method: {method}")
