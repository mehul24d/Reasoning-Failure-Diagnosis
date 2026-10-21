"""Aggregate per-token feature activations into per-token community
activation levels, producing the time series that changepoint detection
runs over."""

from __future__ import annotations

import numpy as np


def compute_community_activation(
    feature_acts: np.ndarray, community_assignment: dict[int, int]
) -> np.ndarray:
    """feature_acts: [n_tokens, d_sae]; community_assignment: {feature_idx: community_id}.

    Returns [n_tokens, n_communities], the mean activation of each
    community's member features at each token position.
    """
    n_tokens = feature_acts.shape[0]
    if not community_assignment:
        return np.zeros((n_tokens, 0))

    n_communities = max(community_assignment.values()) + 1
    series = np.zeros((n_tokens, n_communities))
    counts = np.zeros(n_communities)

    for feature_idx, community_id in community_assignment.items():
        series[:, community_id] += feature_acts[:, feature_idx]
        counts[community_id] += 1

    counts[counts == 0] = 1.0
    return series / counts
