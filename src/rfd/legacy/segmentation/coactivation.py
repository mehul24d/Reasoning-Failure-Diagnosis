"""Build a feature coactivation graph from a [n_tokens, d_sae] SAE feature
activation matrix."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


def _correlation_graph(feature_acts: np.ndarray, active_idx: np.ndarray, min_abs_corr: float) -> nx.Graph:
    sub = feature_acts[:, active_idx]
    # np.corrcoef needs variance > 0 per row; guard constant columns.
    stds = sub.std(axis=0)
    keep = stds > 1e-12
    sub, active_idx = sub[:, keep], active_idx[keep]
    if sub.shape[1] < 2:
        graph = nx.Graph()
        graph.add_nodes_from(active_idx.tolist())
        return graph

    corr = np.corrcoef(sub, rowvar=False)
    graph = nx.Graph()
    graph.add_nodes_from(active_idx.tolist())
    n = len(active_idx)
    for i in range(n):
        for j in range(i + 1, n):
            w = corr[i, j]
            # Coactivation = firing *together*, so only positive correlation
            # counts as an edge; anti-correlated features are not coactive.
            if w >= min_abs_corr and not np.isnan(w):
                graph.add_edge(int(active_idx[i]), int(active_idx[j]), weight=float(w))
    return graph


def _pmi_graph(feature_acts: np.ndarray, active_idx: np.ndarray, min_abs_corr: float) -> nx.Graph:
    binary = (feature_acts[:, active_idx] > 0).astype(np.float64)
    n_tokens = binary.shape[0]
    p_single = binary.mean(axis=0)  # P(feature active)
    joint = (binary.T @ binary) / n_tokens  # P(i and j active)

    graph = nx.Graph()
    graph.add_nodes_from(active_idx.tolist())
    n = len(active_idx)
    eps = 1e-12
    for i in range(n):
        for j in range(i + 1, n):
            p_ij = joint[i, j]
            denom = p_single[i] * p_single[j]
            if p_ij <= eps or denom <= eps:
                continue
            pmi = np.log(p_ij / denom)
            # Same rationale as correlation: only positive PMI (co-occur
            # more than chance) counts as coactivation.
            if pmi >= min_abs_corr:
                graph.add_edge(int(active_idx[i]), int(active_idx[j]), weight=float(pmi))
    return graph


def build_coactivation_graph(feature_acts: np.ndarray, coactivation_cfg: dict[str, Any]) -> nx.Graph:
    """feature_acts: [n_tokens, d_sae]. Only features active at least once
    across the trace become graph nodes."""
    active_idx = np.where((feature_acts > 0).any(axis=0))[0]
    if len(active_idx) == 0:
        return nx.Graph()

    method = coactivation_cfg.get("method", "correlation")
    min_abs = coactivation_cfg.get("min_abs_corr", 0.1)
    if method == "correlation":
        return _correlation_graph(feature_acts, active_idx, min_abs)
    if method == "pmi":
        return _pmi_graph(feature_acts, active_idx, min_abs)
    raise ValueError(f"Unknown coactivation method: {method}")
