"""Orchestrates: feature matrix -> coactivation graph -> communities ->
changepoints -> predicted phase boundaries/labels for one trace.

Phase LABELING is a first-pass positional heuristic (segments are labeled
in temporal order: first -> problem_understanding, last -> self_verification,
remainder split between operation_planning / arithmetic_execution). This is
a known simplification -- real phase attribution should use feature
interpretation, not segment position. TODO for Semester 7.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rfd.legacy.segmentation.changepoint import detect_changepoints
from rfd.legacy.segmentation.coactivation import build_coactivation_graph
from rfd.legacy.segmentation.community import detect_communities
from rfd.legacy.segmentation.community_activation import compute_community_activation

PHASES = [
    "problem_understanding",
    "operation_planning",
    "arithmetic_execution",
    "self_verification",
]


def segments_from_boundaries(boundaries: list[int], n_tokens: int) -> list[tuple[int, int]]:
    points = [0] + sorted(boundaries) + [n_tokens]
    return [(points[i], points[i + 1]) for i in range(len(points) - 1) if points[i] < points[i + 1]]


def label_segments(segments: list[tuple[int, int]]) -> list[str]:
    """Positional heuristic labeling onto the 4 fixed phases."""
    n = len(segments)
    if n == 0:
        return []
    if n == 1:
        return ["arithmetic_execution"]
    if n == 2:
        return ["problem_understanding", "self_verification"]
    if n == 3:
        return ["problem_understanding", "arithmetic_execution", "self_verification"]

    # n >= 4: first -> understanding, last -> verification, remainder split
    # evenly between planning and execution, execution gets the larger half.
    labels = ["problem_understanding"]
    middle_n = n - 2
    n_planning = middle_n // 2
    labels += ["operation_planning"] * n_planning
    labels += ["arithmetic_execution"] * (middle_n - n_planning)
    labels.append("self_verification")
    return labels


def segment_trace(feature_acts: np.ndarray, segmentation_cfg: dict[str, Any]) -> dict[str, Any]:
    n_tokens = feature_acts.shape[0]
    graph = build_coactivation_graph(feature_acts, segmentation_cfg["coactivation"])
    community_assignment = detect_communities(graph, segmentation_cfg["community_detection"])
    series = compute_community_activation(feature_acts, community_assignment)
    boundaries = detect_changepoints(series, segmentation_cfg["changepoint"])
    segments = segments_from_boundaries(boundaries, n_tokens)
    labels = label_segments(segments)

    return {
        "n_tokens": n_tokens,
        "n_communities": series.shape[1],
        "boundaries": boundaries,
        "segments": [{"start": s, "end": e, "phase": p} for (s, e), p in zip(segments, labels)],
    }
