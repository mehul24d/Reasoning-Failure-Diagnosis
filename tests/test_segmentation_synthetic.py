import numpy as np

from rfd.segmentation.changepoint import detect_changepoints
from rfd.segmentation.coactivation import build_coactivation_graph
from rfd.segmentation.community import detect_communities
from rfd.segmentation.community_activation import compute_community_activation
from rfd.segmentation.phase_segmenter import segment_trace


def _synthetic_two_block_features(seed: int = 0) -> np.ndarray:
    """20 tokens x 10 features. Features 0-4 fire in tokens [0,10), features
    5-9 fire in tokens [10,20) -- a clean single changepoint at token 10,
    and two anti-correlated feature communities."""
    rng = np.random.default_rng(seed)
    n_tokens, d_sae = 20, 10
    acts = np.zeros((n_tokens, d_sae))
    acts[:10, 0:5] = 1.0 + 0.05 * rng.standard_normal((10, 5))
    acts[10:, 5:10] = 1.0 + 0.05 * rng.standard_normal((10, 5))
    return np.clip(acts, 0, None)


SEGMENTATION_CFG = {
    "coactivation": {"method": "correlation", "min_abs_corr": 0.3},
    "community_detection": {"method": "louvain", "resolution": 1.0, "seed": 42},
    "changepoint": {
        "method": "ruptures",
        "ruptures": {"model": "l2", "penalty": 1.0, "min_size": 2},
    },
}


def test_coactivation_graph_separates_two_blocks():
    acts = _synthetic_two_block_features()
    graph = build_coactivation_graph(acts, SEGMENTATION_CFG["coactivation"])
    assert set(graph.nodes()) == set(range(10))
    # within-block edges should be positive, cross-block should be absent
    # or negative (and thus not present at this min_abs_corr threshold).
    for u, v, data in graph.edges(data=True):
        same_block = (u < 5) == (v < 5)
        if same_block:
            assert data["weight"] > 0


def test_community_detection_recovers_two_blocks():
    acts = _synthetic_two_block_features()
    graph = build_coactivation_graph(acts, SEGMENTATION_CFG["coactivation"])
    assignment = detect_communities(graph, SEGMENTATION_CFG["community_detection"])

    block_a_communities = {assignment[i] for i in range(5)}
    block_b_communities = {assignment[i] for i in range(5, 10)}
    assert len(block_a_communities) == 1
    assert len(block_b_communities) == 1
    assert block_a_communities != block_b_communities


def test_changepoint_detection_finds_boundary_near_10():
    acts = _synthetic_two_block_features()
    graph = build_coactivation_graph(acts, SEGMENTATION_CFG["coactivation"])
    assignment = detect_communities(graph, SEGMENTATION_CFG["community_detection"])
    series = compute_community_activation(acts, assignment)
    boundaries = detect_changepoints(series, SEGMENTATION_CFG["changepoint"])

    assert len(boundaries) >= 1
    assert any(abs(b - 10) <= 2 for b in boundaries)


def test_segment_trace_end_to_end_produces_four_or_fewer_labeled_segments():
    acts = _synthetic_two_block_features()
    result = segment_trace(acts, SEGMENTATION_CFG)

    assert result["n_tokens"] == 20
    assert len(result["segments"]) >= 1
    total_covered = sum(seg["end"] - seg["start"] for seg in result["segments"])
    assert total_covered == 20
    for seg in result["segments"]:
        assert seg["phase"] in {
            "problem_understanding",
            "operation_planning",
            "arithmetic_execution",
            "self_verification",
        }


def test_segment_trace_handles_empty_features():
    acts = np.zeros((5, 10))
    result = segment_trace(acts, SEGMENTATION_CFG)
    assert result["n_tokens"] == 5
    # no active features -> no graph nodes -> single catch-all segment
    assert len(result["segments"]) == 1
