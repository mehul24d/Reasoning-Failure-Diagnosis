"""Community detection over the feature coactivation graph."""

from __future__ import annotations

from typing import Any

import networkx as nx


def detect_communities(graph: nx.Graph, community_cfg: dict[str, Any]) -> dict[int, int]:
    """Returns {feature_node: community_id}. Isolated nodes each get their
    own singleton community."""
    method = community_cfg.get("method", "louvain")

    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {node: i for i, node in enumerate(graph.nodes())}

    if method == "louvain":
        communities = nx.algorithms.community.louvain_communities(
            graph,
            weight="weight",
            resolution=community_cfg.get("resolution", 1.0),
            seed=community_cfg.get("seed", 42),
        )
    elif method == "leiden":
        try:
            import igraph as ig
            import leidenalg
        except ImportError as e:
            raise NotImplementedError(
                "Leiden community detection requires `igraph` + `leidenalg` "
                "(not in pyproject.toml deps). TODO for Semester 7. "
                "Use community_detection.method: louvain for now."
            ) from e
        node_list = list(graph.nodes())
        idx_of = {n: i for i, n in enumerate(node_list)}
        edges = [(idx_of[u], idx_of[v]) for u, v in graph.edges()]
        weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
        ig_graph = ig.Graph(n=len(node_list), edges=edges, edge_attrs={"weight": weights})
        partition = leidenalg.find_partition(
            ig_graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=community_cfg.get("resolution", 1.0),
        )
        communities = [{node_list[i] for i in part} for part in partition]
    else:
        raise ValueError(f"Unknown community detection method: {method}")

    assignment: dict[int, int] = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            assignment[node] = community_id
    return assignment
