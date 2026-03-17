"""Pure graph helpers shared by the application performance model.

These operate on a passed networkx graph (no class state), so the same logic can
be reused for both the execution-count (``ne``) de-looping pass and the
response-time simplification pass without duplication.
"""
from __future__ import annotations

from typing import List

import networkx as nx


def discover_cycles(graph: nx.DiGraph, start_point: str = "Start") -> List[list]:
    """Return the graph's cycles, oriented head→tail.

    Combines ``nx.simple_cycles`` with a DFS ``find_cycle`` fallback (expanded
    into simple paths), then keeps only cycles whose first node is strictly
    closer to ``start_point`` than the last — i.e. oriented so ``cycle[-1]`` is
    the loop's exit/back-edge source.
    """
    cycles = list(nx.simple_cycles(graph))

    try:
        seed = [edge[0] for edge in nx.find_cycle(graph)]
        for path in nx.all_simple_paths(graph, source=seed[0], target=seed[-1]):
            if path not in cycles:
                cycles.append(path)
    except nx.NetworkXNoCycle:
        pass

    return [
        cycle
        for cycle in cycles
        if nx.shortest_path_length(graph, start_point, cycle[0])
        < nx.shortest_path_length(graph, start_point, cycle[-1])
    ]
