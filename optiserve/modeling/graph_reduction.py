"""Pure graph helpers shared by the application performance model.

These operate on a passed networkx graph (no class state), so the same logic can
be reused for both the execution-count (``ne``) de-looping pass and the
response-time simplification pass without duplication.

The reduction's expensive queries are **topology-only**: their answers depend on
the node set, the edge set and the transition probabilities, and *not* on the
per-node response times. That distinction is what makes them cacheable across an
optimization sweep — the optimizer rewrites ``rt``/``mem`` thousands of times
while the topologies it reduces repeat the same handful of shapes. Profiling the
brute-force ground-truth sweep put ~33 % of total runtime in
:func:`discover_cycles` and ~21 % in the model's ``is_simple`` predicate, all of
it recomputing answers for graphs that had not structurally changed.

:func:`topology_key` captures node and edge *iteration order*, not just the
sets. networkx enumerates cycles in an order that follows that iteration order,
and the caller mutates the graph while walking the result — so an
order-insensitive key would be a correctness bug, not an optimization.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

import networkx as nx

__all__ = ["CachedGraphAnalysis", "discover_cycles", "topology_key"]


def discover_cycles(graph: nx.DiGraph, start_point: str = "Start") -> list[list]:
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


def topology_key(graph: nx.DiGraph, *extra: Hashable) -> tuple[Any, ...]:
    """An exact, order-sensitive fingerprint of a graph's structure.

    Two graphs with the same key give identical results from every topology-only
    algorithm — including the identical *ordering* the reduction depends on.
    """
    return (
        tuple(graph.nodes),
        tuple((u, v, data.get("weight")) for u, v, data in graph.edges(data=True)),
        *extra,
    )


class CachedGraphAnalysis:
    """Memoizes topology-only graph queries against an injected cache.

    Given a :class:`~optiserve.modeling.cache.NullEvaluationCache` it is a
    transparent pass-through, so caching stays strictly opt-in and the default
    path is byte-identical to the uncached implementation.
    """

    def __init__(self, cache: Any) -> None:
        self._cache = cache

    @property
    def stats(self) -> Any:
        return self._cache.stats

    def memoized(
        self, tag: str, graph: nx.DiGraph, compute: Callable[[], Any], *extra: Hashable
    ) -> Any:
        """Return ``compute()``, cached on ``(tag, topology of graph, extra)``.

        ``compute`` must be a pure function of the graph's topology and edge
        weights. It is passed as a thunk so the caller keeps its exact original
        implementation — the cache never becomes a second, drifting copy of the
        algorithm it accelerates.
        """
        key = (tag, *topology_key(graph, *extra))
        found, cached = self._cache.lookup(key)
        if found:
            return cached
        value = compute()
        self._cache.put(key, value)
        return value

    def discover_cycles(self, graph: nx.DiGraph, start_point: str = "Start") -> list[list]:
        key = ("cycles", *topology_key(graph, start_point))
        found, cached = self._cache.lookup(key)
        if found:
            # The reduction mutates the graph while iterating these paths and
            # some callers append to them; hand out a fresh copy every time.
            return [list(cycle) for cycle in cached]
        cycles = discover_cycles(graph, start_point)
        self._cache.put(key, [tuple(cycle) for cycle in cycles])
        return cycles
