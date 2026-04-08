"""Per-function model-variant accuracy.

Each function node exposes an ordered list of model variants. Their accuracy is
either **measured** (provided on the node as ``accuracy_list``) or, when not
supplied, a **normalized rank** ``i / N`` for ``i = 1..N`` — i.e. the best (last)
variant scores 1.0. The optimizer combines per-node accuracies via a
caller-supplied end-to-end formula.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable

import networkx as nx


class AccuracyModel:
    """Resolves per-node, per-variant accuracy values (measured or ranked)."""

    def __init__(self, values: dict[Hashable, list[float]]):
        self._values = values

    @classmethod
    def from_graph(cls, graph: nx.DiGraph, function_nodes: list[Hashable]) -> AccuracyModel:
        values: dict[Hashable, list[float]] = {}
        for node in function_nodes:
            variants = graph.nodes[node]["models_list"]
            n = len(variants)
            measured = graph.nodes[node].get("accuracy_list")
            if measured is not None and all(a is not None for a in measured):
                values[node] = [float(a) for a in measured]
            else:
                values[node] = [i / n for i in range(1, n + 1)]
        return cls(values)

    def variant_accuracies(self, node: Hashable) -> list[float]:
        return self._values[node]

    def end_to_end(
        self, model_configuration: dict[Hashable, int], formula: Callable[..., float]
    ) -> float:
        """Apply the end-to-end ``formula`` to the selected variant's accuracy
        per node, in node order."""
        accuracies = [self._values[node][model_configuration[node]] for node in self._values]
        return formula(*accuracies)
