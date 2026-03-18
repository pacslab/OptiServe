"""Builder and validator for OptiServe workflow graphs.

A workflow is a probabilistic control-flow graph over function nodes with
sentinel ``Start`` / ``End`` nodes and edge weights that are transition
probabilities. This class formalizes that schema and provides the bridge from
per-function performance models to the discrete ``perf_profile`` tables the
application model and optimizer consume — the step that previously only existed
as ad-hoc code in the notebooks.

Two ways to add a function:

- :meth:`add_function` — a fixed single-configuration node (memory + latency),
  for pure application performance modeling.
- :meth:`add_ml_function` — a multi-variant node whose ``perf_profile`` is
  materialized from each variant's performance model over a memory grid, for
  optimization.
"""
from __future__ import annotations

from typing import Hashable, Iterable, List, Optional

import networkx as nx

from optiserve.workflow.node import FunctionNode, ModelVariant


class WorkflowGraph:
    START = "Start"
    END = "End"

    def __init__(self):
        self._graph = nx.DiGraph()
        self._graph.add_node(self.START)
        self._graph.add_node(self.END)
        self._functions: dict = {}

    # -- construction ------------------------------------------------------- #
    def add_function(
        self,
        node_id: Hashable,
        memory_mb: float,
        response_time_ms: float,
    ) -> "WorkflowGraph":
        """Add a fixed single-configuration function node."""
        self._graph.add_node(node_id, mem=memory_mb, rt=response_time_ms)
        return self

    def add_ml_function(
        self,
        node_id: Hashable,
        variants: List[ModelVariant],
        memory_grid: Iterable[int],
        *,
        initial_memory: Optional[int] = None,
        initial_variant: int = 0,
    ) -> "WorkflowGraph":
        """Add a multi-variant function node, materializing its ``perf_profile``
        from each variant's performance model over ``memory_grid`` (the bridge)."""
        node = FunctionNode(node_id, variants, list(memory_grid))
        profile = node.profile_table()
        mem0 = int(initial_memory if initial_memory is not None else node.memory_grid[0])
        self._graph.add_node(
            node_id,
            perf_profile=profile,
            models_list=node.model_names,
            accuracy_list=node.accuracies,
            mem=mem0,
            rt=profile[initial_variant][mem0],
        )
        self._functions[node_id] = node
        return self

    def add_edge(self, u: Hashable, v: Hashable, probability: float) -> "WorkflowGraph":
        """Add a directed edge with a transition probability."""
        self._graph.add_weighted_edges_from([(u, v, probability)])
        return self

    def add_edges(self, edges: Iterable[tuple]) -> "WorkflowGraph":
        """Add many ``(u, v, probability)`` edges at once."""
        self._graph.add_weighted_edges_from(list(edges))
        return self

    # -- validation / export ----------------------------------------------- #
    def validate(self) -> "WorkflowGraph":
        """Check the structural invariants the application model relies on.

        Note: out-edge probabilities are intentionally *not* required to sum to
        1 — a node may combine a probabilistic branch with a deterministic
        (tp=1) parallel fan-out, so the sum is context-dependent.
        """
        g = self._graph
        if self.START not in g or self.END not in g:
            raise ValueError("Workflow must contain Start and End nodes.")
        if g.in_degree(self.START) != 0:
            raise ValueError("Start must have no incoming edges.")
        if g.out_degree(self.END) != 0:
            raise ValueError("End must have no outgoing edges.")

        functions = [n for n in g.nodes if n not in (self.START, self.END)]
        for node in functions:
            if not nx.has_path(g, self.START, node):
                raise ValueError(f"Node {node!r} is not reachable from Start.")
            if not nx.has_path(g, node, self.END):
                raise ValueError(f"Node {node!r} cannot reach End.")

        for u, v, data in g.edges(data=True):
            weight = data.get("weight")
            if weight is None or not (0 < weight <= 1):
                raise ValueError(
                    f"Edge {u!r}->{v!r} must have a probability weight in (0, 1]."
                )
        return self

    def to_networkx(self) -> nx.DiGraph:
        """Return a copy of the underlying networkx DiGraph, ready for
        :class:`~optiserve.modeling.application_model.ApplicationPerformanceModeling`."""
        return self._graph.copy()
