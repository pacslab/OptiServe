"""Typed schema for the function nodes of a workflow graph.

These formalize the node-attribute convention that previously lived only inside
the experiment notebooks: each function node carries one or more ML-model
variants, each with a performance model (latency = f(memory)) and an optional
measured accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional


@dataclass
class ModelVariant:
    """One model a function can serve.

    ``performance_model`` maps memory (MB) → latency (ms) — typically a fitted
    :class:`~optiserve.modeling.parametric.ParamFunction`, but any callable
    works. ``accuracy`` is the measured accuracy when known; when ``None`` the
    optimizer falls back to a normalized rank derived from variant order.
    """

    name: str
    performance_model: Callable[[float], float]
    accuracy: Optional[float] = None


@dataclass
class FunctionNode:
    """A workflow function with its variants and the memory grid over which its
    discrete performance profile is materialized."""

    node_id: Hashable
    variants: List[ModelVariant]
    memory_grid: List[int]

    def profile_table(self) -> List[Dict[int, float]]:
        """Discrete ``perf_profile``: one ``{memory_mb: latency_ms}`` dict per
        variant, evaluated over the memory grid."""
        return [
            {int(m): float(v.performance_model(m)) for m in self.memory_grid}
            for v in self.variants
        ]

    @property
    def model_names(self) -> List[str]:
        return [v.name for v in self.variants]

    @property
    def accuracies(self) -> List[Optional[float]]:
        return [v.accuracy for v in self.variants]
