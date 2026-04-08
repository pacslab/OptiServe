"""Typed result of an optimization run.

A :class:`NamedTuple`, so existing 6-tuple unpacking
(``rt, cost, acc, mem, model, iters = optimizer.BPBC(...)``) keeps working while
callers can also use named fields.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import NamedTuple


class OptimizationResult(NamedTuple):
    response_time_ms: float
    cost: float
    accuracy: float
    memory_config: dict[Hashable, int]
    model_config: dict[Hashable, int]
    iterations: int
