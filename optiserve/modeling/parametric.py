"""Parametric performance model: execution time as a function of memory.

Fits ``rt(m) = a0 + a1 * exp(-m / a2)`` to profiling samples via non-linear
least squares. Persisted as a joblib pickle (the ``modeled_functions/*.mdl``
cache).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import joblib
import numpy as np
from scipy.optimize import curve_fit

from optiserve.exceptions import UnfeasibleConstraint
from optiserve.logging import get_logger
from optiserve.profiling.sample import Exploration

logger = get_logger(__name__)

# curve_fit iteration cap — generous, so fitting rarely fails to converge.
_MAX_FEV = int(1e8)


def model_function(x, a0, a1, a2):
    """Execution-time model: ``a0 + a1 * exp(-x / a2)`` (falls back to ``a0``
    when ``a2 == 0``). ``x`` is memory (MB)."""
    return (a0 + a1 * np.exp(-x / a2)) if a2 != 0 else a0


def _default_bounds() -> Tuple[list, list]:
    return ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])


@dataclass
class ParamFunction:
    """Callable fitted curve ``rt = f(memory)`` with fit/persist/minimize."""

    function: Any = model_function
    bounds: Tuple[list, list] = field(default_factory=_default_bounds)
    params: Any = None

    def __call__(self, x: Union[float, np.ndarray]):
        return self.function(x, *self.params)

    def fit(self, exploration: Exploration) -> None:
        if self.params is None:
            self.params = [exploration.durations[0] // 10] * 3
        self.params = curve_fit(
            f=self.function,
            xdata=exploration.memories,
            ydata=exploration.durations,
            maxfev=_MAX_FEV,
            p0=self.params,
            bounds=self.bounds,
        )[0]

    def minimize(
        self,
        memory_space: np.ndarray,
        latency_constraint_threshold_ms: Optional[float] = None,
    ) -> int:
        """Return the cost-optimal memory (argmin of ``rt(m) * m``).

        When a latency constraint is given, only feasible memories are
        considered; if none are feasible, ``UnfeasibleConstraint`` is raised
        (previously this was silently swallowed and the unconstrained optimum
        returned)."""
        exec_time = self(memory_space)
        costs = exec_time * memory_space

        if latency_constraint_threshold_ms is not None:
            feasible = exec_time < latency_constraint_threshold_ms
            if not np.any(feasible):
                raise UnfeasibleConstraint(
                    "No feasible memory configuration for latency requirement "
                    f"{latency_constraint_threshold_ms} ms."
                )
            memory_space = memory_space[feasible]
            costs = costs[feasible]

        return int(memory_space[np.argmin(costs)])

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParamFunction":
        return joblib.load(path)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("ParamFunction saved to %s.", path)
