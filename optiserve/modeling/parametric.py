"""Parametric performance model: execution time as a function of memory.

Fits ``rt(m) = a0 + a1 * exp(-m / a2)`` to profiling samples via non-linear
least squares. Persisted as a joblib pickle (the ``modeled_functions/*.mdl``
cache).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


#: Smallest admissible decay constant. ``a2`` appears as ``exp(-m / a2)``, so it
#: must stay strictly positive; anything smaller than this is numerically a step
#: function and is not a curve the profiler could have measured.
_MIN_DECAY_MB = 1e-3


def _default_bounds() -> tuple[list, list]:
    """Physically admissible parameter region for ``a0 + a1 * exp(-m / a2)``.

    All three parameters are non-negative by construction: ``a0`` is the
    asymptotic execution time as memory grows, ``a1`` the (positive) speed-up
    available from adding memory, and ``a2`` the decay constant in MB. The fit
    used to be unbounded, which let a sparse or noisy sample set — exactly what
    the sampler's three-point seeding phase produces — land on non-physical
    parameters such as ``a0 = -3.7e5, a1 = +3.7e5``. Those look plausible
    *inside* the profiled range and then go negative when the workflow bridge
    extrapolates past it, which ``build_app3`` does: it fits on ≤3 008 MB and
    materialises the profile out to 9 984 MB.

    (The 18 committed ``modeled_functions/*.mdl`` all sit well inside this
    region, so bounding does not move any published model.)
    """
    return ([0.0, 0.0, _MIN_DECAY_MB], [np.inf, np.inf, np.inf])


@dataclass
class ParamFunction:
    """Callable fitted curve ``rt = f(memory)`` with fit/persist/minimize."""

    function: Any = model_function
    bounds: tuple[list, list] = field(default_factory=_default_bounds)
    params: Any = None

    def __call__(self, x: float | np.ndarray):
        return self.function(x, *self.params)

    def initial_guess(self, exploration: Exploration) -> np.ndarray:
        """A dimensionally sensible starting point for the three parameters.

        The previous seed was ``[durations[0] // 10] * 3`` for all three — the
        same number used as a time (ms), an amplitude (ms) and a decay constant
        (MB). For a fast function that evaluates to ``[2, 2, 2]``: a 2 MB decay
        constant makes ``exp(-m/a2)`` underflow to zero at every sampled memory,
        the exponential term carries no gradient, and the optimiser leaves
        ``a1``/``a2`` exactly where they started — a fitted "curve" that is a
        flat line, so ``minimize`` always returns the smallest memory.

        Instead: ``a0`` = the fastest observation (the asymptote), ``a1`` = the
        observed spread (the available speed-up), ``a2`` = the midpoint of the
        sampled memory range (the scale over which that speed-up is realised).
        """
        durations = np.asarray(exploration.durations, dtype=float)
        memories = np.asarray(exploration.memories, dtype=float)

        fastest = float(np.min(durations)) if durations.size else 1.0
        spread = float(np.max(durations) - fastest) if durations.size else 1.0
        scale = float(np.mean(memories)) if memories.size else 512.0

        return np.array(
            [
                max(fastest, 0.0),
                max(spread, 1.0),
                max(scale, _MIN_DECAY_MB * 10),
            ],
            dtype=float,
        )

    def fit(self, exploration: Exploration) -> None:
        lower, upper = self.bounds
        if self.params is None:
            p0 = self.initial_guess(exploration)
        else:
            # Warm-start from the previous fit, but a stored parameter set can
            # predate a bounds change (or come from an older .mdl); clip it in
            # rather than letting curve_fit reject an out-of-bounds p0.
            p0 = np.clip(np.asarray(self.params, dtype=float), lower, upper)

        self.params = curve_fit(
            f=self.function,
            xdata=exploration.memories,
            ydata=exploration.durations,
            maxfev=_MAX_FEV,
            p0=p0,
            bounds=self.bounds,
        )[0]

    def minimize(
        self,
        memory_space: np.ndarray,
        latency_constraint_threshold_ms: float | None = None,
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
    def load(cls, path: str | Path) -> ParamFunction:
        return joblib.load(path)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("ParamFunction saved to %s.", path)
