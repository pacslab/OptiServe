"""Tests for the parametric performance model (fit / minimize / persistence)."""
from pathlib import Path

import numpy as np
import pytest

from optiserve.exceptions import UnfeasibleConstraint
from optiserve.modeling.parametric import ParamFunction, model_function
from optiserve.profiling.sample import Exploration, Sample


def _synthetic_exploration(a0=100.0, a1=4000.0, a2=500.0):
    memories = np.arange(128, 3009, 128)
    samples = [
        Sample(memory_mb=int(m), duration_ms=float(model_function(m, a0, a1, a2)))
        for m in memories
    ]
    return Exploration(samples)


def test_bounds_default_not_shared_between_instances():
    # Regression: the dataclass previously shared one mutable bounds list.
    a, b = ParamFunction(), ParamFunction()
    assert a.bounds is not b.bounds


def test_fit_recovers_parameters():
    pf = ParamFunction()
    pf.fit(_synthetic_exploration(a0=100.0, a1=4000.0, a2=500.0))
    # exec time at large memory approaches a0.
    assert pf(100_000) == pytest.approx(100.0, abs=1.0)


def test_minimize_raises_on_infeasible_constraint():
    pf = ParamFunction()
    pf.fit(_synthetic_exploration())
    memory_space = np.arange(128, 3009, 128)
    # An impossibly tight latency requirement must now raise, not silently
    # return the unconstrained optimum.
    with pytest.raises(UnfeasibleConstraint):
        pf.minimize(memory_space, latency_constraint_threshold_ms=0.001)


@pytest.mark.parametrize("name", ["f1", "resnet_resnet-18"])
def test_cached_mdl_loads_and_evaluates(name):
    path = Path("modeled_functions") / f"{name}.mdl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    pf = ParamFunction.load(path)
    value = float(pf(1024))
    assert value > 0
