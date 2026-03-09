"""Regression guard for the consolidated exception hierarchy."""
import pytest

from optiserve.exceptions import (
    CostCalculationError,
    FunctionTimeout,
    InvocationError,
    NotEnoughMemory,
    OptimizationError,
    OptiServeError,
    SamplingError,
    UnfeasibleConstraint,
)


def test_everything_derives_from_optiserve_error():
    for exc in (
        InvocationError,
        NotEnoughMemory,
        FunctionTimeout,
        SamplingError,
        OptimizationError,
        UnfeasibleConstraint,
        CostCalculationError,
    ):
        assert issubclass(exc, OptiServeError)


def test_timeout_coupled_to_not_enough_memory():
    # Intentional: the profiling sampler treats a timeout like an OOM result.
    assert issubclass(FunctionTimeout, NotEnoughMemory)
    assert issubclass(NotEnoughMemory, InvocationError)


def test_invocation_error_carries_duration():
    assert InvocationError("boom", duration_ms=42).duration_ms == 42


def test_cost_calculation_error_message_is_set():
    # Regression: the old __initn__ typo left .message unset.
    assert CostCalculationError().message == "Error in cost calculation."


def test_raise_and_catch_via_base():
    with pytest.raises(OptiServeError):
        raise UnfeasibleConstraint()
