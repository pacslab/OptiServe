"""OptiServe exception hierarchy.

All library errors derive from :class:`OptiServeError`, so callers can catch
everything the framework raises with a single ``except OptiServeError``.

    Exception
    └── OptiServeError
        ├── InvocationError(message, duration_ms=None)   a Lambda invocation failed
        │   ├── NotEnoughMemory                          memory config too small — prune it
        │   │   └── FunctionTimeout                      the function timed out            [*]
        │   └── MaxInvocationAttemptsReached
        ├── SamplingError
        │   ├── LogParsingError
        │   └── NoMemoryLeft
        ├── OptimizationError
        │   └── UnfeasibleConstraint
        ├── CostCalculationError
        └── FunctionConfigurationError

[*] ``FunctionTimeout`` intentionally subclasses ``NotEnoughMemory``. The
    profiling sampler handles a timeout the same way it handles an
    out-of-memory result — by raising the explored memory floor (more memory
    generally means faster execution and a lower chance of timing out). This
    coupling is deliberate; the sampler's ``except NotEnoughMemory`` branch
    relies on it. Keep the relationship if you change this file.
"""

from __future__ import annotations

__all__ = [
    "CostCalculationError",
    "FunctionConfigurationError",
    "FunctionTimeout",
    "InvocationError",
    "LogParsingError",
    "MaxInvocationAttemptsReached",
    "NoMemoryLeft",
    "NotEnoughMemory",
    "OptiServeError",
    "OptimizationError",
    "SamplingError",
    "UnfeasibleConstraint",
]


class OptiServeError(Exception):
    """Base class for every error raised by OptiServe."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Invocation / profiling errors
# --------------------------------------------------------------------------- #
class InvocationError(OptiServeError):
    """A Lambda invocation failed. ``duration_ms`` is the billed duration when
    one could be recovered from the logs (otherwise ``None``)."""

    def __init__(self, message: str, duration_ms: int | None = None):
        super().__init__(message)
        self.duration_ms = duration_ms


class NotEnoughMemory(InvocationError):
    """The current memory configuration is too small to run the function."""

    def __init__(
        self,
        message: str = "Not enough memory configurations to explore.",
        duration_ms: int | None = None,
    ):
        super().__init__(message, duration_ms)


class FunctionTimeout(NotEnoughMemory):
    """The function exceeded its execution-time limit. See module note [*]."""

    def __init__(
        self,
        message: str = "Function timed out; the execution time limit was reached.",
        duration_ms: int | None = None,
    ):
        super().__init__(message, duration_ms)


class MaxInvocationAttemptsReached(InvocationError):
    """All retry attempts for an invocation were exhausted."""

    def __init__(
        self,
        message: str = "Maximum number of invocation attempts reached.",
        duration_ms: int | None = None,
    ):
        super().__init__(message, duration_ms)


# --------------------------------------------------------------------------- #
# Sampling / log errors
# --------------------------------------------------------------------------- #
class SamplingError(OptiServeError):
    """A profiling sample could not be produced."""

    def __init__(self, message: str = ""):
        super().__init__(message)


class LogParsingError(SamplingError):
    def __init__(self, message: str = "Error parsing log file."):
        super().__init__(message)


class NoMemoryLeft(SamplingError):
    def __init__(self, message: str = "No memory left in the memory space to explore with."):
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Optimization errors
# --------------------------------------------------------------------------- #
class OptimizationError(OptiServeError):
    """An optimization run failed."""

    def __init__(self, message: str = "Optimization failed."):
        super().__init__(message)


class UnfeasibleConstraint(OptimizationError):
    def __init__(self, message: str = "One or more provided constraints are unfeasible."):
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Cost / configuration errors
# --------------------------------------------------------------------------- #
class CostCalculationError(OptiServeError):
    def __init__(self, message: str = "Error in cost calculation."):
        super().__init__(message)


class FunctionConfigurationError(OptiServeError):
    def __init__(
        self,
        message: str = (
            "Error in function configuration. Make sure the provided function "
            "exists, and the configuration parameters are correct."
        ),
    ):
        super().__init__(message)
