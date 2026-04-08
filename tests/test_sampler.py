"""Sampler tests — focus on the memory-floor pruning that previously looped
forever (stale local binding) and its termination behavior. No AWS."""

import numpy as np
import pytest

from optiserve.exceptions import NoMemoryLeft, NotEnoughMemory
from optiserve.profiling.sampler import Sampler


class FakeExplorer:
    """Stand-in for Explorer: raises NotEnoughMemory below a feasibility floor."""

    def __init__(self, memory_space, feasible_from):
        self.function_name = "f"
        self.memory_spaces = {"None": np.array(memory_space, dtype=int)}
        self._feasible_from = feasible_from
        self.calls = []

    def explore_multi_threading(
        self, num_of_invocations, num_of_threads, memory_mb, model_name="None"
    ):
        self.calls.append(memory_mb)
        if memory_mb < self._feasible_from:
            raise NotEnoughMemory()
        return [100.0] * num_of_invocations

    def _explore(self):
        return 100.0


def _sampler(explorer):
    # cv_threshold high so the dynamic-sampling loop never runs in tests.
    return Sampler(explorer, profiling_iterations=3, cv_threshold=10.0)


def test_prunes_floor_then_succeeds_and_terminates():
    explorer = FakeExplorer([128, 256, 384, 512, 640], feasible_from=384)
    sampler = _sampler(explorer)
    sampler.exploration_init(model_name="None")

    # It must have raised the floor past the infeasible sizes and then explored
    # three feasible configs (first=384, plus the two seed points 512 & 640).
    assert explorer.memory_spaces["None"][0] == 384
    assert len(sampler.explorations["None"]) == 3 * 3  # 3 configs x 3 iterations
    assert 128 in explorer.calls and 384 in explorer.calls


def test_exhaustion_raises_no_memory_left():
    explorer = FakeExplorer([128, 256, 384], feasible_from=10_000)
    sampler = _sampler(explorer)
    with pytest.raises(NoMemoryLeft):
        sampler.exploration_init(model_name="None")
