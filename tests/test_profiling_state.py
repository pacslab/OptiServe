"""Tests for profiling checkpoints and resume.

These guard the property that makes a multi-hour, money-costing profiling run
survivable: an interrupted run continues from what it already paid for, and a
run whose *parameters* changed starts fresh instead of blending incomparable
measurements into one curve.
"""

import json

import numpy as np
import pytest

from optiserve.exceptions import NotEnoughMemory
from optiserve.observability import EventName
from optiserve.profiling.sample import Sample
from optiserve.profiling.sampler import Sampler
from optiserve.profiling.state import (
    JsonCheckpointStore,
    NullCheckpointStore,
    ProfilingState,
    run_id_for,
)


class FakeExplorer:
    """Explorer stand-in: infeasible below a floor, counts invocations."""

    def __init__(self, memory_space, feasible_from=0, payload="{}"):
        self.function_name = "f"
        self.payload = payload
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


def _sampler(explorer, store=None):
    return Sampler(explorer, profiling_iterations=3, cv_threshold=10.0, checkpoint_store=store)


# --------------------------------------------------------------------------- #
# run id
# --------------------------------------------------------------------------- #
def test_run_id_is_stable_for_identical_parameters():
    assert run_id_for("f", "m", payload="{}", n=4) == run_id_for("f", "m", payload="{}", n=4)


def test_run_id_changes_when_a_parameter_changes():
    # The whole point: a different payload is a different experiment, and its
    # samples must not be appended to the previous run's curve.
    assert run_id_for("f", "m", payload="{}") != run_id_for("f", "m", payload='{"a":1}')


def test_run_id_is_filesystem_safe():
    run_id = run_id_for("arn:aws:lambda:us-east-1:1:function/x", "model/v1", a=1)
    assert "/" not in run_id and ":" not in run_id.replace("__", "")


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def test_round_trip_preserves_samples_and_space(tmp_path):
    store = JsonCheckpointStore(tmp_path)
    state = ProfilingState(
        run_id="r",
        function_name="f",
        model_name="m",
        samples=[Sample(128, 900.5), Sample(256, 500.25)],
        memory_space=[128, 256, 384],
    )
    store.save(state)

    restored = store.load("r")
    assert [(s.memory_mb, s.duration_ms) for s in restored.samples] == [
        (128, 900.5),
        (256, 500.25),
    ]
    assert restored.memory_space == [128, 256, 384]
    assert store.list_runs() == ["r"]


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    store = JsonCheckpointStore(tmp_path)
    for i in range(5):
        store.save(
            ProfilingState(
                run_id="r", function_name="f", model_name="m", samples=[Sample(128, float(i))]
            )
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.json"]


def test_unreadable_checkpoint_is_ignored_not_raised(tmp_path):
    store = JsonCheckpointStore(tmp_path)
    store.path_for("r").write_text("{ this is not json")
    assert store.load("r") is None


def test_checkpoint_from_an_incompatible_version_is_ignored(tmp_path):
    store = JsonCheckpointStore(tmp_path)
    store.path_for("r").write_text(
        json.dumps({"version": 999, "run_id": "r", "function_name": "f", "model_name": "m"})
    )
    assert store.load("r") is None


def test_missing_and_deleted_checkpoints_return_none(tmp_path):
    store = JsonCheckpointStore(tmp_path)
    assert store.load("absent") is None
    store.save(ProfilingState(run_id="r", function_name="f", model_name="m"))
    store.delete("r")
    assert store.load("r") is None
    store.delete("r")  # deleting twice must not raise


# --------------------------------------------------------------------------- #
# sampler integration
# --------------------------------------------------------------------------- #
def test_sampler_persists_progress_and_resumes_without_reinvoking(tmp_path, recorded_events):
    store = JsonCheckpointStore(tmp_path)

    first = FakeExplorer([128, 256, 384, 512, 640])
    sampler = _sampler(first, store)
    sampler.exploration_init("None")
    invocations_first_run = len(first.calls)
    samples_first_run = len(sampler.explorations["None"])
    assert invocations_first_run > 0

    # A fresh process: same function, same parameters, checkpoint on disk.
    second = FakeExplorer([128, 256, 384, 512, 640])
    resumed = _sampler(second, store)
    resumed.exploration_init("None")

    assert second.calls == [], "resume must not repay for samples already collected"
    assert len(resumed.explorations["None"]) == samples_first_run
    assert recorded_events.count(EventName.CHECKPOINT_RESUMED) == 1
    assert recorded_events.count(EventName.CHECKPOINT_SAVED) >= 1


def test_resume_restores_the_pruned_memory_space(tmp_path):
    store = JsonCheckpointStore(tmp_path)
    first = FakeExplorer([128, 256, 384, 512, 640], feasible_from=384)
    _sampler(first, store).exploration_init("None")
    assert first.memory_spaces["None"][0] == 384  # floor was raised

    second = FakeExplorer([128, 256, 384, 512, 640], feasible_from=384)
    resumed = _sampler(second, store)
    resumed.exploration_init("None")
    # The pruning is part of the run's state: without it, a resumed run would
    # re-probe (and repay for) memory sizes already proven infeasible.
    assert list(second.memory_spaces["None"]) == [384, 512, 640]
    assert second.calls == []


def test_changed_parameters_start_a_new_run(tmp_path):
    store = JsonCheckpointStore(tmp_path)
    _sampler(FakeExplorer([128, 256, 384, 512, 640]), store).exploration_init("None")

    different = FakeExplorer([128, 256, 384, 512, 640], payload='{"other": 1}')
    fresh = Sampler(different, profiling_iterations=3, cv_threshold=10.0, checkpoint_store=store)
    fresh.exploration_init("None")
    assert different.calls, "a different payload is a different experiment"
    assert len(store.list_runs()) == 2


def test_null_store_is_the_default_and_never_resumes(tmp_path):
    explorer = FakeExplorer([128, 256, 384, 512, 640])
    sampler = _sampler(explorer)
    assert isinstance(sampler._checkpoints, NullCheckpointStore)
    sampler.exploration_init("None")
    assert explorer.calls  # no resume, full exploration


@pytest.mark.parametrize("noise_reduction", [True, False])
def test_noise_reduction_switch_controls_measurement_substitution(noise_reduction):
    explorer = FakeExplorer([128, 256, 384])
    sampler = Sampler(
        explorer,
        profiling_iterations=3,
        cv_threshold=0.0,  # force the substitution loop to want to run
        noise_reduction=noise_reduction,
    )
    measured = [100.0, 130.0, 90.0]
    result = sampler._explore_dynamically(list(measured))
    if noise_reduction:
        assert result != measured  # substitutes were accepted
    else:
        assert result == measured  # raw measurements preserved
