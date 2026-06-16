"""Tests for the benchmark-application builders and the model cache lookup.

The ``.mdl`` cache is gitignored and can only be regenerated against live AWS,
so tests that need it skip cleanly. The *lookup* logic is tested unconditionally,
because "works only when you happen to run from the repo root" was the actual
defect.
"""

import contextlib
import io
import os

import pytest

from optiserve.datasets import MODELS_DIR_ENV, build_app3, resolve_models_dir


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    directory = tmp_path / "modeled_functions"
    directory.mkdir()
    monkeypatch.setenv(MODELS_DIR_ENV, str(directory))
    return directory


def test_explicit_directory_wins(tmp_path, models_dir):
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert resolve_models_dir(other) == other


def test_an_explicit_missing_directory_is_an_error_not_a_fallback(tmp_path, models_dir):
    """Falling back would silently load a *different* set of fitted curves than
    the caller asked for."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_models_dir(tmp_path / "absent")


def test_environment_variable_is_honoured(models_dir):
    assert resolve_models_dir() == models_dir


def test_resolution_is_independent_of_the_working_directory(models_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_models_dir() == models_dir


def _write_model(directory, name):
    from optiserve.modeling.parametric import ParamFunction

    curve = ParamFunction()
    curve.params = [200.0, 3000.0, 600.0]
    curve.save(directory / f"{name}.mdl")


def test_a_missing_model_names_what_is_available(models_dir):
    _write_model(models_dir, "f1")
    with pytest.raises(FileNotFoundError) as excinfo:
        build_app3(models_dir)
    message = str(excinfo.value)
    assert "resnet_resnet-18" in message
    assert "f1" in message  # tells the user what the directory does contain


def test_a_corrupt_model_file_names_the_file(models_dir):
    """A truncated .mdl used to surface as a bare pickle KeyError with no
    mention of the path — for a gitignored cache, close to undebuggable."""
    (models_dir / "f1.mdl").write_bytes(b"not really a model")
    with pytest.raises(ValueError) as excinfo:
        build_app3(models_dir)
    assert "f1.mdl" in str(excinfo.value)
    assert "regenerate" in str(excinfo.value)


@pytest.mark.skipif(
    not os.path.isdir("modeled_functions") or not os.path.exists("modeled_functions/f1.mdl"),
    reason="the .mdl cache is gitignored and requires live-AWS profiling to regenerate",
)
def test_app3_builds_from_the_cached_models():
    workflow, formula = build_app3()
    graph = workflow.to_networkx()

    assert {"Start", "End", 1, 2, 3} == set(graph.nodes)
    # resnet has four variants, yolo four, f1 one.
    assert len(graph.nodes[2]["models_list"]) == 4
    assert len(graph.nodes[3]["models_list"]) == 4
    assert len(graph.nodes[1]["models_list"]) == 1
    # The weighted end-to-end formula: 2*resnet + yolo.
    assert formula(0.0, 0.5, 0.25) == pytest.approx(1.25)
    # Every variant's profile is materialised over its grid and is positive.
    for node in (1, 2, 3):
        for profile in graph.nodes[node]["perf_profile"]:
            assert profile
            assert all(latency > 0 for latency in profile.values())


@pytest.mark.skipif(
    not os.path.exists("modeled_functions/f1.mdl"),
    reason="the .mdl cache is gitignored and requires live-AWS profiling to regenerate",
)
def test_app3_is_optimizable_end_to_end():
    from optiserve.modeling.application_model import ApplicationPerformanceModeling
    from optiserve.optimization.application_optimizer import ApplicationOptimizer

    workflow, formula = build_app3()
    app = ApplicationPerformanceModeling(
        workflow.to_networkx(), delay_type="SFN", cache_evaluations=True
    )
    app.cost_calculator.aws_pricing_units = {"compute": 1.6667e-5, "request": 2e-7}

    with contextlib.redirect_stdout(io.StringIO()):
        optimizer = ApplicationOptimizer(app)
        result = optimizer.BPBC(optimizer.maximal_cost, 0.0, formula)

    assert result.response_time_ms > 0
    assert optimizer.minimal_cost <= result.cost <= optimizer.maximal_cost + 1e-6
