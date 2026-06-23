"""CLI tests.

The container image's ENTRYPOINT is this CLI, so these are also the tests that
keep the shipped image working. The `profile` guard gets particular attention:
it is the only command that mutates and invokes a live, billable function.
"""

import json
import math

import pytest

from optiserve.cli import build_accuracy_formula, build_parser, load_workflow, main

WORKFLOW = {
    "delay_type": "None",
    "pricing": {"compute": 1e-5, "request": 2e-7},
    "accuracy_formula": "mean",
    "functions": [
        {
            "id": "a",
            "memory_grid": [512, 1024, 2048],
            "variants": [
                {
                    "name": "small",
                    "accuracy": 0.70,
                    "model": {"kind": "linear", "intercept": 900, "slope": 0.10},
                },
                {
                    "name": "large",
                    "accuracy": 0.95,
                    "model": {"kind": "linear", "intercept": 1400, "slope": 0.10},
                },
            ],
        },
        {
            "id": "b",
            "memory_grid": {"start": 512, "stop": 2049, "step": 512},
            "variants": [
                {
                    "name": "fast",
                    "accuracy": 0.60,
                    "model": {"kind": "exponential", "a0": 200, "a1": 3000, "a2": 600},
                },
            ],
        },
    ],
    "edges": [["Start", "a", 1.0], ["a", "b", 1.0], ["b", "End", 1.0]],
}


@pytest.fixture
def workflow_file(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(WORKFLOW))
    return path


# --------------------------------------------------------------------------- #
# Accuracy formulas — data, never code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spec,args,expected",
    [
        ("mean", (0.4, 0.6), 0.5),
        ("min", (0.4, 0.6), 0.4),
        ("sum", (0.4, 0.6), 1.0),
        ("product", (0.5, 0.5), 0.25),
        ({"weighted": [2, 1]}, (0.5, 0.5), 1.5),
        (None, (0.4, 0.6), 0.5),
    ],
)
def test_named_accuracy_formulas(spec, args, expected):
    assert build_accuracy_formula(spec)(*args) == pytest.approx(expected)


def test_weighted_formula_rejects_an_arity_mismatch():
    formula = build_accuracy_formula({"weighted": [1, 1, 1]})
    with pytest.raises(ValueError, match="expects 3"):
        formula(0.5, 0.5)


@pytest.mark.parametrize("spec", ["__import__('os').system('id')", "lambda a: a", 42, {}])
def test_arbitrary_expressions_are_rejected_not_evaluated(spec):
    """A workflow spec is data — often produced by another tool. Reading one
    must never be equivalent to executing it."""
    with pytest.raises(ValueError):
        build_accuracy_formula(spec)


# --------------------------------------------------------------------------- #
# Spec loading
# --------------------------------------------------------------------------- #
def test_spec_builds_a_valid_workflow():
    graph = load_workflow(WORKFLOW).to_networkx()
    assert set(graph.nodes) == {"Start", "End", "a", "b"}
    assert graph.nodes["a"]["models_list"] == ["small", "large"]
    assert graph.nodes["a"]["accuracy_list"] == [0.70, 0.95]
    # Both memory-grid spellings materialise a profile.
    assert sorted(graph.nodes["a"]["perf_profile"][0]) == [512, 1024, 2048]
    assert sorted(graph.nodes["b"]["perf_profile"][0]) == [512, 1024, 1536, 2048]


def test_linear_and_exponential_models_evaluate():
    graph = load_workflow(WORKFLOW).to_networkx()
    assert graph.nodes["a"]["perf_profile"][0][1024] == pytest.approx(900 - 0.10 * 1024)
    expected = 200 + 3000 * math.exp(-1024 / 600)
    assert graph.nodes["b"]["perf_profile"][0][1024] == pytest.approx(expected)


def test_an_unknown_model_kind_is_rejected():
    spec = json.loads(json.dumps(WORKFLOW))
    spec["functions"][0]["variants"][0]["model"] = {"kind": "magic"}
    with pytest.raises(ValueError, match="unknown performance model kind"):
        load_workflow(spec)


def test_an_invalid_graph_is_rejected_at_load():
    spec = json.loads(json.dumps(WORKFLOW))
    spec["edges"] = [["Start", "a", 1.0]]  # 'a' cannot reach End
    with pytest.raises(ValueError):
        load_workflow(spec)


# --------------------------------------------------------------------------- #
# optimize
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["bpbc", "bcpc", "bapb"])
def test_optimize_runs_offline_for_every_strategy(workflow_file, tmp_path, strategy):
    output = tmp_path / "result.json"
    exit_code = main(
        [
            "--quiet",
            "optimize",
            "--workflow",
            str(workflow_file),
            "--strategy",
            strategy,
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0

    payload = json.loads(output.read_text())
    assert payload["strategy"] == strategy
    assert payload["response_time_ms"] > 0
    assert payload["cost"] > 0
    assert set(payload["memory_config"]) == {"a", "b"}
    assert "cache_stats" in payload


def test_optimize_respects_an_accuracy_floor(workflow_file, tmp_path):
    output = tmp_path / "result.json"
    main(
        [
            "--quiet",
            "optimize",
            "--workflow",
            str(workflow_file),
            "--strategy",
            "bpbc",
            "--accuracy",
            "0.75",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())
    # 'a' must be on the accurate variant for mean(0.95, 0.60) = 0.775 >= 0.75.
    assert payload["model_config"]["a"] == 1


def test_optimize_writes_json_to_stdout_by_default(workflow_file, capsys):
    assert main(["--quiet", "optimize", "--workflow", str(workflow_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "bpbc"


def test_cache_can_be_disabled(workflow_file, tmp_path):
    output = tmp_path / "r.json"
    main(
        [
            "--quiet",
            "optimize",
            "--workflow",
            str(workflow_file),
            "--no-cache",
            "--output",
            str(output),
        ]
    )
    stats = json.loads(output.read_text())["cache_stats"]
    assert stats["rt"]["hits"] == 0 and stats["rt"]["misses"] == 0


def test_a_missing_workflow_file_is_a_clean_error(tmp_path):
    assert main(["--quiet", "optimize", "--workflow", str(tmp_path / "nope.json")]) == 1


# --------------------------------------------------------------------------- #
# profile — the only command that touches a live function
# --------------------------------------------------------------------------- #
def test_profile_refuses_to_mutate_a_live_function_without_confirmation(caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="optiserve"):
        exit_code = main(["profile", "--function", "production-inference"])

    assert exit_code == 2
    message = " ".join(record.getMessage() for record in caplog.records)
    assert "--yes" in message
    assert "production-inference" in message


def test_profile_does_not_call_aws_when_it_refuses(monkeypatch):
    """The guard must come before any client construction, or 'refusing' still
    costs an API call and a credential lookup."""
    import botocore.client

    def explode(*args, **kwargs):
        raise AssertionError("the CLI touched AWS before the confirmation guard")

    monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", explode)
    assert main(["--quiet", "profile", "--function", "f"]) == 2


# --------------------------------------------------------------------------- #
# version / parser
# --------------------------------------------------------------------------- #
def test_version_reports_the_package_and_runtime(capsys):
    assert main(["--quiet", "version"]) == 0
    payload = json.loads(capsys.readouterr().out)
    from optiserve import __version__

    assert payload["optiserve"] == __version__
    assert "python" in payload and "platform" in payload


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
