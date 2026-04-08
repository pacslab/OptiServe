"""Tests for the workflow builder, validator, and the model->profile bridge."""

import pytest

from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.workflow import ModelVariant, WorkflowGraph


def test_builder_round_trips_to_the_golden_app():
    """A WorkflowGraph built from the notebook app must model identically."""
    mem = {1: 512, 2: 1024, 3: 2048, 4: 4096, 5: 128, 6: 256}
    rt = {1: 100, 2: 500, 3: 1000, 4: 210, 5: 130, 6: 100}
    wg = WorkflowGraph()
    for n in range(1, 7):
        wg.add_function(n, mem[n], rt[n])
    wg.add_edges(
        [
            ("Start", 1, 1.0),
            (1, 2, 0.2),
            (1, 3, 0.8),
            (1, 4, 1.0),
            (2, 5, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
            (5, 6, 0.7),
            (6, 6, 0.2),
            (6, "End", 0.8),
            (5, 1, 0.3),
        ]
    )
    wg.validate()

    app = ApplicationPerformanceModeling(wg.to_networkx(), delay_type="None")
    app.update_ne()
    app.get_simple_dag()
    assert round(float(app.get_avg_rt()), 6) == 1739.285714


def test_bridge_materializes_perf_profile():
    """add_ml_function evaluates each variant's model over the memory grid."""
    v1 = ModelVariant("v1", lambda m: 1000 - 0.1 * m, accuracy=0.80)
    v2 = ModelVariant("v2", lambda m: 800 - 0.05 * m, accuracy=0.92)
    wg = WorkflowGraph()
    wg.add_ml_function(1, [v1, v2], memory_grid=[128, 256, 512])
    node = wg.to_networkx().nodes[1]

    assert node["models_list"] == ["v1", "v2"]
    assert node["accuracy_list"] == [0.80, 0.92]
    assert node["perf_profile"][0][128] == pytest.approx(1000 - 0.1 * 128)
    assert node["perf_profile"][1][512] == pytest.approx(800 - 0.05 * 512)
    # initial config defaults to the smallest memory / first variant.
    assert node["mem"] == 128
    assert node["rt"] == pytest.approx(node["perf_profile"][0][128])


def test_validate_rejects_unreachable_node():
    wg = WorkflowGraph()
    wg.add_function(1, 512, 100)
    wg.add_function(2, 512, 100)  # never connected
    wg.add_edge("Start", 1, 1.0)
    wg.add_edge(1, "End", 1.0)
    with pytest.raises(ValueError):
        wg.validate()
