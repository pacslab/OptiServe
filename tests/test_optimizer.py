"""Tests for the optimizer's typed result and pluggable accuracy model.

(Full behavior is locked by tests/golden/test_optimizer_golden.py.)
"""

import contextlib
import io

from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.accuracy import AccuracyModel
from optiserve.optimization.application_optimizer import ApplicationOptimizer
from optiserve.optimization.result import OptimizationResult
from optiserve.workflow import ModelVariant, WorkflowGraph


def _build_optimizer(accuracies):
    variants = [
        ModelVariant("a", lambda m: 500 - 0.1 * m, accuracy=accuracies[0]),
        ModelVariant("b", lambda m: 400 - 0.1 * m, accuracy=accuracies[1]),
    ]
    wg = WorkflowGraph()
    wg.add_ml_function(1, variants, [128, 256, 512])
    wg.add_edges([("Start", 1, 1.0), (1, "End", 1.0)])
    app = ApplicationPerformanceModeling(wg.to_networkx(), delay_type="None")
    app.cost_calculator.aws_pricing_units = {"compute": 1e-5, "request": 2e-7}
    with contextlib.redirect_stdout(io.StringIO()):
        return ApplicationOptimizer(app)


def test_measured_accuracy_is_used_when_provided():
    opt = _build_optimizer([0.95, 0.40])  # inverts the rank order
    assert opt.model_accuracy_list[1] == [0.95, 0.40]


def test_accuracy_falls_back_to_rank_when_absent():
    opt = _build_optimizer([None, None])
    assert opt.model_accuracy_list[1] == [0.5, 1.0]  # i/N for i in 1..2


def test_accuracy_model_end_to_end_formula():
    model = AccuracyModel({"a": [0.5, 1.0], "b": [0.25, 0.75]})
    # mean of variant 1 of 'a' (1.0) and variant 0 of 'b' (0.25)
    value = model.end_to_end({"a": 1, "b": 0}, lambda x, y: (x + y) / 2)
    assert value == 0.625


def test_result_is_tuple_compatible():
    opt = _build_optimizer([None, None])
    with contextlib.redirect_stdout(io.StringIO()):
        result = opt.BPBC(opt.maximal_cost, 0.5, lambda a: a, BCR=False)
    assert isinstance(result, OptimizationResult)
    rt, cost, acc, mem, model, iters = result  # unpacks like the old 6-tuple
    assert result.response_time_ms == rt and result.memory_config == mem
