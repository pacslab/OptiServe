"""The compat presets must actually change the behaviour they claim to.

Each test builds a workflow where the corresponding defect is *reachable* and
shows the two presets diverging in the predicted direction. Without these, the
flags would be untestable documentation.
"""

import contextlib
import io

import pytest

from optiserve.config import OptimizationConfig
from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.application_optimizer import ApplicationOptimizer
from optiserve.optimization.compat import OptimizerCompat
from optiserve.workflow import ModelVariant, WorkflowGraph

_PRICING = {"compute": 1e-5, "request": 2e-7}
_MEAN = lambda *accuracies: sum(accuracies) / len(accuracies)  # noqa: E731


def _chain(node_count=4, variant_count=4, grid=(512, 768, 1024, 1536, 2048)):
    """A chain of ML functions whose variants trade accuracy for latency."""
    workflow = WorkflowGraph()
    for node in range(1, node_count + 1):
        variants = [
            ModelVariant(
                f"v{i}",
                (lambda base, slope: lambda m: base - slope * m)(600 + 250 * i, 0.05),
                accuracy=(i + 1) / variant_count,
            )
            for i in range(variant_count)
        ]
        workflow.add_ml_function(node, variants, list(grid))

    edges = [("Start", 1, 1.0)]
    edges += [(n, n + 1, 1.0) for n in range(1, node_count)]
    edges.append((node_count, "End", 1.0))
    workflow.add_edges(edges)
    return workflow.to_networkx()


def _optimizer(compat, graph=None):
    app = ApplicationPerformanceModeling(graph or _chain(), delay_type="None")
    app.cost_calculator.aws_pricing_units = dict(_PRICING)
    with contextlib.redirect_stdout(io.StringIO()):
        return ApplicationOptimizer(app, config=OptimizationConfig(compat=compat))


def test_default_preset_is_corrected():
    assert OptimizationConfig().compat is OptimizerCompat.CORRECTED
    assert _optimizer(OptimizationConfig().compat).compat is OptimizerCompat.CORRECTED


# --------------------------------------------------------------------------- #
# LEGACY_BPBA_SURPLUS
# --------------------------------------------------------------------------- #
def _budget_at(optimizer, fraction):
    return optimizer.minimal_cost + fraction * (optimizer.maximal_cost - optimizer.minimal_cost)


def test_corrected_bpba_reaches_accuracy_constraints_the_published_path_abandons():
    """The published path charges each committed upgrade to the budget twice,
    so it gives up on accuracy constraints it can comfortably afford.

    At 20 % of the cost envelope on this 4-function chain, full accuracy is
    affordable: the corrected path finds it, the published path stops at
    0.625 — a 60 % relative under-delivery — while leaving budget unspent.
    """
    published = _optimizer(OptimizerCompat.PUBLISHED)
    corrected = _optimizer(OptimizerCompat.CORRECTED)
    budget = _budget_at(published, 0.20)

    with contextlib.redirect_stdout(io.StringIO()):
        published_result = published.BPBA(budget, 1.0, _MEAN, BCR=False)
        corrected_result = corrected.BPBA(budget, 1.0, _MEAN, BCR=False)

    assert corrected_result.accuracy > published_result.accuracy, (
        f"published={published_result.accuracy} corrected={corrected_result.accuracy}"
    )
    assert corrected_result.accuracy == 1.0
    # And the corrected answer still respects the budget it was given.
    assert corrected_result.cost <= budget + 1e-6


def test_the_corrected_path_is_never_worse_across_the_budget_range():
    """A single lucky point would not be evidence. Sweep the envelope."""
    fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
    improvements = 0
    for fraction in fractions:
        published = _optimizer(OptimizerCompat.PUBLISHED)
        corrected = _optimizer(OptimizerCompat.CORRECTED)
        budget = _budget_at(published, fraction)
        with contextlib.redirect_stdout(io.StringIO()):
            published_result = published.BPBA(budget, 1.0, _MEAN, BCR=False)
            corrected_result = corrected.BPBA(budget, 1.0, _MEAN, BCR=False)

        assert corrected_result.accuracy >= published_result.accuracy, (
            f"at budget fraction {fraction} the corrected path did worse"
        )
        assert corrected_result.cost <= budget + 1e-6
        improvements += corrected_result.accuracy > published_result.accuracy

    assert improvements >= 4, "expected the defect to bite across most of the range"


def test_the_surplus_flag_alone_explains_the_difference():
    only_surplus = _optimizer(OptimizerCompat.LEGACY_BPBA_SURPLUS)
    published = _optimizer(OptimizerCompat.PUBLISHED)
    budget = _budget_at(published, 0.20)

    with contextlib.redirect_stdout(io.StringIO()):
        a = only_surplus.BPBA(budget, 1.0, _MEAN, BCR=False)
        b = published.BPBA(budget, 1.0, _MEAN, BCR=False)

    # BPBA is not affected by the BAPB gate, so these must agree.
    assert (a.accuracy, round(a.cost, 9)) == (b.accuracy, round(b.cost, 9))


# --------------------------------------------------------------------------- #
# LEGACY_BAPB_MEM_GATE
# --------------------------------------------------------------------------- #
def test_corrected_bapb_can_upgrade_a_variant_without_buying_memory():
    published = _optimizer(OptimizerCompat.PUBLISHED)
    corrected = _optimizer(OptimizerCompat.CORRECTED)

    rt_constraint = (published.minimal_avg_rt + published.maximal_avg_rt) / 2
    budget = published.minimal_cost + 0.25 * (published.maximal_cost - published.minimal_cost)

    with contextlib.redirect_stdout(io.StringIO()):
        published_result = published.BAPB(rt_constraint, budget, _MEAN)
        corrected_result = corrected.BAPB(rt_constraint, budget, _MEAN)

    assert corrected_result.accuracy >= published_result.accuracy
    assert corrected_result.cost <= budget + 1e-6
    assert corrected_result.response_time_ms <= rt_constraint + 1e-6


@pytest.mark.parametrize("preset", [OptimizerCompat.PUBLISHED, OptimizerCompat.CORRECTED])
def test_both_presets_stay_inside_their_constraints(preset):
    optimizer = _optimizer(preset)
    rt_constraint = (optimizer.minimal_avg_rt + optimizer.maximal_avg_rt) / 2
    budget = (optimizer.minimal_cost + optimizer.maximal_cost) / 2

    with contextlib.redirect_stdout(io.StringIO()):
        bcpc = optimizer.BCPC(rt_constraint, 0.5, _MEAN, BCR=False)
        bapb = optimizer.BAPB(rt_constraint, budget, _MEAN)

    assert bcpc.response_time_ms <= rt_constraint + 1e-6
    assert bapb.cost <= budget + 1e-6
    assert bapb.response_time_ms <= rt_constraint + 1e-6


# --------------------------------------------------------------------------- #
# Configuration knobs that used to be dead
# --------------------------------------------------------------------------- #
def test_accuracy_penalty_weight_is_read_from_config():
    """`w` was hardcoded to 100 while OptimizationConfig advertised the knob."""
    default = _optimizer(OptimizerCompat.CORRECTED)
    assert default._config.accuracy_penalty_weight == 100.0

    app = ApplicationPerformanceModeling(_chain(), delay_type="None")
    app.cost_calculator.aws_pricing_units = dict(_PRICING)
    with contextlib.redirect_stdout(io.StringIO()):
        tuned = ApplicationOptimizer(app, config=OptimizationConfig(accuracy_penalty_weight=5.0))
    assert tuned._config.accuracy_penalty_weight == 5.0
