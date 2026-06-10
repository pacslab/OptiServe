"""The evaluation cache must be an optimization, never a behaviour change.

The analytical model and the greedy optimizer are locked by golden baselines
because their numbers are published. Memoizing them is only acceptable if the
memoized path is *bit-identical* to the recomputed one — so that is what these
tests assert, across every strategy, both graph shapes, and the brute-force
ground-truth CSV.

They also pin the two hazards that make a cache of this kind subtly wrong:
a stale reduced graph, and per-reduction bookkeeping that silently accumulates.
"""

import contextlib
import io
import os
import tempfile
from pathlib import Path

import pytest
from optimizer_cases import _ACCURACY_FORMULA, _PRICING, _acyclic, _cyclic, _round

from optiserve.evaluation.experiments import generate_perf_cost_table
from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.modeling.cache import EvaluationCache, NullEvaluationCache
from optiserve.optimization.application_optimizer import ApplicationOptimizer


def _optimizer(builder, *, cached):
    app = ApplicationPerformanceModeling(builder(), delay_type="None", cache_evaluations=cached)
    app.cost_calculator.aws_pricing_units = dict(_PRICING)
    with contextlib.redirect_stdout(io.StringIO()):
        return app, ApplicationOptimizer(app)


def _run_all_strategies(builder, *, cached):
    app, optimizer = _optimizer(builder, cached=cached)
    budget = (optimizer.minimal_cost + optimizer.maximal_cost) / 2
    rt_constraint = (optimizer.minimal_avg_rt + optimizer.maximal_avg_rt) / 2
    with contextlib.redirect_stdout(io.StringIO()):
        results = {
            "BPBA:none": optimizer.BPBA(budget, 0.5, _ACCURACY_FORMULA, BCR=False),
            "BPBA:rtm": optimizer.BPBA(budget, 0.5, _ACCURACY_FORMULA, BCR=True, BCRtype="RT/M"),
            "BPBA:max": optimizer.BPBA(budget, 0.5, _ACCURACY_FORMULA, BCR=True, BCRtype="MAX"),
            "BCPA:none": optimizer.BCPA(rt_constraint, 0.5, _ACCURACY_FORMULA, BCR=False),
            "BCPA:max": optimizer.BCPA(
                rt_constraint, 0.5, _ACCURACY_FORMULA, BCR=True, BCRtype="MAX"
            ),
            "BAPB:ertc": optimizer.BAPB(
                rt_constraint, budget, _ACCURACY_FORMULA, BCR=True, BCRtype="ERT/C"
            ),
        }
    return app, {name: _round(result) for name, result in results.items()}


@pytest.mark.parametrize("builder,name", [(_acyclic, "acyclic"), (_cyclic, "cyclic")])
def test_every_strategy_is_identical_with_and_without_the_cache(builder, name):
    _, uncached = _run_all_strategies(builder, cached=False)
    app, cached = _run_all_strategies(builder, cached=True)

    assert uncached == cached, f"{name}: the cache changed an optimizer result"
    # And it actually did something, otherwise the assertion above is vacuous.
    stats = app.cache_stats()
    assert stats["rt"]["hits"] + stats["cost"]["hits"] > 0


def test_ground_truth_table_is_byte_identical_with_the_cache():
    outputs = {}
    for cached in (False, True):
        _, optimizer = _optimizer(_cyclic, cached=cached)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "pct.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                generate_perf_cost_table(optimizer, path)
            outputs[cached] = Path(path).read_text()

    assert outputs[False] == outputs[True]


def test_a_cache_hit_does_not_leave_a_stale_reduced_graph():
    """A hit skips the reduction, so `simple_dag` describes a previous
    configuration. `get_avg_rt()` must notice and rebuild rather than answer
    from it."""
    app, _ = _optimizer(_cyclic, cached=True)

    baseline = app.evaluate_avg_rt()

    # Move to a different configuration (miss), then back (hit).
    original = {n: app.workflow_graph.nodes[n]["rt"] for n in (1, 2, 3)}
    for node in (1, 2, 3):
        app.workflow_graph.nodes[node]["rt"] = 111.0
    app.update_rt()
    app.evaluate_avg_rt()

    for node, value in original.items():
        app.workflow_graph.nodes[node]["rt"] = value
    app.update_rt()

    assert app.evaluate_avg_rt() == baseline
    # The reduced graph was never rebuilt for this configuration — get_avg_rt
    # must do it itself rather than reading the 111.0 reduction.
    assert app.get_avg_rt() == baseline


def test_reduction_bookkeeping_does_not_accumulate():
    app, _ = _optimizer(_cyclic, cached=False)
    counts = []
    for _ in range(3):
        app.get_simple_dag()
        counts.append((len(app.approximations), app.p_node_num, app.b_node_num))
    assert len(set(counts)) == 1, (
        f"successive reductions of the same graph must produce the same bookkeeping, got {counts}"
    )


# --------------------------------------------------------------------------- #
# The cache primitive itself
# --------------------------------------------------------------------------- #
def test_cache_counts_hits_and_misses():
    cache = EvaluationCache(max_entries=4)
    assert cache.lookup("a") == (False, None)
    cache.put("a", 1)
    assert cache.lookup("a") == (True, 1)
    assert cache.stats.hits == 1 and cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


def test_cache_can_store_none_without_recomputing_forever():
    cache = EvaluationCache()
    cache.put("k", None)
    found, value = cache.lookup("k")
    assert found is True and value is None


def test_cache_evicts_least_recently_used():
    cache = EvaluationCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.lookup("a")  # refresh 'a'
    cache.put("c", 3)  # evicts 'b'
    assert cache.lookup("b")[0] is False
    assert cache.lookup("a")[0] is True
    assert cache.stats.evictions == 1


def test_cache_rejects_a_nonsense_bound():
    with pytest.raises(ValueError):
        EvaluationCache(max_entries=0)


def test_null_cache_always_misses():
    cache = NullEvaluationCache()
    cache.put("a", 1)
    assert cache.lookup("a") == (False, None)
    assert len(cache) == 0
