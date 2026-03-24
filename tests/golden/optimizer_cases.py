"""Golden-master battery for the application optimizer.

Builds small ML workflows (acyclic and cyclic) with injected (non-AWS) pricing
and runs every greedy strategy / BCR variant, capturing the returned
(rt, cost, accuracy, mem_config, model_config, iterations) so the Stage-7
optimizer refactor can be proven behavior-preserving.
"""
from __future__ import annotations

import contextlib
import io

from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.application_optimizer import ApplicationOptimizer
from optiserve.workflow import ModelVariant, WorkflowGraph

_GRID = [128, 256, 512, 1024]
_PRICING = {"compute": 1e-5, "request": 2e-7}
_ACCURACY_FORMULA = lambda *accs: sum(accs) / len(accs)  # noqa: E731 (mean, any arity)


def _variants(base):
    return [
        ModelVariant(f"v{i}", (lambda b, s: (lambda m: b - s * m))(base + 200 * i, 0.05 * (i + 1)))
        for i in range(3)
    ]


def _acyclic():
    wg = WorkflowGraph()
    wg.add_ml_function(1, _variants(800), _GRID)
    wg.add_ml_function(2, _variants(600), _GRID)
    wg.add_edges([("Start", 1, 1.0), (1, 2, 1.0), (2, "End", 1.0)])
    return wg.to_networkx()


def _cyclic():
    wg = WorkflowGraph()
    wg.add_ml_function(1, _variants(800), _GRID)
    wg.add_ml_function(2, _variants(700), _GRID)
    wg.add_ml_function(3, _variants(600), _GRID)
    wg.add_edges([
        ("Start", 1, 1.0), (1, 2, 1.0), (2, 3, 0.7), (2, 1, 0.3),
        (3, 3, 0.2), (3, "End", 0.8),
    ])
    return wg.to_networkx()


def _optimizer(graph):
    app = ApplicationPerformanceModeling(graph, delay_type="None")
    app.cost_calculator.aws_pricing_units = dict(_PRICING)
    with contextlib.redirect_stdout(io.StringIO()):
        return ApplicationOptimizer(app, mem_list={}, model_list={})


def _round(result):
    rt, cost, acc, mem_cfg, model_cfg, iters = result
    return {
        "rt": round(float(rt), 4),
        "cost": round(float(cost), 4),
        "accuracy": round(float(acc), 4),
        "mem": {str(k): int(v) for k, v in mem_cfg.items()},
        "model": {str(k): int(v) for k, v in model_cfg.items()},
        "iterations": int(iters),
    }


def compute_all():
    out = {}
    for graph_name, builder in (("acyclic", _acyclic), ("cyclic", _cyclic)):
        opt = _optimizer(builder())
        budget = (opt.minimal_cost + opt.maximal_cost) / 2
        rt_c = (opt.minimal_avg_rt + opt.maximal_avg_rt) / 2
        acc_c = 0.5
        with contextlib.redirect_stdout(io.StringIO()):
            runs = {
                "BPBA:none": opt.BPBA(budget, acc_c, _ACCURACY_FORMULA, BCR=False),
                "BPBA:rtm": opt.BPBA(budget, acc_c, _ACCURACY_FORMULA, BCR=True, BCRtype="RT/M"),
                "BPBA:max": opt.BPBA(budget, acc_c, _ACCURACY_FORMULA, BCR=True, BCRtype="MAX"),
                "BCPA:none": opt.BCPA(rt_c, acc_c, _ACCURACY_FORMULA, BCR=False),
                "BCPA:max": opt.BCPA(rt_c, acc_c, _ACCURACY_FORMULA, BCR=True, BCRtype="MAX"),
                "BAPB:ertc": opt.BAPB(rt_c, budget, _ACCURACY_FORMULA, BCR=True, BCRtype="ERT/C"),
            }
        for key, result in runs.items():
            out[f"{graph_name}:{key}"] = _round(result)
    return out
