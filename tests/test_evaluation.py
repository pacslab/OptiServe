"""End-to-end evaluation pipeline test: ground truth -> opt curve ->
optimization-accuracy metrics, on a small offline graph."""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))

from optimizer_cases import _ACCURACY_FORMULA, _PRICING, _acyclic

from optiserve.config import OptimizationConfig
from optiserve.evaluation import (
    bapb_optimization_accuracy,
    bcpc_optimization_accuracy,
    bpbc_optimization_accuracy,
    generate_perf_cost_table,
    run_opt_curve,
)
from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.application_optimizer import ApplicationOptimizer
from optiserve.optimization.compat import OptimizerCompat


def test_evaluation_pipeline_produces_sane_accuracy():
    app = ApplicationPerformanceModeling(_acyclic(), delay_type="None")
    app.cost_calculator.aws_pricing_units = dict(_PRICING)
    with contextlib.redirect_stdout(io.StringIO()):
        # The committed CSV/accuracy baselines are published numbers.
        opt = ApplicationOptimizer(app, config=OptimizationConfig(compat=OptimizerCompat.PUBLISHED))

    budget = list(np.linspace(opt.minimal_cost, opt.maximal_cost, 3))
    perf = list(np.linspace(opt.minimal_avg_rt, opt.maximal_avg_rt, 3))
    acc = [0.4, 0.5, 0.6]
    acc_cols = ["f1_acc_value", "f2_acc_value"]

    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "G")
        gt_path = os.path.join(d, "pct.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            run_opt_curve(opt, prefix, budget, perf, acc, _ACCURACY_FORMULA, BCRthreshold=0.2)
            generate_perf_cost_table(opt, gt_path)
        ground_truth = pd.read_csv(gt_path)
        bpbc = pd.read_csv(prefix + "_BPBC.csv")
        bcpc = pd.read_csv(prefix + "_BCPC.csv")
        bapb = pd.read_csv(prefix + "_BAPB.csv")

    a_bpbc = bpbc_optimization_accuracy(bpbc, ground_truth, _ACCURACY_FORMULA, acc_cols)
    a_bcpc = bcpc_optimization_accuracy(bcpc, ground_truth, _ACCURACY_FORMULA, acc_cols)
    a_bapb = bapb_optimization_accuracy(bapb, ground_truth, _ACCURACY_FORMULA, acc_cols)

    # The greedy answer should be at (or close to) the brute-force optimum, so
    # accuracy percentages land around 100% (allow slack for the heuristic).
    for arr in (a_bpbc, a_bcpc, a_bapb):
        finite = arr[np.isfinite(arr)]
        assert len(finite) > 0
        assert np.all(finite <= 100.0 + 1e-6)
        assert np.all(finite >= 80.0)
