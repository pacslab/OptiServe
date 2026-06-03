"""Golden-master: the relocated experiment drivers must produce byte-identical
CSVs to the original optimizer methods."""

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from optimizer_cases import _ACCURACY_FORMULA, _PRICING, _acyclic

from optiserve.config import OptimizationConfig
from optiserve.evaluation.experiments import generate_perf_cost_table, run_opt_curve
from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.application_optimizer import ApplicationOptimizer
from optiserve.optimization.compat import OptimizerCompat

BASELINE = json.loads((Path(__file__).parent / "evaluation_baseline.json").read_text())


def test_experiment_drivers_reproduce_csv_golden():
    app = ApplicationPerformanceModeling(_acyclic(), delay_type="None")
    app.cost_calculator.aws_pricing_units = dict(_PRICING)
    with contextlib.redirect_stdout(io.StringIO()):
        # The committed CSV/accuracy baselines are published numbers.
        opt = ApplicationOptimizer(app, config=OptimizationConfig(compat=OptimizerCompat.PUBLISHED))

    budget = list(np.linspace(opt.minimal_cost, opt.maximal_cost, 2))
    perf = list(np.linspace(opt.minimal_avg_rt, opt.maximal_avg_rt, 2))
    acc = [0.4, 0.6]

    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "G")
        with contextlib.redirect_stdout(io.StringIO()):
            run_opt_curve(opt, prefix, budget, perf, acc, _ACCURACY_FORMULA, BCRthreshold=0.2)
            generate_perf_cost_table(opt, os.path.join(d, "pct.csv"))
        produced = {
            name: Path(path).read_text()
            for name, path in (
                ("_BPBC.csv", prefix + "_BPBC.csv"),
                ("_BCPC.csv", prefix + "_BCPC.csv"),
                ("_BAPB.csv", prefix + "_BAPB.csv"),
                ("pct.csv", os.path.join(d, "pct.csv")),
            )
        }

    for name, expected in BASELINE.items():
        assert produced[name] == expected, f"{name} differs from the golden output"
