#!/usr/bin/env python3
"""Regenerate the golden-master baselines.

Run this **only** when you have deliberately changed modeling or optimizer
behaviour, and review the resulting diff line by line — these files are the
contract that published numbers have not silently moved.

    python tests/golden/regenerate.py --list
    python tests/golden/regenerate.py optimizer_corrected
    python tests/golden/regenerate.py --all

The ``optimizer`` / ``evaluation`` / ``modeling`` / ``app_modeling`` baselines
pin *published* results. Regenerating one of those means you are changing what
the paper reported; the diff belongs in the same commit as the reason.
``optimizer_corrected`` pins the CORRECTED compat preset and is the one you
regenerate when fixing an optimizer defect.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))


def _app_modeling() -> dict:
    from _app_modeling_case import compute

    from optiserve.modeling.application_model import ApplicationPerformanceModeling

    return compute(ApplicationPerformanceModeling)


def _modeling() -> dict:
    from modeling_cases import compute_all

    from optiserve.modeling.application_model import ApplicationPerformanceModeling

    return compute_all(ApplicationPerformanceModeling)


def _optimizer() -> dict:
    from optimizer_cases import compute_all

    from optiserve.optimization.compat import OptimizerCompat

    return compute_all(compat=OptimizerCompat.PUBLISHED)


def _optimizer_corrected() -> dict:
    from optimizer_cases import compute_all

    from optiserve.optimization.compat import OptimizerCompat

    return compute_all(compat=OptimizerCompat.CORRECTED)


def _evaluation() -> dict:
    import numpy as np
    from optimizer_cases import _ACCURACY_FORMULA, _PRICING, _acyclic

    from optiserve.config import OptimizationConfig
    from optiserve.evaluation.experiments import generate_perf_cost_table, run_opt_curve
    from optiserve.modeling.application_model import ApplicationPerformanceModeling
    from optiserve.optimization.application_optimizer import ApplicationOptimizer
    from optiserve.optimization.compat import OptimizerCompat

    app = ApplicationPerformanceModeling(_acyclic(), delay_type="None")
    app.cost_calculator.aws_pricing_units = dict(_PRICING)
    with contextlib.redirect_stdout(io.StringIO()):
        optimizer = ApplicationOptimizer(
            app, config=OptimizationConfig(compat=OptimizerCompat.PUBLISHED)
        )

    budget = list(np.linspace(optimizer.minimal_cost, optimizer.maximal_cost, 2))
    performance = list(np.linspace(optimizer.minimal_avg_rt, optimizer.maximal_avg_rt, 2))
    accuracy = [0.4, 0.6]

    with tempfile.TemporaryDirectory() as directory:
        prefix = os.path.join(directory, "G")
        with contextlib.redirect_stdout(io.StringIO()):
            run_opt_curve(
                optimizer,
                prefix,
                budget,
                performance,
                accuracy,
                _ACCURACY_FORMULA,
                BCRthreshold=0.2,
            )
            generate_perf_cost_table(optimizer, os.path.join(directory, "pct.csv"))
        return {
            "_BPBC.csv": open(prefix + "_BPBC.csv").read(),
            "_BCPC.csv": open(prefix + "_BCPC.csv").read(),
            "_BAPB.csv": open(prefix + "_BAPB.csv").read(),
            "pct.csv": open(os.path.join(directory, "pct.csv")).read(),
        }


BASELINES = {
    "app_modeling": ("app_modeling_baseline.json", _app_modeling),
    "modeling": ("modeling_baseline.json", _modeling),
    "optimizer": ("optimizer_baseline.json", _optimizer),
    "optimizer_corrected": ("optimizer_baseline_corrected.json", _optimizer_corrected),
    "evaluation": ("evaluation_baseline.json", _evaluation),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # No `choices=`: with nargs="*" argparse validates the empty default
    # against the choice list and rejects a bare `--list`/`--all` invocation.
    parser.add_argument(
        "names", nargs="*", metavar="NAME", help=f"one or more of: {', '.join(BASELINES)}"
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, (filename, _) in BASELINES.items():
            print(f"{name:22} -> tests/golden/{filename}")
        return 0

    names = list(BASELINES) if args.all else args.names
    if not names:
        parser.error("name one or more baselines, or pass --all / --list")
    unknown = [name for name in names if name not in BASELINES]
    if unknown:
        parser.error(
            f"unknown baseline(s) {', '.join(unknown)}; choose from {', '.join(BASELINES)}"
        )

    changed = False
    for name in names:
        filename, compute = BASELINES[name]
        path = HERE / filename

        previous = None
        if path.exists():
            try:
                previous = json.loads(path.read_text())
            except json.JSONDecodeError:
                previous = None

        produced = compute()
        path.write_text(json.dumps(produced, indent=2, sort_keys=True) + "\n")

        # Compare parsed content, not bytes: a pure reformat is not a change in
        # behaviour, and reporting it as one would train reviewers to wave the
        # diff through.
        if previous is None:
            status, changed = "NEW", True
        elif previous == produced:
            status = "unchanged"
        else:
            status, changed = "CHANGED", True
        print(f"{status:10} {path.relative_to(HERE.parent.parent)}")

    if changed:
        print(
            "\nAt least one baseline moved. Review the diff before committing: "
            "these files are the record that published results did not change "
            "by accident.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
