"""Optimizer-quality metrics: how close each greedy strategy's answer is to the
brute-force optimum.

Given an optimization-curve CSV (from :func:`run_opt_curve`) and the exhaustive
ground-truth table (from :func:`generate_perf_cost_table`), these compute a
per-constraint "optimization accuracy" percentage — 100 % means the greedy
answer matched the brute-force optimum. This logic previously lived only in the
evaluation notebooks.

``accuracy_formula`` is the end-to-end accuracy function (e.g. ``lambda a0, a1,
a2: 2*a1 + a2``); ``acc_value_columns`` are the ground-truth columns holding each
function's per-variant accuracy value (e.g. ``['f1_acc_value', 'f2_acc_value',
'f3_acc_value']``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BPBC_RT_COLS = ["BCR_disabled_RT", "BCR_RT/M_RT", "BCR_ERT/C_RT", "BCR_MAX_RT"]
_BCPC_COST_COLS = ["BCR_disabled_Cost", "BCR_M/RT_Cost", "BCR_C/ERT_Cost", "BCR_MAX_Cost"]


def _config_accuracy(ground_truth: pd.DataFrame, formula, acc_value_columns) -> pd.Series:
    return formula(*[ground_truth[col] for col in acc_value_columns])


def bpbc_optimization_accuracy(
    opt_curve: pd.DataFrame,
    ground_truth: pd.DataFrame,
    accuracy_formula,
    acc_value_columns: list[str],
) -> np.ndarray:
    """Best-Performance-under-Budget-and-aCcuracy: greedy min-RT vs brute-force
    min-RT among accuracy- and budget-feasible configurations."""
    df = opt_curve.copy()
    df["Best_Answer_RT"] = df[_BPBC_RT_COLS].min(axis=1)
    df = df[df["BCR_disabled_Acc_Score"] >= df["Accuracy_Constraint"]]

    accuracy = _config_accuracy(ground_truth, accuracy_formula, acc_value_columns)
    best_rt = []
    for budget, required in zip(df["Budget"], df["Accuracy_Constraint"], strict=True):
        feasible = ground_truth[(accuracy >= required) & (ground_truth["Cost"] <= budget)]
        best_rt.append(feasible["RT"].min() if not feasible.empty else np.nan)
    best_rt_array = np.array(best_rt, dtype=float)
    best_answer = df["Best_Answer_RT"].to_numpy(dtype=float)
    return 100 - ((best_answer - best_rt_array) / best_answer) * 100


def bcpc_optimization_accuracy(
    opt_curve: pd.DataFrame,
    ground_truth: pd.DataFrame,
    accuracy_formula,
    acc_value_columns: list[str],
) -> np.ndarray:
    """Best-Cost-under-Performance-and-aCcuracy: greedy min-cost vs brute-force
    min-cost among accuracy- and latency-feasible configurations."""
    df = opt_curve.copy()
    df["Best_Answer_Cost"] = df[_BCPC_COST_COLS].min(axis=1)
    df = df[df["BCR_disabled_Acc_Score"] >= df["Accuracy_Constraint"]]

    accuracy = _config_accuracy(ground_truth, accuracy_formula, acc_value_columns)
    best_cost = []
    for perf_constraint, required in zip(
        df["Performance_Constraint"], df["Accuracy_Constraint"], strict=True
    ):
        feasible = ground_truth[(accuracy >= required) & (ground_truth["RT"] <= perf_constraint)]
        best_cost.append(feasible["Cost"].min() if not feasible.empty else np.nan)
    best_cost_array = np.array(best_cost, dtype=float)
    best_answer = df["Best_Answer_Cost"].to_numpy(dtype=float)
    return 100 - ((best_answer - best_cost_array) / best_answer) * 100


def bapb_optimization_accuracy(
    opt_curve: pd.DataFrame,
    ground_truth: pd.DataFrame,
    accuracy_formula,
    acc_value_columns: list[str],
) -> np.ndarray:
    """Best-Accuracy-under-Performance-and-Budget: greedy max-accuracy vs
    brute-force max-accuracy among latency- and budget-feasible configurations."""
    df = opt_curve.copy()
    df["Best_Answer_Accuracy"] = df[["BCR_disabled_Acc_Score"]].max(axis=1)
    df = df[
        (df["BCR_disabled_RT"] <= df["Performance_Constraint"])
        & (df["BCR_disabled_Cost"] <= df["Budget"])
    ]

    accuracy = _config_accuracy(ground_truth, accuracy_formula, acc_value_columns)
    best_accuracy = []
    for perf_constraint, budget in zip(df["Performance_Constraint"], df["Budget"], strict=True):
        feasible = ground_truth[
            (ground_truth["Cost"] <= budget) & (ground_truth["RT"] <= perf_constraint)
        ]
        best_accuracy.append(accuracy[feasible.index].max() if not feasible.empty else np.nan)
    best_accuracy_array = np.array(best_accuracy, dtype=float)
    best_answer = df["Best_Answer_Accuracy"].to_numpy(dtype=float)
    return 100 - ((best_accuracy_array - best_answer) / best_answer) * 100
