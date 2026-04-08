"""Reusable plotting helpers for model validation and optimization results.

matplotlib is imported lazily so the core library has no hard plotting
dependency at import time. Each function accepts an optional ``ax`` and returns
it, so plots compose into larger figures.
"""

from __future__ import annotations

from collections.abc import Sequence


def plot_real_vs_modeled_duration(
    memory: Sequence[float],
    real_durations: Sequence[float],
    modeled_durations: Sequence[float],
    *,
    ax=None,
    title: str = "Real vs. modeled duration",
):
    """Overlay measured and modeled execution time against memory."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(memory, real_durations, label="Measured", marker="o", linestyle="")
    ax.plot(memory, modeled_durations, label="Modeled")
    ax.set_xlabel("Memory (MB)")
    ax.set_ylabel("Duration (ms)")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_cost_time_tradeoff(
    memory: Sequence[float],
    costs: Sequence[float],
    durations: Sequence[float],
    *,
    ax=None,
    title: str = "Cost / time vs. memory",
):
    """Twin-axis plot of cost (left) and duration (right) against memory."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(memory, costs, color="tab:blue", label="Cost")
    ax.set_xlabel("Memory (MB)")
    ax.set_ylabel("Cost (USD)", color="tab:blue")
    twin = ax.twinx()
    twin.plot(memory, durations, color="tab:orange", label="Duration")
    twin.set_ylabel("Duration (ms)", color="tab:orange")
    ax.set_title(title)
    return ax


def plot_optimization_accuracy(
    accuracy_by_app: dict,
    *,
    ax=None,
    title: str = "Optimization accuracy by application",
):
    """Jittered scatter of optimization-accuracy percentages per application.

    ``accuracy_by_app`` maps an application label to an array of per-constraint
    accuracy percentages.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()
    for i, (_label, values) in enumerate(accuracy_by_app.items()):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        jitter = np.linspace(-0.15, 0.15, len(values)) if len(values) else np.zeros(0)
        ax.scatter([i] * len(values) + jitter, values, s=20, alpha=0.7)
    ax.set_xticks(range(len(accuracy_by_app)))
    ax.set_xticklabels(list(accuracy_by_app.keys()))
    ax.set_ylabel("Optimization accuracy (%)")
    ax.set_title(title)
    return ax
