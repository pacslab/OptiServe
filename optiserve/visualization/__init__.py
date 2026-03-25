"""Plotting helpers (matplotlib imported lazily inside each function)."""
from optiserve.visualization.plots import (
    plot_cost_time_tradeoff,
    plot_optimization_accuracy,
    plot_real_vs_modeled_duration,
)

__all__ = [
    "plot_real_vs_modeled_duration",
    "plot_cost_time_tradeoff",
    "plot_optimization_accuracy",
]
