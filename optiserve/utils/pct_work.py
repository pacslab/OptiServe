"""Multiprocessing worker for parallel perf-cost-table generation.

A module-level function (picklable for the ``spawn`` start method) that a
notebook can hand to ``multiprocessing.Process`` to shard the exhaustive
configuration sweep across workers.
"""
from optiserve.evaluation.experiments import generate_perf_cost_table
from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.application_optimizer import ApplicationOptimizer


def pct_work(app_g, filename, start_iterations, end_iterations, mem_list=None, model_list=None):
    app = ApplicationPerformanceModeling(graph=app_g.copy())
    optimizer = ApplicationOptimizer(app, mem_list=mem_list, model_list=model_list)
    generate_perf_cost_table(
        optimizer, file=filename, start_iterations=start_iterations, end_iterations=end_iterations
    )
