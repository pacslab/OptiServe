import multiprocessing as mp

from src.modeling.application_performance_modeling import ApplicationPerformanceModeling
from src.optimizer.application_optimizer import ApplicationOptimizer

def pct_work(app_g, filename, start_iterations, end_iterations, mem_list, model_list):
    app = ApplicationPerformanceModeling(graph=app_g.copy())
    optimizer = ApplicationOptimizer(app, mem_list=mem_list, model_list=model_list)
    optimizer.get_perf_cost_table(file=filename, start_iterations=start_iterations, end_iterations=end_iterations)