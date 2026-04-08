"""Offline example: build a small ML workflow, model it, and optimize it.

Runs without AWS by using synthetic performance models and injected pricing.

    python examples/optimize_workflow.py
"""

from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.application_optimizer import ApplicationOptimizer
from optiserve.workflow import ModelVariant, WorkflowGraph


def perf(base_ms, slope):
    """A toy latency-vs-memory curve: latency decreases with memory."""
    return lambda memory_mb: base_ms - slope * memory_mb


def main():
    memory_grid = [128, 256, 512, 1024, 2048]

    # Function 1: three model variants trading accuracy for latency.
    f1_variants = [
        ModelVariant("small", perf(900, 0.10), accuracy=0.70),
        ModelVariant("medium", perf(1100, 0.10), accuracy=0.85),
        ModelVariant("large", perf(1400, 0.10), accuracy=0.95),
    ]
    # Function 2: two variants.
    f2_variants = [
        ModelVariant("fast", perf(600, 0.05), accuracy=0.60),
        ModelVariant("accurate", perf(950, 0.05), accuracy=0.92),
    ]

    workflow = (
        WorkflowGraph()
        .add_ml_function(1, f1_variants, memory_grid)
        .add_ml_function(2, f2_variants, memory_grid)
        .add_edges([("Start", 1, 1.0), (1, 2, 1.0), (2, "End", 1.0)])
        .validate()
    )

    app = ApplicationPerformanceModeling(workflow.to_networkx(), delay_type="SFN")
    # Inject fixed Lambda unit prices so the example needs no AWS credentials.
    app.cost_calculator.aws_pricing_units = {"compute": 1.6667e-5, "request": 2e-7}

    optimizer = ApplicationOptimizer(app)
    end_to_end_accuracy = lambda a, b: (a + b) / 2  # noqa: E731

    # Minimize end-to-end latency under a budget and an accuracy floor (BPBC).
    result = optimizer.BPBC(
        budget=optimizer.maximal_cost,
        accuracy_constraint=0.75,
        accuracy_formula=end_to_end_accuracy,
    )

    print("=== BPBC: minimize latency under budget + accuracy ===")
    print(f"  response time : {result.response_time_ms:.1f} ms")
    print(f"  cost          : {result.cost:.4f}")
    print(f"  accuracy      : {result.accuracy:.3f}")
    print(f"  memory config : {result.memory_config}")
    print(f"  variant config: {result.model_config}")


if __name__ == "__main__":
    main()
