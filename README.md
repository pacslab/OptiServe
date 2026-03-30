# OptiServe

**OptiServe** is a system for **jointly optimizing cost, latency, and accuracy** in serverless applications with machine learning workloads. It supports complex application workflows composed of multiple functions, each with different performance and accuracy characteristics, and finds configurations that satisfy application-level constraints.

<p align="center">
  <img src="./docs/OptiServe.png" alt="OptiServe Logo" width="280"/>
</p>

## ✨ Overview

Serverless computing simplifies deployment, but makes it harder to tune performance. OptiServe tackles this challenge by:

- Modeling latency and cost for both ML and non-ML functions.
- Capturing the impact of model accuracy on end-to-end workflow behavior.
- Solving tri-objective optimization problems using graph-based heuristics.
- Automatically identifying optimal memory and model choices for each function in a workflow.

## 🔍 Features

- **Tri-objective optimization** of serverless workflows (cost, latency, accuracy).
- **Performance modeling** through lightweight profiling.
- **Search space reduction** using critical paths and benefit-cost heuristics.
- **Support for workflows with branching, parallelism, cycles, and self-loops.**
- **Offline-testable core** — the modeling and optimization layers have no AWS or
  filesystem dependency; only live profiling touches AWS.

## 🚀 Quickstart

Build a workflow, model it, and optimize it — no AWS needed (synthetic models and
injected pricing):

```python
from optiserve import WorkflowGraph, ModelVariant, ApplicationPerformanceModeling, ApplicationOptimizer

grid = [128, 256, 512, 1024, 2048]
f1 = [ModelVariant("small", lambda m: 900 - 0.1*m, accuracy=0.70),
      ModelVariant("large", lambda m: 1400 - 0.1*m, accuracy=0.95)]
f2 = [ModelVariant("fast", lambda m: 600 - 0.05*m, accuracy=0.60),
      ModelVariant("accurate", lambda m: 950 - 0.05*m, accuracy=0.92)]

workflow = (WorkflowGraph()
    .add_ml_function(1, f1, grid)
    .add_ml_function(2, f2, grid)
    .add_edges([("Start", 1, 1.0), (1, 2, 1.0), (2, "End", 1.0)])
    .validate())

app = ApplicationPerformanceModeling(workflow.to_networkx(), delay_type="SFN")
app.cost_calculator.aws_pricing_units = {"compute": 1.6667e-5, "request": 2e-7}  # or fetched from AWS

opt = ApplicationOptimizer(app)
result = opt.BPBC(budget=opt.maximal_cost, accuracy_constraint=0.75,
                  accuracy_formula=lambda a, b: (a + b) / 2)
print(result.response_time_ms, result.cost, result.memory_config, result.model_config)
```

The full runnable version is [`examples/optimize_workflow.py`](./examples/optimize_workflow.py).

## 📚 Documentation

- [Architecture](./docs/architecture.md) — layered design, dependency graph, data flow.
- [Developer guide](./docs/developer_guide.md) — setup, tests, extending OptiServe.
- [`experiments/`](./experiments) — notebooks demonstrating the end-to-end research workflow.

## 🛠 How to install?

OptiServe is a standard Python package (`pyproject.toml`). We developed and
tested it on **Python 3.11**. AWS credentials are read from the standard AWS
credential chain (environment variables, `~/.aws/credentials`, or an instance
role) — no `.env` file is required for the library itself.

1. Clone the project and move into the root directory:

```bash
git clone https://github.com/pacslab/optiserve.git
cd optiserve
```

2. Install the package (editable install recommended for development):

<details open>
<summary><strong>Option A: Using pip (canonical)</strong></summary>

```bash
python -m pip install -e .                 # core library
python -m pip install -e ".[experiments]"  # + notebook / benchmark deps
```

</details>

<details open>
<summary><strong>Option B: Using Conda</strong></summary>

```bash
conda env create -f environment.yml
conda activate optiserve
```

</details>

> Dependencies are declared once, in `pyproject.toml`. `requirements.txt` mirrors
> the core runtime deps for convenience; `environment.yml` provides a Conda path.

## ▶️ How to use?

- **As a library** — see the Quickstart above and [`examples/`](./examples).
- **Reproducing the experiments** — the Jupyter notebooks in
  [`experiments/`](./experiments) demonstrate profiling, modeling, optimization,
  and the evaluation figures.
- **Tests** — `pytest -q` runs the offline suite, including golden-master
  regressions that lock the analytical core.

## 🗂 Project layout

```
optiserve/      the framework (aws, cost, profiling, modeling, workflow,
                optimization, evaluation, datasets, visualization)
experiments/    benchmark functions + evaluation notebooks
examples/       runnable usage examples
docs/           architecture and developer documentation
tests/          unit + golden-master regression tests
```
