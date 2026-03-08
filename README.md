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

To see how OptiServe works and how to apply it to your own workflows, please check the Jupyter notebooks in the [`experiments`](./experiments) directory.
