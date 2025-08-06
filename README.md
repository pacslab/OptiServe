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

Create a .env file in the root direcoty based on .env.example template and enter you AWS credentials.
```bash
cp .env.example .env
vi .env
```

We used **Python 3.11.13** to develop and test OptiServe. You can install the dependencies using either `conda` or `pip`. Make sure you're using **Python 3.11** if installing manually with `pip`.

1. Clone the project and move into the root directory:

```bash
git clone https://github.com/your-username/optiserve.git
cd optiserve
```

2. Install dependencies:

<details>
<summary><strong>Option A: Using Conda</strong></summary>

```bash
conda env create -f environment.yml
conda activate optiserve
```

</details>

<details>
<summary><strong>Option B: Using pip</strong></summary>

```bash
python -m pip install -r requirements.txt
```

</details>

## ▶️ How to use?

To see how OptiServe works and how to apply it to your own workflows, please check the Jupyter notebooks in the [`experiments`](./experiments) directory.
