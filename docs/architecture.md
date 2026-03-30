# OptiServe Architecture

OptiServe models and optimizes serverless ML applications on AWS Lambda. It has
two engines:

- **Function modeling** (online): profile a deployed Lambda across memory sizes
  and fit a latency-vs-memory curve per model variant.
- **Application modeling + optimization** (offline): analytically estimate a
  workflow's end-to-end latency, cost, and per-function execution counts, then
  greedily search for the memory/variant configuration that optimizes one
  objective under constraints on the others.

## Layered packages

```
optiserve/
├── config.py          frozen dataclasses for all tunables (no magic numbers)
├── exceptions.py      one OptiServeError-rooted hierarchy
├── logging.py         per-module loggers, no import-time side effects
├── aws/               the ONLY place boto3 is used
│   ├── session.py         session/region/credentials
│   ├── lambda_client.py   Invoker + ConfigManager
│   ├── logs_client.py     CloudWatch Logs (function + application)
│   ├── pricing_client.py  Price List API
│   ├── log_parser.py      REPORT metric parsing
│   └── function_config.py FunctionConfig value object
├── cost.py            CostCalculator (pricing formula, uses aws/)
├── profiling/         measurement → samples
│   ├── sample.py          Sample + Exploration value objects
│   ├── explorer.py        memory/variant sweep orchestration
│   └── sampler.py         adaptive sampling strategy
├── modeling/          performance models
│   ├── parametric.py      ParamFunction  rt(m)=a0+a1·e^(−m/a2)
│   ├── fitting.py         online active-learning fit (Objective + Optimizer)
│   ├── function_model.py  FunctionPerformanceModeling facade
│   ├── application_model.py  ApplicationPerformanceModeling (graph algebra)
│   └── graph_reduction.py    shared cycle-discovery helpers
├── workflow/          the workflow graph schema + model→profile bridge
│   ├── node.py            ModelVariant, FunctionNode
│   └── graph.py           WorkflowGraph builder / validator
├── optimization/      the application-level optimizer
│   ├── application_optimizer.py  BPBC / BCPC / BAPB greedy PRCP strategies
│   ├── accuracy.py        AccuracyModel (measured accuracy or normalized rank)
│   └── result.py          OptimizationResult
├── evaluation/        experiment harness (separate from the library)
│   ├── experiments.py     ground-truth table + optimization-curve sweeps
│   └── accuracy_metrics.py optimizer-vs-brute-force accuracy
├── datasets/          benchmark application builders (App3)
└── visualization/     matplotlib plotting helpers (lazy import)
```

## Dependency graph (acyclic, bottom-up)

```
config, exceptions, logging          ← leaves, imported everywhere
        ▲
aws  ───┘
cost ──► aws
profiling ──► aws, cost
workflow  ──► modeling.parametric      (pure; no AWS)
modeling  ──► profiling, cost, workflow
optimization ──► modeling, workflow
evaluation   ──► optimization, modeling
datasets     ──► modeling, workflow
visualization──► (matplotlib only)
```

`boto3` appears only under `aws/`; the analytical layers (`modeling`,
`optimization`, `workflow`) have no AWS or filesystem dependency and are fully
unit-testable offline.

## End-to-end data flow

```
                     ┌─────────────── online (AWS) ───────────────┐
 deployed Lambda ──► profiling.Explorer ──► profiling.Sampler ──► modeling.fitting
   (per variant)        (invoke, logs)        (adaptive)             (curve fit)
                                                                        │
                                                          modeling.ParamFunction
                                                          (modeled_functions/*.mdl)
                                                                        │
        ┌──────────────── offline (no AWS) ──────────────────┐         │
        │                                                     ▼         ▼
   workflow.WorkflowGraph.add_ml_function ◄──────── the bridge: sample models
        │  (perf_profile, models_list, accuracy_list)         over a memory grid
        ▼
   modeling.ApplicationPerformanceModeling
        │  update_ne() → executions/node
        │  get_simple_dag() → reduce branches/parallels/loops
        │  get_avg_rt(), get_avg_cost()
        ▼
   optimization.ApplicationOptimizer
        │  BPBC / BCPC / BAPB  (+ BCR pruning, PRCP critical paths)
        ▼
   optimization.OptimizationResult(rt, cost, accuracy, mem_config, model_config)

   evaluation.run_opt_curve / generate_perf_cost_table → CSV artifacts
   evaluation.*_optimization_accuracy → optimizer-vs-brute-force quality
```

## Core concepts

- **ModelVariant** — one ML model a function can serve (performance model +
  optional measured accuracy).
- **WorkflowGraph** — probabilistic control-flow graph with `Start`/`End`
  sentinels and edge transition probabilities; `add_ml_function` materializes
  each node's discrete `perf_profile` from its variants (the model→graph bridge).
- **ApplicationPerformanceModeling** — collapses self-loops/cycles
  (`rt/(1−p)`), parallel joins (max), and probabilistic branches (expected
  value) to compute expected end-to-end latency and cost.
- **ApplicationOptimizer** — three greedy Probability-Refined-Critical-Path
  strategies (BPBC: min latency; BCPC: min cost; BAPB: max accuracy), each under
  two constraints, with optional Benefit-Cost-Ratio pruning.

## Optimization strategies

| Strategy | Optimizes      | Subject to               | Artifact suffix |
|----------|----------------|--------------------------|-----------------|
| `BPBC`   | min latency    | budget + accuracy        | `_BPBC.csv`     |
| `BCPC`   | min cost       | latency + accuracy       | `_BCPC.csv`     |
| `BAPB`   | max accuracy   | latency + budget         | `_BAPB.csv`     |

(The methods also answer to their historical names `BPBA` / `BCPA`.)
