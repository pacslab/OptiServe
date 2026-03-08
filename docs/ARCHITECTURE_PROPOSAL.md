# OptiServe — Architecture Proposal & Refactoring Plan

> Status: **Proposal — approved (decisions locked), awaiting go-ahead on Stage 1.**
> No source code has been changed yet.
> This document is the output of a full read of every module, notebook, binary
> artifact, and the deleted `src/application` prototype recovered from git.
>
> **Locked decisions:** (1) fix confirmed bugs + document each; (2) full internal
> redesign (dedupe copy-pasted heuristics/graph code); (3) support real measured
> accuracy as a first-class input, normalized-rank as fallback; (4) reconciled to
> `origin/main` as the canonical base.
>
> **Reconciliation outcome:** work now proceeds on branch **`refactor/framework`**
> based at `origin/main` (`dfb68bd`) — the complete codebase incl. the bert/resnet/
> yolo benchmark sources. Full prior local state is preserved in two backup refs:
> `backup/local-main-f26fbc8` (the local commit) and `backup/local-worktree-snapshot`
> (the entire dirty working tree incl. untracked notebooks/artifacts). Generated
> artifacts (`modeled_functions/*.mdl`, `App*_accuracy.npy`, result PDFs) are
> restored into the working tree as cached inputs. Origin had already fixed a few
> minor debt items (e.g. `CostCalculator`'s unused `function_name` param); the core
> `application_optimizer.py` and modeling file are unchanged from the analysis.

---

## 1. What OptiServe actually is (understanding)

OptiServe answers one research question: **given a serverless ML workflow, what
per-function memory size and ML-model variant minimizes/maximizes one objective
(cost, latency, or accuracy) subject to constraints on the other two?**

It is built from **two independent engines** that today live tangled together:

### Engine A — Function performance modeling (online, per function)
Profiles a *real* deployed AWS Lambda across memory sizes, fits a parametric
latency curve, and recommends a memory size. This is a Parrotfish-style
active-learning loop.

```
Explorer ──invoke Lambda──► Sampler ──samples──► Objective (acquisition)
   │                            │                     │
   │                            ▼                     ▼
ConfigManager            Exploration          Optimizer (online)
(set mem/model)          (mem,duration)        picks next memory
                                                     │
                                                     ▼
                                           ParamFunction  rt(m)=a0+a1·e^(−m/a2)
                                           (persisted as modeled_functions/*.mdl)
```

### Engine B — Application performance modeling + optimization (offline, whole workflow)
Takes a probabilistic workflow graph (networkx `DiGraph`, `Start`/`End`
sentinels, edge weight = transition probability) whose nodes carry a discrete
`perf_profile[model_variant][memory] → latency` table, analytically computes
expected end-to-end latency / cost / per-node execution counts, and runs three
greedy **Probability-Refined-Critical-Path (PRCP)** heuristics with optional
**Benefit-Cost-Ratio (BCR)** pruning:

| Method (code) | Artifact/paper name | Optimizes | Subject to |
|---|---|---|---|
| `BPBA` | **BPBC** | min latency | budget + accuracy |
| `BCPA` | **BCPC** | min cost | latency + accuracy |
| `BAPB` | **BAPB** (a.k.a. "ALAS") | max accuracy | latency + budget |

```
WorkflowGraph ─► ApplicationPerformanceModeling ─► ApplicationOptimizer
  (mem/rt/         update_ne()  → executions/node    BPBC / BCPC / BAPB
   perf_profile,   get_simple_dag() → reduce         + BCR pruning
   probabilities)  get_avg_rt()/get_avg_cost()       find_PRCP()
                                                            │
                                                            ▼
                                             (rt, cost, accuracy, mem_cfg, model_cfg)
```

The analytical model (`ApplicationPerformanceModeling`, 696 lines) is the real
intellectual core: it collapses **self-loops and cycles** via geometric-series
algebra (`rt/(1−p)`, `ne/(1−p)`), **parallel paths** via max-latency join, and
**probabilistic branches** via expected value, reducing an arbitrary
probabilistic graph to a simple DAG and enumerating path probabilities.

### The missing bridge (key finding)
**Nothing in `src/` connects Engine A to Engine B.** The fitted `ParamFunction`
models (`.mdl`) are turned into the graph's discrete `perf_profile` tables *only
by hand inside notebooks* (`p_model(mem)` over a memory grid). The deleted
`src/application/Function` class had exactly this bridge
(`performance_modeling: ParamFunction`, `get_execution_time(mem, model)`,
`normalized_accuracy = rank/len(models)`), confirming the intended design. **This
bridge is the clearest thing the framework is missing**, and reintroducing it
(cleanly, not as the broken OO prototype) is a primary goal of the refactor.

### Data & artifact flow (as-is)
```
deployed Lambda ─► profiling ─► ParamFunction (.mdl cache)
                       │                 │
                CloudWatch logs           │ (notebook: sample over memory grid)
                (validation CSV)          ▼
                                    perf_profile tables on a networkx graph
                                          │
                                          ▼
                          ApplicationPerformanceModeling
                                          │
                                          ▼
                          ApplicationOptimizer.get_opt_curve  ─► opt_curve_data/App*_{BPBC,BCPC,BAPB}.csv
                          ApplicationOptimizer.get_perf_cost_table ─► perf_cost_data/App*_part*.csv (brute force)
                                          │
                        (notebook: optimizer-vs-bruteforce accuracy)
                                          ▼
                          App*_{BPBC,BCPC,BAPB}_accuracy.npy ─► optimization-accuracy.pdf
                                                                 fixed-optimization-result.pdf
```

---

## 2. Problems (technical debt inventory)

Grouped by severity. "Confirmed" = verified against the code by two independent
readers.

### A. Structural / architectural
1. **`src` is not a package name** — every import is `from src.X`; the deleted
   `setup.py` would have shipped a distribution literally named `src`. No
   `pyproject.toml`; the code only runs from the repo root. No installable
   framework exists.
2. **Two unrelated engines share `src/optimizer/`** — the online single-function
   tuner (`Optimizer`/`Objective`/`ParamFunction`) and the whole-application
   optimizer (`ApplicationOptimizer`) have nothing in common but live together;
   the online tuner is really a *model-fitting* concern.
3. **Experiment harness is welded into the library** — `get_perf_cost_table`
   and `get_opt_curve` (CSV-writing brute-force + sweep drivers) sit inside
   `ApplicationOptimizer`; ground-truth accuracy computation lives only in
   notebook cells. Library code and evaluation code are not separated.
4. **Implicit, untyped domain model** — the graph node schema
   (`perf_profile`, `mem`, `rt`, `models_list`, `BCR`, `available_mem`) is an
   undocumented convention defined only inside notebooks; `Start`/`End` are
   magic strings; the mem/rt units and cost `×1e6` scaling are documented
   nowhere.
5. **Cross-package coupling** — `profiler` imports `analytics` (LogParser);
   `Sampler` reaches into `explorer.invoker._function_name` and
   `explorer._explore()` privates; `Sampler`/`Explorer` share a mutable
   `memory_spaces` dict as a side channel.
6. **Orphaned code** — `src/constraints/` imports the deleted
   `src/application/function` and is unimportable; `utils/consts.py` and
   `utils/zipper.py` have zero importers; `utils/pct_work.py` is untracked.

### B. Confirmed correctness bugs (code cannot run / silently wrong)
7. `application_performance_modeling.py:371` — `type(item) == 'str'` compares a
   type object to the string `'str'`; the B-node termination check is **dead**,
   so `get_simple_dag()` can finish with unresolved branch nodes.
8. `function_performance_modeling.py:127` — `get_performance()` calls
   `self.get_performance_model_as_function(...)`, **which does not exist** →
   guaranteed `AttributeError`.
9. `application_optimizer.py:93` — `np.prod(list_a, list_b)` passes the second
   list as the `axis` argument; `get_perf_cost_table` crashes when
   `end_iterations is None`.
10. `sampler.py` `_explore_first_config` — after pruning it rewrites
    `self.memory_spaces[...]` but the loop keeps reading a **stale local**
    binding → retries the same failing memory forever (infinite loop); the
    epilogue `raise NoMemoryLeft()` also fires on a *successful* length-3 space.
11. `optimizer.py:63` — the `NotEnoughMemory` handler writes `self.memory_space`
    which is **never read**; failed memories can be re-selected indefinitely.
12. `aws_function_logs.py` `get_logs_df` — the grouped-by-model branch iterates
    a freshly created **empty** DataFrame instead of the log dict → always
    returns empty.
13. `invoker.py` — the passed `boto_session` is **ignored**;
    `boto3.client('lambda')` uses the default session (wrong region/creds);
    `FunctionTimeout(duration_ms=timeout_s)` passes seconds into an ms field.
14. `config_manager.py` `set_config` — the wait loop can **never terminate**
    when `model_name is None` but the function already has a `MODEL_NAME` env
    var; `ResourceConflictException` → unbounded recursion.
15. `log_parser.py` `parse_function_execution_time` — `except InvocationError`
    swallows its subclasses `FunctionTimeout`/`NotEnoughMemory`, converting
    log-detected timeouts/OOMs into ordinary duration samples and defeating the
    Sampler's OOM-based memory pruning.
16. `exceptions/cost_calculation_error.py` — constructor misspelled `__initn__`
    (dead); `parametric_function.py` — mutable dataclass default (`bounds`
    lists shared across instances); `parametric_function.minimize` catches its
    own `UnfeasibleConstraint` and silently returns the unconstrained optimum.
17. `get_opt_curve` — invokes `BAPB(BCR=True, BCRtype='ERT/C')` but writes the
    results under **`BCR_disabled_*`** columns (mislabeled experiment data).

### C. Duplication
18. The lexicographic **candidate-selection block** (`max` metric → tie-break →
    `reversed_dict = dict(zip(values, keys))` → pick `target_node`/`target_mem`)
    is copy-pasted **~6 times** across `BPBA`/`BCPA`/`BAPB`; the phase-1
    model-upgrade loop is near-identical between `BPBA` and `BCPA`.
19. The **cycle-elimination algorithm** (self-loop + back-edge removal + weight
    renormalization) is duplicated **4×** in `application_performance_modeling.py`
    (`get_avg_ne` ×2, `process_self_loops`, `simplify_loops`) — `ne`-domain vs
    `rt`-domain copies.
20. The **CloudWatch polling loop** is duplicated verbatim between
    `aws_function_logs` and `aws_application_logs` (belongs in the base class);
    `LogParser` has two identical extraction loops.
21. The **notebook validation pipeline** (filter warm starts → groupby-mean →
    per-row cost → evaluate model over memory grid → real-vs-modeled plots) is
    duplicated between `profile_function.ipynb` and `ml_modeling.ipynb` with
    divergent details (cost `×1` vs `×1_000_000`).
22. `requirements.txt` and `environment.yml` are hand-maintained duplicates
    (already drifted); `scipy` is used but unpinned; `sklearn`/`sagemaker`/
    `requests` are pinned but unused in `src/`.

### D. Research-integrity items an author must own (not "bugs" to silently fix)
23. **Accuracy = normalized rank** `(i+1)/N` — model "accuracy" is fabricated
    from variant ordering, not measured. All reported accuracy numbers derive
    from this placeholder.
24. `Sampler._explore_dynamically` **replaces measured durations** with
    substitutes chosen to force coefficient-of-variation ≤ 5% — silent outlier
    rewriting of experimental data.
25. `CostCalculator` takes `max()` across pricing tiers (always the most
    expensive tier); pricing region hardcoded `us-east-1`.
26. `plot-optimization-results.ipynb` adds **`+0.17`** to specific plotted
    accuracy points; `calculate_optimization_accuracy.ipynb` annotates
    **hardcoded** `top_values=[97.23, …]` with no computation in the repo.
27. Missing inputs: **5 of 6** application graph/formula definitions
    (only App3 is committed); **f2/f5/f6** handler sources exist nowhere; ML
    benchmark sources (bert/resnet/yolo) live only on `origin/main`.

---

## 3. Proposed architecture

### 3.1 Core concepts (the domain model to make explicit)
- **FunctionModel** — a deployed function + its per-variant performance model(s)
  and per-variant accuracy. Owns `latency(memory, variant)`.
- **ModelVariant** — one ML model a function can serve; carries `accuracy`.
- **PerformanceModel** — `latency = f(memory)`; continuous (`ParamFunction`) or
  discrete (`ProfileTable`).
- **WorkflowGraph** — validated probabilistic control-flow graph of function
  nodes; the formal version of today's implicit node-attribute schema.
- **ApplicationModel** — analytical estimator of expected latency / cost /
  executions over a `WorkflowGraph`.
- **Objective** ∈ {cost, latency, accuracy}; **Constraint** on the other two.
- **Configuration** — per-function `(memory, variant)` assignment.
- **OptimizationResult** — a typed result (replaces the current 6-tuple).

### 3.2 Package layout
Rename `src/` → **`optiserve/`** (installable). Layered, acyclic dependencies.

```
optiserve/
    __init__.py            # curated public API + __version__
    config.py              # frozen dataclasses: Profiling/Modeling/Optimization/AWS config
    exceptions.py          # single consolidated hierarchy (was src/exceptions/*)
    logging.py             # per-module loggers, no import-time root config

    aws/                   # thin boto3 adapters — the ONLY place boto3 is touched
        session.py         #   region/credentials
        lambda_client.py   #   invoke + configure (Invoker + ConfigManager, fixed)
        logs_client.py     #   CloudWatch Logs Insights (base + function + application)
        pricing_client.py  #   AWS Pricing API access

    cost.py                # LambdaCostModel (pricing formula; uses aws.pricing_client)

    profiling/             # measurement → samples
        sample.py          #   Sample, Exploration (value objects)
        explorer.py        #   memory/variant sweep orchestration
        sampler.py         #   adaptive sampling strategy (fixed)
        log_parser.py      #   REPORT parsing (was analytics/log_parser)

    modeling/
        parametric.py      #   ParamFunction  rt(m)=a0+a1·e^(−m/a2)  (+ save/load)
        fitting.py         #   online active-learning fit (Optimizer + Objective) — internal
        function_model.py  #   FunctionPerformanceModel (facade; fixed)
        application_model.py #  ApplicationPerformanceModel (expected rt/cost/ne)
        graph_reduction.py #   branch/parallel/loop simplification (extracted, deduped)

    workflow/
        node.py            #   FunctionNode schema (mem, variants, profile, accuracy)
        graph.py           #   WorkflowGraph builder/validator + the MISSING BRIDGE:
                           #   materialize perf_profile tables from FunctionModels

    optimization/
        optimizer.py       #   ApplicationOptimizer (thin orchestrator)
        strategies.py      #   BPBC / BCPC / BAPB unified over one greedy engine
        critical_path.py   #   PRCP computation (find_PRCP)
        bcr.py             #   benefit-cost-ratio pruning modes
        accuracy.py        #   accuracy model (normalized-rank default; real accuracy injectable)
        result.py          #   OptimizationResult dataclass

    evaluation/            # experiment harness (out of the library core)
        ground_truth.py    #   exhaustive perf-cost table (was get_perf_cost_table)
        opt_curve.py       #   constraint sweeps (was get_opt_curve)
        accuracy_metrics.py#   optimizer-vs-bruteforce accuracy (was notebook cells)

    datasets/
        applications.py    #   App1..App6 graph + accuracy-formula builders

    visualization/
        model_plots.py     #   real-vs-modeled duration/cost
        result_plots.py    #   tradeoff curves, accuracy scatter
        graph_plots.py     #   workflow rendering (was drawGraph)

docs/            architecture, module docs, developer guide, examples
examples/        thin runnable scripts (replace notebook driver logic)
experiments/     notebooks reduced to thin drivers over the library
tests/           unit + golden-master regression tests
pyproject.toml   single source of dependencies (core + [aws] + [experiments] extras)
```

### 3.3 Dependency graph (must stay acyclic)
```
config, exceptions, logging          ← leaves, imported everywhere
        ▲
aws  ───┘
cost ──► aws
profiling ──► aws, cost
workflow  ──► (pure; no AWS)                     modeling ──► profiling, cost, workflow
optimization ──► modeling, workflow
evaluation   ──► optimization, modeling, workflow
datasets     ──► workflow
visualization──► (matplotlib; consumes result/model objects only)
```
Rules: `boto3` appears **only** under `aws/`; the analytical `modeling`/
`optimization`/`workflow` layers have **no AWS or filesystem dependency** and
are fully unit-testable offline.

### 3.4 Module responsibilities (purpose · public API · internals · deps)

- **`config.py`** — purpose: replace scattered magic numbers with frozen
  dataclasses. API: `ProfilingConfig`, `ModelingConfig`, `OptimizationConfig`,
  `AWSConfig`. Internals: plain dataclasses with defaults (128/3009 bounds,
  SFN delays 18.81/1, CV 0.05, BCR thresholds, `w=100`). Deps: none.
- **`exceptions.py`** — one hierarchy (`InvocationError`, `NotEnoughMemory`,
  `FunctionTimeout`, `SamplingError`, `NoMemoryLeft`, `OptimizationError`,
  `UnfeasibleConstraint`, …), all exported, `__initn__` fixed, and the
  Timeout-subclasses-NotEnoughMemory hack made explicit or removed. Deps: none.
- **`aws/lambda_client.py`** — invoke + configure a Lambda; honors the passed
  session; bounded, non-recursive `set_config` waiter. API:
  `invoke(payload) → response`, `set_config(memory, timeout, variant)`,
  `get_config()`, `reset_config()`. Deps: `aws.session`, `exceptions`.
- **`aws/logs_client.py`** — CloudWatch Insights with the polling loop in the
  base class; `FunctionLogs.as_dataframe(...)` fixed. Deps: `profiling.log_parser`.
- **`aws/pricing_client.py`** + **`cost.py`** — fetch pricing; `LambdaCostModel.
  cost(memory_mb, duration_ms)`; region/architecture parameterized. Deps: `aws`.
- **`profiling/*`** — `Explorer` (sweeps), `Sampler` (adaptive, stale-binding
  fixed), `Sample`/`Exploration` (typed), `LogParser`. Deps: `aws`, `cost`,
  `config`, `exceptions`.
- **`modeling/parametric.py`** — `ParamFunction` (immutable bounds, honest
  `minimize`). **`fitting.py`** — the online fit loop (dead write fixed).
  **`function_model.py`** — `FunctionPerformanceModel` facade (ctor-ordering
  and `get_performance` bugs fixed). **`application_model.py`** +
  **`graph_reduction.py`** — expected rt/cost/ne with the cycle algorithm
  deduped and the `is_simple`/`np.product` bugs fixed; a `get_simple_dag`
  progress guard. Deps: `profiling`, `cost`, `workflow`.
- **`workflow/graph.py`** — `WorkflowGraph` builder + validator (probabilities,
  Start/End, reachability) **and the bridge**: `attach_profiles(function_models,
  memory_grid)` materializes each node's `perf_profile`. Deps: `modeling.parametric`.
- **`optimization/*`** — `ApplicationOptimizer` delegates to `strategies` (one
  greedy engine parameterized by objective/constraints), `critical_path`, `bcr`,
  `accuracy`; returns `OptimizationResult`. Deps: `modeling`, `workflow`.
- **`evaluation/*`** — brute-force `ground_truth`, `opt_curve` sweeps,
  `accuracy_metrics`. Deps: `optimization`, `modeling`.
- **`datasets/applications.py`** — `build_app(n) → (WorkflowGraph, accuracy_formula)`
  for App1..App6. Deps: `workflow`. *(needs the 5 missing app definitions).*
- **`visualization/*`** — pure plotting over result/model objects. Deps: matplotlib.

---

## 4. Staged refactoring plan

Each stage is independently reviewable, keeps the tree runnable, and is guarded
by tests. **Golden-master tests are the safety net for behavior preservation**:
before touching the analytical core we capture current outputs (the
`application_modeling.ipynb` reference values are already known —
`ne={1:1.4286, 2:0.2857, 3:1.1429, 4:1.4286, 5:1.4286, 6:1.25}`,
`cost=63.9674`, `rt=1739.2857` — plus App3's opt-curve CSVs) and assert they
still reproduce.

| Stage | Scope | Files | Improvement | Risk | Mitigation |
|---|---|---|---|---|---|
| **1. Packaging** | `src/`→`optiserve/`, `pyproject.toml`, consolidate deps | all imports, new pyproject | installable framework, no `src` antipattern | **`.mdl` pickles reference `src.optimizer.parametric_function`** — renaming breaks all 18 | compat shim module + a one-shot re-serialization script; import smoke test |
| **2. Cross-cutting** | merge exceptions→`exceptions.py`, `logging.py`, delete dead `consts`/`zipper`, add `config.py` | exceptions/*, utils/* | single hierarchy, per-module logs, no import side effects | low | unit test import of every module |
| **3. AWS layer** | extract `aws/`, dedupe polling, fix invoker/config bugs; `cost.py` | profiler/*, analytics/* | boto3 isolated, live-path bugs fixed | med (live-AWS only) | mock-boto3 unit tests; gate behavior fixes on Q1 |
| **4. Profiling** | `profiling/`, fix Sampler infinite loop + log-parser swallow | explorer/sampler/sample/exploration | correct data acquisition | med | Sampler tests with a fake Explorer |
| **5. Modeling core** | `parametric`/`fitting`/`function_model`/`application_model`/`graph_reduction`; fix `is_simple`, `np.product`, dedupe cycles | modeling/* + optimizer/{parametric,objective,optimizer} | the analytical heart, deduped & correct | **HIGH** | **golden-master** on example graph + App3 |
| **6. Workflow** | `workflow/graph.py`+`node.py`, add the model→profile **bridge** | new | formal schema, closes the Engine-A↔B gap | low (additive) | round-trip test vs notebook-built graphs |
| **7. Optimization** | split experiment methods out; unify BPBC/BCPC/BAPB; `OptimizationResult`; fix mislabeled columns/naming | application_optimizer.py (1020 lines) | ~6× dedup, typed results | **HIGH** | golden-master on App3 `*_{BPBC,BCPC,BAPB}.csv` before dedup |
| **8. Evaluation/datasets/viz** | promote notebook logic; App1..App6 builders; plotting | new + notebooks | reproducible experiments | med | needs missing app defs (Q4) |
| **9. Docs** | architecture, module docs, dev guide, install, examples, diagrams, README | docs/, README | research-grade docs | low | — |

Commits stay logically grouped (one concern per commit); no stage breaks the
build.

---

## 5. Decisions required before Stage 1 (blocking)

These change the plan itself and are asked separately:
1. **Bug policy** — fix confirmed bugs (§2B) or preserve exact behavior?
2. **Refactor depth** — full internal redesign, or repackage + interfaces only?
3. **Accuracy model** — support real measured accuracy as a first-class input,
   or keep normalized-rank only? (§2D-23)
4. **Starting point** — refactor the local working tree as-is, or reconcile with
   `origin/main` first (local is 1 ahead / 4 behind; ML benchmark sources live
   only on origin)?

## 6. Non-blocking open questions (can proceed with a documented assumption)
- Units: what is `get_avg_cost`'s `×1_000_000` (USD per million invocations?);
  are optimizer budgets in the same unit?
- Provenance of `18.81 ms`/`1 ms` SFN delays — configurable constants?
- `Sampler` CV≤5% duration substitution (§2D-24) — documented methodology or
  prototype hack to remove?
- Pricing `max()` across tiers (§2D-25) — intentional worst-case?
- The `+0.17` plot edit and hardcoded `top_values` (§2D-26) — legitimate
  corrections or to be removed?
- Where should experiment artifacts (`*.npy`, PDFs, `.mdl`) live and be tracked?
- Should `.mdl` be re-serialized as plain parameter arrays/JSON (robust to module
  moves) instead of joblib pickles of `ParamFunction`?
- Fate of `src/constraints/` (orphaned/broken) — delete (recommended) or rewire?
