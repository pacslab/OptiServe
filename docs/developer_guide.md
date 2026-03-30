# Developer Guide

## Setup

```bash
git clone https://github.com/pacslab/optiserve.git
cd optiserve
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # library + pytest
pip install -e ".[experiments]"  # + notebook / benchmark deps
```

AWS credentials (only needed for live profiling) are read from the standard AWS
credential chain — environment variables, `~/.aws/credentials`, or an instance
role.

## Running the tests

```bash
pytest -q
```

The suite is fully offline. It includes **golden-master** regressions under
`tests/golden/` that lock the numerical behavior of the analytical core:

- `app_modeling_baseline.json` — the notebook app (rt = 1739.2857).
- `modeling_baseline.json` — a battery of branch / parallel / self-loop / mixed
  graphs under both delay models.
- `optimizer_baseline.json` — every strategy / BCR variant on acyclic and cyclic
  graphs.
- `evaluation_baseline.json` — byte-for-byte CSV output of the experiment drivers.

If you intentionally change modeling or optimizer behavior, regenerate the
relevant baseline (see the capture code in each `*_cases.py`) and review the diff
carefully — these guard published results.

## Extending OptiServe

**Add a model variant to a function.** Construct a `ModelVariant(name,
performance_model, accuracy=...)` where `performance_model` maps memory (MB) →
latency (ms) — usually a fitted `ParamFunction`. Pass the variants to
`WorkflowGraph.add_ml_function`.

**Add a benchmark application.** Add a builder to
`optiserve/datasets/applications.py` that assembles a `WorkflowGraph` and returns
it with its end-to-end accuracy formula (see `build_app3`).

**Use measured accuracy instead of ranks.** Provide `accuracy=` on each
`ModelVariant`; `AccuracyModel` uses measured values when every variant on a node
has one, otherwise falls back to the normalized rank `i/N`.

**Provide your own pricing.** Inject a `PricingClient` or set
`CostCalculator(...).aws_pricing_units = {"compute": ..., "request": ...}` to run
cost/optimization offline (as the tests and examples do).

## Conventions

- boto3 access lives only under `optiserve/aws/`.
- Get a logger with `from optiserve.logging import get_logger; logger =
  get_logger(__name__)`. The library configures no handlers; call
  `optiserve.logging.configure_logging()` in scripts/notebooks to see output.
- Tunables live in `optiserve/config.py` dataclasses, not as inline literals.

## The .mdl model cache

Fitted models are joblib pickles under `modeled_functions/` (gitignored;
regenerating requires live AWS). They reference
`optiserve.modeling.parametric`. If that module is ever moved, re-point existing
caches with `python scripts/migrate_mdl.py` (and `--check` to verify).
