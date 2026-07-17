# Developer Guide

## Setup

```bash
git clone https://github.com/2arian3/OptiServe.git
cd OptiServe
make install                     # venv + dev extras, editable
# or, by hand:
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install --no-deps -e .
pip install -e ".[experiments]"  # + notebook / benchmark deps
```

Or skip the local environment entirely:

```bash
docker compose run --rm tests        # offline suite
docker compose run --rm integration  # AWS adapters against a moto server
docker compose run --rm example      # the offline optimization example
```

AWS credentials (only needed for live profiling) are read from the standard AWS
credential chain — environment variables, `~/.aws/credentials`, or an instance
role.

## Running the tests

```bash
make check           # lint + type check + dependency sync + offline suite
pytest -q                        # everything (155 tests)
pytest -q -m "not integration"   # offline only — no network at all
pytest -q -m integration         # AWS adapters against in-process moto
pytest -q -m golden              # only the published-result regressions
```

`tests/conftest.py` scrubs AWS configuration for *every* test, so a missing mock
surfaces as an authentication error in that test rather than as a surprise API
call against a real account. Integration tests opt back in to a mocked endpoint
themselves.

It includes **golden-master** regressions under
`tests/golden/` that lock the numerical behavior of the analytical core:

- `app_modeling_baseline.json` — the notebook app (rt = 1739.2857).
- `modeling_baseline.json` — a battery of branch / parallel / self-loop / mixed
  graphs under both delay models.
- `optimizer_baseline.json` — every strategy / BCR variant on acyclic and cyclic
  graphs.
- `evaluation_baseline.json` — byte-for-byte CSV output of the experiment drivers.
- `optimizer_baseline_corrected.json` — the same optimizer matrix under the
  **CORRECTED** compat preset (see below).

If you intentionally change modeling or optimizer behavior, regenerate the
relevant baseline and review the diff line by line — these guard published
results:

```bash
python tests/golden/regenerate.py --list
python tests/golden/regenerate.py optimizer_corrected
python tests/golden/regenerate.py --all      # reports CHANGED / unchanged
```

The script compares *parsed content*, not bytes, so a reformat is not reported
as a behavior change.

## Published results vs. corrected results

Two of the greedy optimizer's defects cannot be fixed without moving numbers
that appear in the paper. Rather than choosing between "silently restate a
published result" and "ship a known-wrong optimizer", each divergence is a
named flag:

```python
from optiserve.config import OptimizationConfig
from optiserve.optimization.compat import OptimizerCompat

# The default — every known defect fixed.
ApplicationOptimizer(app)

# Bug-for-bug reproduction of the published runs.
ApplicationOptimizer(app, config=OptimizationConfig(
    compat=OptimizerCompat.PUBLISHED))
```

`OptimizerCompat.PUBLISHED` currently covers the BPBC budget double-count, the
BAPB memory gate, and the BCR alias namespace mismatch — each documented in
`optiserve/optimization/compat.py` with what it does and why it is wrong. Both
presets are frozen by a golden baseline, and a test asserts they actually
differ, so the flags cannot decay into no-ops.

**Always state which preset produced a number you report.**

## Profiling a live function safely

`profile` mutates a deployed function's memory, timeout and `MODEL_NAME`, and
invokes it hundreds of times. Never call the profiler outside its session:

```python
model = FunctionPerformanceModeling("my-fn", checkpoint_store=JsonCheckpointStore("out/ckpt"))
with model.profiling_session():        # restores on EVERY exit path
    curve = model.get_performance_model()
```

or from the CLI, which additionally turns `SIGTERM` into a clean unwind:

```bash
optiserve profile --function my-fn --yes \
  --checkpoint-dir output/checkpoints \
  --trace output/run.jsonl \
  --output-dir modeled_functions
```

A run resumes from its checkpoint if interrupted. Changing the payload,
iteration count or CV settings starts a *new* run rather than mixing
incomparable measurements into one curve.

## Observability

The library emits nothing until you ask it to:

```python
from optiserve.observability import JsonlSink, LoggingSink, hooks
hooks.add(JsonlSink("output/run.jsonl"))   # durable per-run audit trail
hooks.add(LoggingSink())                   # human-readable progress
```

Sinks can never fail a run — a raising sink loses its own event and nothing
else. Inside Lambda or Fargate use `EmfSink`, which publishes CloudWatch metrics
through stdout with no extra API call per sample.

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
- `optiserve/modeling/application_model.py`, `optiserve/optimization/application_optimizer.py`
  and `optiserve/evaluation/experiments.py` are **frozen**: their behavior is
  locked by golden baselines, so they are excluded from ruff's stylistic rules
  and from the formatter. A "cleanup" there is a silent result change. They are
  never excluded from pyflakes.
- Run `make format` before committing; CI runs `ruff format --check`.

## The .mdl model cache

Fitted models are joblib pickles under `modeled_functions/` (gitignored;
regenerating requires live AWS). They reference
`optiserve.modeling.parametric`. If that module is ever moved, re-point existing
caches with `python scripts/migrate_mdl.py` (and `--check` to verify).
