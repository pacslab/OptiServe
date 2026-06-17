"""Command-line interface for OptiServe.

Three subcommands, matching the three things the framework actually does:

    optiserve profile   --function NAME ...      profile a live Lambda -> .mdl curve
    optiserve optimize  --workflow SPEC.json     optimize a workflow offline
    optiserve version                            report version and environment

``optimize`` is fully offline — it is what the container image runs, and what
CI smoke-tests — while ``profile`` is the only path that touches AWS. That one
is guarded: it refuses to mutate a live function without ``--yes``, and it wraps
the run in :meth:`~optiserve.modeling.function_model.FunctionPerformanceModeling
.profiling_session` so the function is restored on every exit path.

The workflow spec is declarative JSON. Deliberately, it does **not** accept
Python expressions for the end-to-end accuracy formula: a spec file is data,
often produced by another tool, and ``eval``-ing it would make reading one
equivalent to running it. Named formulas cover the cases the paper uses, and
``weighted`` covers the rest.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from optiserve import __version__
from optiserve.logging import configure_logging, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Accuracy formulas
# --------------------------------------------------------------------------- #
def _weighted(weights: Sequence[float]) -> Callable[..., float]:
    def formula(*accuracies: float) -> float:
        if len(accuracies) != len(weights):
            raise ValueError(
                f"accuracy formula expects {len(weights)} functions, got {len(accuracies)}"
            )
        return sum(w * a for w, a in zip(weights, accuracies, strict=True))

    return formula


_NAMED_FORMULAS: dict[str, Callable[..., float]] = {
    "mean": lambda *a: sum(a) / len(a),
    "min": lambda *a: min(a),
    "sum": lambda *a: sum(a),
    "product": lambda *a: _product(a),
}


def _product(values: Sequence[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def build_accuracy_formula(spec: Any) -> Callable[..., float]:
    """Resolve a spec's ``accuracy_formula`` to a callable.

    Accepts a name from :data:`_NAMED_FORMULAS`, or
    ``{"weighted": [w0, w1, ...]}``. Never evaluates arbitrary code.
    """
    if spec is None:
        return _NAMED_FORMULAS["mean"]
    if isinstance(spec, str):
        try:
            return _NAMED_FORMULAS[spec]
        except KeyError:
            raise ValueError(
                f"unknown accuracy_formula {spec!r}; expected one of "
                f"{sorted(_NAMED_FORMULAS)} or {{'weighted': [...]}}"
            ) from None
    if isinstance(spec, dict) and "weighted" in spec:
        weights = [float(w) for w in spec["weighted"]]
        return _weighted(weights)
    raise ValueError(
        f"unsupported accuracy_formula {spec!r}; expected one of "
        f"{sorted(_NAMED_FORMULAS)} or {{'weighted': [...]}}"
    )


# --------------------------------------------------------------------------- #
# Workflow spec -> WorkflowGraph
# --------------------------------------------------------------------------- #
def _performance_model(spec: dict) -> Callable[[float], float]:
    """Build a memory -> latency callable from a variant's ``model`` block."""
    from optiserve.modeling.parametric import ParamFunction

    kind = spec.get("kind", "param_file")
    if kind == "param_file":
        return ParamFunction.load(Path(spec["path"]))
    if kind == "linear":
        intercept = float(spec["intercept"])
        slope = float(spec["slope"])
        return lambda memory_mb: intercept - slope * memory_mb
    if kind == "exponential":
        a0, a1, a2 = float(spec["a0"]), float(spec["a1"]), float(spec["a2"])
        model = ParamFunction()
        model.params = [a0, a1, a2]
        return model
    raise ValueError(f"unknown performance model kind {kind!r}")


def load_workflow(spec: dict):
    """Build a :class:`~optiserve.workflow.graph.WorkflowGraph` from a spec dict."""
    from optiserve.workflow import ModelVariant, WorkflowGraph

    graph = WorkflowGraph()
    for function in spec["functions"]:
        variants = [
            ModelVariant(
                name=variant["name"],
                performance_model=_performance_model(variant["model"]),
                accuracy=variant.get("accuracy"),
            )
            for variant in function["variants"]
        ]
        grid = function["memory_grid"]
        if isinstance(grid, dict):  # {"start": .., "stop": .., "step": ..}
            grid = list(range(grid["start"], grid["stop"], grid.get("step", 128)))
        graph.add_ml_function(function["id"], variants, grid)

    graph.add_edges([tuple(edge) for edge in spec["edges"]])
    return graph.validate()


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_optimize(args: argparse.Namespace) -> int:
    from optiserve.modeling.application_model import ApplicationPerformanceModeling
    from optiserve.optimization.application_optimizer import ApplicationOptimizer

    spec = json.loads(Path(args.workflow).read_text())
    workflow = load_workflow(spec)
    formula = build_accuracy_formula(spec.get("accuracy_formula"))

    app = ApplicationPerformanceModeling(
        workflow.to_networkx(),
        delay_type=spec.get("delay_type", "SFN"),
        # Memoization on: for the greedy strategies it is a measured ~2x with
        # provably identical output (tests/test_cache_equivalence.py).
        cache_evaluations=not args.no_cache,
    )

    pricing = spec.get("pricing")
    if pricing:
        # Fixed unit prices: the analytical layer then needs no AWS at all.
        app.cost_calculator.aws_pricing_units = dict(pricing)

    optimizer = ApplicationOptimizer(app)

    if args.strategy == "bpbc":
        budget = args.budget if args.budget is not None else optimizer.maximal_cost
        result = optimizer.BPBC(
            budget=budget,
            accuracy_constraint=args.accuracy,
            accuracy_formula=formula,
            BCR=args.bcr,
        )
    elif args.strategy == "bcpc":
        constraint = args.latency if args.latency is not None else optimizer.maximal_avg_rt
        result = optimizer.BCPC(
            rt_constraint=constraint,
            accuracy_constraint=args.accuracy,
            accuracy_formula=formula,
            BCR=args.bcr,
        )
    else:  # bapb
        constraint = args.latency if args.latency is not None else optimizer.maximal_avg_rt
        budget = args.budget if args.budget is not None else optimizer.maximal_cost
        result = optimizer.BAPB(
            rt_constraint=constraint,
            budget=budget,
            accuracy_formula=formula,
            BCR=args.bcr,
        )

    payload = {
        "strategy": args.strategy,
        "response_time_ms": float(result.response_time_ms),
        "cost": float(result.cost),
        "accuracy": float(result.accuracy),
        "memory_config": {str(k): int(v) for k, v in result.memory_config.items()},
        "model_config": {str(k): int(v) for k, v in result.model_config.items()},
        "iterations": int(result.iterations),
        "cache_stats": app.cache_stats(),
    }
    _emit(payload, args.output)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from optiserve.config import ProfilingConfig
    from optiserve.modeling.function_model import FunctionPerformanceModeling
    from optiserve.observability import JsonlSink, hooks
    from optiserve.profiling.state import JsonCheckpointStore

    if not args.yes:
        logger.error(
            "Refusing to profile %s without --yes: profiling repeatedly rewrites "
            "the live function's MemorySize, Timeout and MODEL_NAME and invokes "
            "it hundreds of times (which costs money). Re-run with --yes to "
            "confirm.",
            args.function,
        )
        return 2

    if args.trace:
        hooks.add(JsonlSink(args.trace))

    config = ProfilingConfig(
        memory_bounds=(args.memory_min, args.memory_max),
        profiling_iterations=args.iterations,
        noise_reduction=not args.no_noise_reduction,
    )
    checkpoints = JsonCheckpointStore(args.checkpoint_dir) if args.checkpoint_dir else None

    model = FunctionPerformanceModeling(
        function_name=args.function,
        memory_bounds=(args.memory_min, args.memory_max),
        region_name=args.region,
        profiling_iterations=args.iterations,
        max_total_sample_count=args.max_samples,
        payload=args.payload,
        available_models=args.models or None,
        config=config,
        checkpoint_store=checkpoints,
    )

    # SIGTERM does not raise, so a container stop would skip the restore that
    # `profiling_session`'s `finally` performs. Turn it into an exception.
    _install_sigterm_handler()

    with model.profiling_session():
        for model_name in model.available_models:
            model.run(model_name=model_name)
            curve = model.param_functions[model_name]
            if args.output_dir:
                destination = Path(args.output_dir) / f"{args.function}_{model_name}.mdl"
                curve.save(destination)
                logger.info("Wrote %s", destination)

    payload = {
        "function": args.function,
        "models": {
            name: {
                "params": [float(p) for p in model.param_functions[name].params],
                "optimal_memory_mb": int(
                    model.param_functions[name].minimize(model.explorer.memory_spaces[name])
                ),
                "samples": len(model.sampler.explorations.get(name, [])),
            }
            for name in model.available_models
        },
    }
    _emit(payload, args.output)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    import platform

    payload = {
        "optiserve": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import boto3

        payload["boto3"] = boto3.__version__
    except Exception:
        payload["boto3"] = "unavailable"
    _emit(payload, args.output)
    return 0


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def _emit(payload: dict, output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
        logger.info("Wrote %s", output)
    else:
        sys.stdout.write(text + "\n")


def _install_sigterm_handler() -> None:
    def handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # not on the main thread, or unsupported
        logger.debug("Could not install a SIGTERM handler", exc_info=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optiserve",
        description=(
            "Model and optimize serverless ML workflows on AWS Lambda (cost / latency / accuracy)."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="errors only")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- optimize ---------------------------------------------------------- #
    optimize = subparsers.add_parser(
        "optimize", help="optimize a workflow offline from a JSON spec"
    )
    optimize.add_argument("--workflow", required=True, help="path to the workflow spec")
    optimize.add_argument(
        "--strategy",
        choices=("bpbc", "bcpc", "bapb"),
        default="bpbc",
        help="bpbc: min latency | bcpc: min cost | bapb: max accuracy",
    )
    optimize.add_argument("--budget", type=float, help="cost budget (bpbc, bapb)")
    optimize.add_argument("--latency", type=float, help="latency limit ms (bcpc, bapb)")
    optimize.add_argument("--accuracy", type=float, default=0.0, help="accuracy floor (bpbc, bcpc)")
    optimize.add_argument("--bcr", action="store_true", help="enable BCR pruning")
    optimize.add_argument("--no-cache", action="store_true", help="disable evaluation memoization")
    optimize.add_argument("--output", "-o", help="write JSON result here")
    optimize.set_defaults(func=cmd_optimize)

    # -- profile ----------------------------------------------------------- #
    profile = subparsers.add_parser(
        "profile", help="profile a live Lambda and fit its latency curve"
    )
    profile.add_argument("--function", required=True, help="Lambda function name")
    profile.add_argument("--region", default="us-east-1")
    profile.add_argument("--memory-min", type=int, default=128)
    profile.add_argument("--memory-max", type=int, default=3008)
    profile.add_argument("--iterations", type=int, default=4)
    profile.add_argument("--max-samples", type=int, default=20)
    profile.add_argument("--payload", default='{"key1": "value1"}')
    profile.add_argument("--models", nargs="*", help="MODEL_NAME values to profile separately")
    profile.add_argument("--checkpoint-dir", help="persist state here so a run resumes")
    profile.add_argument("--output-dir", help="write fitted .mdl curves here")
    profile.add_argument("--trace", help="write a JSONL event trace here")
    profile.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="keep raw measured durations (disables CV substitution)",
    )
    profile.add_argument(
        "--yes",
        action="store_true",
        help="confirm mutating and invoking the live function",
    )
    profile.add_argument("--output", "-o", help="write JSON result here")
    profile.set_defaults(func=cmd_profile)

    # -- version ----------------------------------------------------------- #
    version = subparsers.add_parser("version", help="print version information")
    version.add_argument("--output", "-o")
    version.set_defaults(func=cmd_version)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.ERROR
    configure_logging(level)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
