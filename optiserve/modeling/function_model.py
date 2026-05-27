"""Per-function performance modeling.

:class:`FunctionPerformanceModeling` is a facade that wires together the
profiler (Explorer/Sampler) and the online fit (Optimizer/Objective) to build a
:class:`~optiserve.modeling.parametric.ParamFunction` latency-vs-memory curve for
one deployed Lambda, optionally per ML-model variant.

This is the only component that touches a *live* function, so it owns three
production responsibilities the rest of the library does not:

* the boto3 session is **injected**, so an offline or mocked run needs no code
  change — only a different session, or ``AWS_ENDPOINT_URL``;
* tunables come from :class:`~optiserve.config.ProfilingConfig` rather than from
  a scattered set of default arguments that nothing could override centrally; and
* the profiled function is **restored** on every exit path, and a partially
  completed run can be **resumed** rather than paid for twice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import boto3
import numpy as np

from optiserve.aws.session import create_session
from optiserve.config import AWSConfig, ProfilingConfig
from optiserve.logging import get_logger
from optiserve.modeling.fitting import Objective, Optimizer
from optiserve.modeling.parametric import ParamFunction
from optiserve.observability.events import EventName
from optiserve.observability.hooks import HookRegistry
from optiserve.observability.hooks import hooks as default_hooks
from optiserve.profiling.explorer import Explorer
from optiserve.profiling.sampler import Sampler
from optiserve.profiling.state import CheckpointStore

logger = get_logger(__name__)


class FunctionPerformanceModeling:
    def __init__(
        self,
        function_name: str,
        max_invocations: int = 5,
        memory_bounds: tuple[int, int] | list[tuple[int, int]] = (128, 3008),
        region_name: str = "us-east-1",
        knowledge_termination_threshold: int = 3,
        profiling_iterations: int = 4,
        max_total_sample_count: int = 20,
        payload: str = '{"key1": "value1"}',
        available_models: list[str] | None = None,
        memory_space_step: int = 1,
        *,
        config: ProfilingConfig | None = None,
        aws_config: AWSConfig | None = None,
        boto_session: boto3.Session | None = None,
        checkpoint_store: CheckpointStore | None = None,
        hooks: HookRegistry | None = None,
    ):
        if not function_name:
            raise ValueError("Function name is required.")

        # Explicit positional/keyword arguments still win, so existing call
        # sites behave identically; `config` supplies what the caller did not
        # pin — previously ProfilingConfig existed but reached nothing.
        settings = config or ProfilingConfig()
        aws = aws_config or AWSConfig(region_name=region_name)

        self.function_name = function_name
        self._hooks = hooks if hooks is not None else default_hooks

        # Default the model list BEFORE building the Explorer, so the memory
        # spaces are keyed correctly (previously this ran after, leaving the
        # memory spaces empty for the list-of-bounds + no-models case).
        if available_models is None:
            available_models = ["None"]
        self.available_models: list[str] = available_models

        # Injected session: an AWS-free run supplies its own, or points
        # AWS_ENDPOINT_URL at a local mock, without touching this class.
        session = boto_session or create_session(region_name=aws.region_name)

        self.explorer = Explorer(
            function_name=function_name,
            max_invocations=max_invocations,
            memory_bounds=memory_bounds,
            boto_session=session,
            payload=payload,
            available_models=available_models,
            memory_space_step=memory_space_step,
            hooks=self._hooks,
        )

        self.param_functions: dict[str, ParamFunction] = {
            model_name: ParamFunction() for model_name in available_models
        }

        self.objectives: dict[str, Objective] = {
            model_name: Objective(
                param_function=self.param_functions[model_name],
                memory_space=self.explorer.memory_spaces[model_name],
                termination_threshold=knowledge_termination_threshold,
                model_name=model_name,
                knowledge_sigma_mb=settings.knowledge_sigma_mb,
                # Re-read the live space: the sampler prunes infeasible memories
                # by replacing the array, and a stale copy leaves the
                # confidence-based termination measuring an argmin that is no
                # longer in the search space.
                space_provider=self._space_provider(model_name),
            )
            for model_name in available_models
        }

        self.sampler = Sampler(
            explorer=self.explorer,
            profiling_iterations=profiling_iterations,
            memory_floor_step_mb=settings.memory_floor_step_mb,
            cv_threshold=settings.cv_threshold,
            max_dynamic_samples=settings.max_dynamic_samples,
            noise_reduction=settings.noise_reduction,
            checkpoint_store=checkpoint_store,
            hooks=self._hooks,
        )
        self.optimizer = Optimizer(
            objectives=self.objectives,
            sampler=self.sampler,
            max_total_sample_count=max_total_sample_count,
            memory_floor_step_mb=settings.memory_floor_step_mb,
        )
        self._explored: dict[str, bool] = defaultdict(lambda: False)

    def _space_provider(self, model_name: str) -> Callable[[], np.ndarray]:
        """A callable returning the *live* memory space for ``model_name``.

        A named method rather than an inline lambda so the closure over
        ``model_name`` is explicit and the return type is checkable.
        """

        def provider() -> np.ndarray:
            return self.explorer.memory_spaces[model_name]

        return provider

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @contextmanager
    def profiling_session(self) -> Iterator[FunctionPerformanceModeling]:
        """Profile with a guaranteed restore of the live function.

        Profiling rewrites a *deployed* function's memory, timeout and
        ``MODEL_NAME``. Use this whenever profiling anything you care about:
        every exit path — including ``KeyboardInterrupt`` and a crash mid-sweep
        — puts the function back the way it was found.

            model = FunctionPerformanceModeling("my-fn")
            with model.profiling_session():
                curve = model.get_performance_model()
        """
        self._hooks.emit(
            EventName.RUN_STARTED,
            function=self.function_name,
            models=list(self.available_models),
        )
        with self.explorer.config_manager.managed():
            try:
                yield self
            finally:
                self._hooks.emit(EventName.RUN_FINISHED, function=self.function_name)

    def _resolve(self, model_name: str | None) -> str:
        return self.available_models[0] if model_name is None else model_name

    def run(self, model_name: str | None = None) -> None:
        model_name = self._resolve(model_name)
        if not self._explored[model_name]:
            self.optimizer.start(model_name=model_name)
            self._explored[model_name] = True

    def get_optimal_memory(
        self,
        latency_constraint_threshold_ms: float | None = None,
        model_name: str | None = None,
    ) -> int | dict[str, int]:
        models = self.available_models if model_name is None else [model_name]
        results = {}
        for name in models:
            self.run(model_name=name)
            results[name] = self.param_functions[name].minimize(
                self.explorer.memory_spaces[name],
                latency_constraint_threshold_ms=latency_constraint_threshold_ms,
            )
        return next(iter(results.values())) if len(results) == 1 else results

    def get_performance_model(self, model_name: str | None = None) -> ParamFunction:
        model_name = self._resolve(model_name)
        self.run(model_name=model_name)
        return self.param_functions[model_name]

    def get_performance(self, memory_mb: float, model_name: str | None = None) -> float:
        """Predicted execution time (ms) at ``memory_mb`` for ``model_name``.

        (Previously this called a non-existent ``get_performance_model_as_function``
        and raised AttributeError.)"""
        return self.get_performance_model(model_name=model_name)(memory_mb)
