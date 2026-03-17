"""Per-function performance modeling.

:class:`FunctionPerformanceModeling` is a facade that wires together the
profiler (Explorer/Sampler) and the online fit (Optimizer/Objective) to build a
:class:`~optiserve.modeling.parametric.ParamFunction` latency-vs-memory curve for
one deployed Lambda, optionally per ML-model variant.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import boto3

from optiserve.logging import get_logger
from optiserve.modeling.fitting import Objective, Optimizer
from optiserve.modeling.parametric import ParamFunction
from optiserve.profiling.explorer import Explorer
from optiserve.profiling.sampler import Sampler

logger = get_logger(__name__)


class FunctionPerformanceModeling:
    def __init__(
        self,
        function_name: str,
        max_invocations: int = 5,
        memory_bounds: Union[Tuple[int, int], List[Tuple[int, int]]] = (128, 3009),
        region_name: str = "us-east-1",
        knowledge_termination_threshold: int = 3,
        profiling_iterations: int = 4,
        max_total_sample_count: int = 20,
        payload: str = '{"key1": "value1"}',
        available_models: Optional[List[str]] = None,
        memory_space_step: int = 1,
    ):
        if not function_name:
            raise ValueError("Function name is required.")

        # Default the model list BEFORE building the Explorer, so the memory
        # spaces are keyed correctly (previously this ran after, leaving the
        # memory spaces empty for the list-of-bounds + no-models case).
        if available_models is None:
            available_models = ["None"]
        self.available_models: List[str] = available_models

        self.explorer = Explorer(
            function_name=function_name,
            max_invocations=max_invocations,
            memory_bounds=memory_bounds,
            boto_session=boto3.Session(region_name=region_name),
            payload=payload,
            available_models=available_models,
            memory_space_step=memory_space_step,
        )

        self.param_functions: Dict[str, ParamFunction] = {
            model_name: ParamFunction() for model_name in available_models
        }

        self.objectives: Dict[str, Objective] = {
            model_name: Objective(
                param_function=self.param_functions[model_name],
                memory_space=self.explorer.memory_spaces[model_name],
                termination_threshold=knowledge_termination_threshold,
            )
            for model_name in available_models
        }

        self.sampler = Sampler(
            explorer=self.explorer, profiling_iterations=profiling_iterations
        )
        self.optimizer = Optimizer(
            objectives=self.objectives,
            sampler=self.sampler,
            max_total_sample_count=max_total_sample_count,
        )
        self._explored: Dict[str, bool] = defaultdict(lambda: False)

    def _resolve(self, model_name: Optional[str]) -> str:
        return self.available_models[0] if model_name is None else model_name

    def run(self, model_name: Optional[str] = None) -> None:
        model_name = self._resolve(model_name)
        if not self._explored[model_name]:
            self.optimizer.start(model_name=model_name)
            self._explored[model_name] = True

    def get_optimal_memory(
        self,
        latency_constraint_threshold_ms: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        models = self.available_models if model_name is None else [model_name]
        results = {}
        for name in models:
            self.run(model_name=name)
            results[name] = self.param_functions[name].minimize(
                self.explorer.memory_spaces[name],
                latency_constraint_threshold_ms=latency_constraint_threshold_ms,
            )
        return next(iter(results.values())) if len(results) == 1 else results

    def get_performance_model(self, model_name: Optional[str] = None) -> ParamFunction:
        model_name = self._resolve(model_name)
        self.run(model_name=model_name)
        return self.param_functions[model_name]

    def get_performance(
        self, memory_mb: float, model_name: Optional[str] = None
    ) -> float:
        """Predicted execution time (ms) at ``memory_mb`` for ``model_name``.

        (Previously this called a non-existent ``get_performance_model_as_function``
        and raised AttributeError.)"""
        return self.get_performance_model(model_name=model_name)(memory_mb)
