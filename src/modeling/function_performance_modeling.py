import boto3
from typing import Optional, Dict, List
from src.profiler.explorer import Explorer
from src.utils.exploration import Exploration
from src.utils.sampler import Sampler
from src.optimizer.objective import Objective
from src.optimizer.optimizer import Optimizer
from src.optimizer.parametric_function import ParamFunction
from collections import defaultdict
from typing import Union, Tuple


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

        self.explorer = Explorer(
            function_name=function_name,
            max_invocations=max_invocations,
            memory_bounds=memory_bounds,
            boto_session=boto3.Session(region_name=region_name),
            payload=payload,
            available_models=available_models,
            memory_space_step=memory_space_step,
        )

        if available_models is None:
            available_models = ["None"]

        self.available_models: List[str] = available_models

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

    def run(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = self.available_models[0]

        if not self._explored[model_name]:
            self.optimizer.start(model_name=model_name)
            self._explored[model_name] = True

    def get_optimal_memory(
        self,
        latency_constraint_threshold_ms: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:

        if model_name is None:
            models_to_run = self.available_models
        else:
            models_to_run = [model_name]

        results = {}

        for model_name in models_to_run:
            if not self._explored[model_name]:
                self.run(model_name=model_name)

            results[model_name] = self.param_functions[model_name].minimize(
                self.explorer.memory_spaces[model_name],
                latency_constraint_threshold_ms=latency_constraint_threshold_ms,
            )

        if len(results) == 1:
            return next(iter(results.values()))
        else:
            return results
        
    
    def get_performance_model(
        self,
        model_name: Optional[str] = None,
    ) -> ParamFunction:
        if model_name is None:
            model_name = self.available_models[0]

        if not self._explored[model_name]:
            self.run(model_name=model_name)

        return self.param_functions[model_name]


    def get_performance(
        self,
        memory_mb: float,
        model_name: Optional[str] = None,
    ) -> float:
        if model_name is None:
            model_name = self.available_models[0]

        return self.get_performance_model_as_function(
            model_name=model_name,
        )(memory_mb)
