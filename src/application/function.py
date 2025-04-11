import uuid
from src.profiler.function_config import FunctionConfig
from src.profiler.config_manager import ConfigManager
from src.profiler.cost_calculator import CostCalculator
from src.modeling.function_performance_modeling import FunctionPerformanceModeling
from typing import Optional
from typing import Union, List, Tuple


class Function:
    def __init__(
        self,
        name: str,
        config: FunctionConfig,
        available_models: Optional[
            list
        ] = None,  # available models should be sorted by accuracy
        memory_bounds: Union[Tuple[int, int], List[Tuple[int, int]]] = (128, 3009),
    ) -> None:

        self.id = uuid.uuid4().hex
        self.name = name
        self.config = config
        self.memory_bounds = memory_bounds
        self.available_models = available_models
        self.normalized_accuracy: Optional[float] = None
        if available_models is not None and config.model_name is not None:
            index = available_models.index(config.model_name)
            if index == -1:
                raise ValueError(
                    f"Model {config.model_name} not found in available models."
                )
            self.normalized_accuracy = index / len(available_models)
        if available_models is not None:
            self.available_normalized_accuracy = [
                i / len(available_models) for i in range(len(available_models))
            ]
        else:
            self.available_normalized_accuracy = None
        self.cost_calculator = CostCalculator(function_name=name)
        self.performance_modeling: Optional[FunctionPerformanceModeling] = None

    @property
    def execution_time(self) -> float:
        return 0.0  # TODO: Calculate execution time based on memory size and model name

    @property
    def cost(self) -> float:
        return self.cost_calculator.calculate_cost(
            memory_mb=self.config.memory_mb,
            duration_ms=self.execution_time,
        )

    def set_performance_modeling(
        self, performance_modeling: FunctionPerformanceModeling
    ) -> None:
        self.performance_modeling = performance_modeling

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Function) and self.id == value.id

    def __str__(self) -> str:
        return f"Function: {self.name} ID({self.id})"

    @property
    def weight_factor(self):
        if self.performance_modeling is None:
            raise ValueError("Performance modeling not set for this function.")
        return self.performance_modeling.get_performance(
            memory_mb=self.config.memory_mb, model_name=self.config.model_name
        )

    @property
    def min_accuracy_model(self) -> Optional[str]:
        return self.available_models[0] if self.available_models else None

    def get_min_memory(self, model_name: Optional[str]) -> int:
        if isinstance(self.memory_bounds, tuple):
            return self.memory_bounds[0]

        if self.available_models is not None and model_name is not None:
            index = self.available_models.index(model_name)
            if index == -1:
                raise ValueError(f"Model {model_name} not found in available models.")

            if isinstance(self.memory_bounds, list):
                return self.memory_bounds[index][0]
            else:
                return self.memory_bounds[0]
        else:
            raise ValueError("Memory bounds not set or model name is None.")

    def get_max_memory(self, model_name: Optional[str]) -> int:
        if isinstance(self.memory_bounds, tuple):
            return self.memory_bounds[1]

        if self.available_models is not None and model_name is not None:
            index = self.available_models.index(model_name)
            if index == -1:
                raise ValueError(f"Model {model_name} not found in available models.")

            if isinstance(self.memory_bounds, list):
                return self.memory_bounds[index][1]
            else:
                return self.memory_bounds[1]
        else:
            raise ValueError("Memory bounds not set or model name is None.")

    def set_config(self, memory_mb: int, model_name: Optional[str] = None) -> None:
        if model_name is None:
            self.config.memory_mb = memory_mb

        if self.available_models is not None and model_name is not None:
            index = self.available_models.index(model_name)
            if index == -1:
                raise ValueError(f"Model {model_name} not found in available models.")
            if isinstance(self.memory_bounds, list):
                mem_bounds = self.memory_bounds[index]
            else:
                mem_bounds = self.memory_bounds

            if memory_mb < mem_bounds[0] or memory_mb > mem_bounds[1]:
                raise ValueError(
                    f"Memory {memory_mb}MB is out of bounds for model {model_name}."
                )

            self.normalized_accuracy = index / len(self.available_models)
            self.config.memory_mb = memory_mb
            self.config.model_name = model_name
