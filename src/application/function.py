import uuid

from src.profiler.function_config import FunctionConfig
from src.profiler.config_manager import ConfigManager
from src.profiler.cost_calculator import CostCalculator
from src.optimizer.parametric_function import ParamFunction

from typing import Optional
from typing import Union, List, Tuple, Dict


class Function:
    def __init__(
        self,
        name: str,
        config: FunctionConfig,
        available_models: Optional[
            list
        ] = None,  # available models should be sorted by accuracy
        memory_bounds: Union[Tuple[int, int], List[Tuple[int, int]]] = (128, 3009),
        performance_modeling: Union[ParamFunction, Dict[str, ParamFunction]] = None,
    ) -> None:

        self.id = uuid.uuid4().hex
        self.name = name
        self.config = config
        self.memory_bounds = memory_bounds
        self.available_models = available_models
        self.normalized_accuracy: Optional[float] = None
        self.available_models = available_models if available_models is not None else []
        if available_models is not None and config.model_name is not None:
            index = available_models.index(config.model_name)
            if index == -1:
                raise ValueError(
                    f"Model {config.model_name} not found in available models."
                )
            self.normalized_accuracy = index / len(available_models)
        if available_models is not None:
            self.available_normalized_accuracy = [
                i / len(available_models) for i in range(1, len(available_models) + 1)
            ]
        else:
            self.available_normalized_accuracy = None
        self.cost_calculator = CostCalculator(function_name=name)
        self.performance_modeling = performance_modeling


    def get_execution_time(self, memory_size: float, model_name: str = None) -> float:
        if self.performance_modeling is None:
            raise ValueError("Performance modeling not set for this function.")

        if model_name not in self.available_models or model_name not in self.performance_modeling:
            raise ValueError(f"Model {model_name} not available for this function.")
        
        return self.performance_modeling[model_name](memory_size) if model_name else self.performance_modeling(memory_size)


    def get_cost(self, memory_size: float, model_name: str = None) -> float:

        execution_time = self.get_execution_time(memory_size, model_name)
        
        return self.cost_calculator.calculate_cost(
            memory_mb=memory_size,
            duration_ms=execution_time,
        )

    def set_performance_modeling(
        self, performance_modeling: Union[ParamFunction, List[ParamFunction]] = None
    ) -> None:
        self.performance_modeling = performance_modeling

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Function) and self.id == value.id

    def __str__(self) -> str:
        return f"Function: {self.name} ID({self.id})"
