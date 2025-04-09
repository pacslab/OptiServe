import uuid
from src.profiler.function_config import FunctionConfig
from src.profiler.cost_calculator import CostCalculator
from typing import Optional


class Function:
    def __init__(
        self,
        name: str,
        config: FunctionConfig,
        available_models: Optional[list] = None,
    ) -> None:

        self.id = uuid.uuid4().hex
        self.name = name
        self.config = config
        self.normalized_accuracy: Optional[float] = None
        if available_models is not None and config.model_name is not None:
            index = available_models.index(config.model_name)
            if index == -1:
                raise ValueError(
                    f"Model {config.model_name} not found in available models."
                )
            self.normalized_accuracy = index / len(available_models)

        self.cost_calculator = CostCalculator(function_name=name)

    @property
    def execution_time(self) -> float:
        return 0.0  # TODO: Calculate execution time based on memory size and model name

    @property
    def cost(self) -> float:
        return self.cost_calculator.calculate_cost(
            memory_mb=self.config.memory_mb,
            duration_ms=self.execution_time,
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Function) and self.id == value.id

    def __str__(self) -> str:
        return f"Function: {self.name} ID({self.id})"
