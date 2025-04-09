import numpy as np
import scipy.stats as stats
from src.optimizer.parametric_function import ParamFunction
from typing import Optional


class Objective:
    def __init__(
        self,
        param_function: ParamFunction,
        memory_space: np.ndarray,
        termination_threshold: float = 3,
        model_name: Optional[str] = None,
    ):
        self.param_function = param_function
        self.memory_space = memory_space
        self.knowledge_values = {x: 0 for x in memory_space}
        self.termination_threshold = termination_threshold
        self.model_name = model_name

    def reset(self):
        self.param_function.params = None
        self.knowledge_values = {x: 0 for x in self.memory_space}

    def get_knowledge(self, memories: np.ndarray):
        knowledge = np.array([self.knowledge_values[memory] for memory in memories])
        return 1.0 + knowledge

    def update_knowledge(self, memory_mb: int):
        for memory in self.knowledge_values:
            self.knowledge_values[memory] += stats.norm.pdf(
                memory, memory_mb, 200
            ) / stats.norm.pdf(memory_mb, memory_mb, 200)

    def get_values(self, memories: np.ndarray):
        real_cost = self.param_function(memories) * memories
        knowledge = self.get_knowledge(memories)
        return real_cost * knowledge

    @property
    def termination_value(self):
        knowledge_values = self.get_knowledge(self.memory_space)
        y = self.param_function(self.memory_space) * self.memory_space
        return knowledge_values[np.argmin(y)]
