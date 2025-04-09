import numpy as np
from typing import Dict
from src.optimizer.objective import Objective
from src.utils.sampler import Sampler
from src.utils.logger import logger
from src.exceptions import (
    NotEnoughMemory,
    NoMemoryLeft,
)


class Optimizer:
    def __init__(
        self,
        objectives: Dict[str, Objective],
        sampler: Sampler,
        max_total_sample_count: int = 30,
    ):
        self.objectives = objectives
        self.sampler = sampler
        self._max_total_sample_count = max_total_sample_count

    def _is_terminated(self, model_name: str):
        sample_count = len(self.sampler.explorations[model_name])
        print("Sample Count: ", sample_count)
        print("Max Sample Count: ", self._max_total_sample_count)
        print("Memories: ", self.sampler.explorations[model_name].memories)
        print("Duration: ", self.sampler.explorations[model_name].durations)
        termination_value = self.objectives[model_name].termination_value
        print(
            "Value and Threshold: ",
            termination_value,
            self.objectives[model_name].termination_threshold,
        )
        return (
            sample_count > self._max_total_sample_count
            or termination_value > self.objectives[model_name].termination_threshold
        )

    def _initialize(self, model_name: str = "None"):
        self.sampler.exploration_init(model_name=model_name)

        exploration = self.sampler.explorations[model_name]

        print(f"Explored memories: {exploration.memories}")

        for memory in set(exploration.memories):
            self.objectives[model_name].update_knowledge(memory)

        try:
            self.objectives[model_name].param_function.fit(exploration)
        except RuntimeError as e:
            logger.error(e.args[0])
            raise RuntimeError("Could not fit the parametric function.")

    def _update(self, memory_mb: int, model_name: str):
        try:
            self.sampler.update_exploration(memory_mb, model_name=model_name)
        except NotEnoughMemory as e:
            logger.error(
                f"Trying with new memories. {self.sampler.explorer.invoker._function_name}: {memory_mb}MB"
            )
            self.memory_space = np.array(
                [
                    mem
                    for mem in self.sampler.memory_spaces[model_name]
                    if mem >= self.sampler.memory_spaces[model_name][0] + 128
                ],
                dtype=int,
            )
            return

        self.objectives[model_name].update_knowledge(memory_mb)
        try:
            self.objectives[model_name].param_function.fit(
                self.sampler.explorations[model_name]
            )
        except RuntimeError as e:
            logger.error(e.args[0])
            raise RuntimeError("Could not fit the parametric function.")

    def _select_next_memory_to_explore(self, model_name: str):
        exploration_memories = set(self.sampler.explorations[model_name].memories)
        memory_space = set(self.sampler.memory_spaces[model_name])

        remainder_memories = np.array(
            list(memory_space - exploration_memories), dtype=int
        )

        if len(remainder_memories) == 0:
            raise NoMemoryLeft()

        values = self.objectives[model_name].get_values(remainder_memories)

        return remainder_memories[np.argmin(values)]

    def start(self, model_name: str = "None"):
        self._initialize(model_name=model_name)

        while not self._is_terminated(model_name=model_name):
            memory = self._select_next_memory_to_explore(model_name=model_name)
            self._update(memory_mb=memory, model_name=model_name)
