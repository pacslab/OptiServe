import math
import numpy as np


from src.profiler.explorer import Explorer
from src.utils.logger import logger
from src.exceptions import (
    SamplingError,
    NotEnoughMemory,
    NoMemoryLeft,
)
from src.utils.sample import Sample
from src.utils.exploration import Exploration
from typing import Optional, List, Dict


class Sampler:
    def __init__(
        self,
        explorer: Explorer,
        profiling_iterations: int,
    ):
        self.explorations: Dict[str, Exploration] = {}
        self.explorer = explorer
        self.memory_spaces = explorer.memory_spaces
        self._profiling_iterations = profiling_iterations

    def exploration_init(self, model_name: str = "None"):
        if model_name not in self.explorer.memory_spaces:
            raise ValueError(
                f"Model name '{model_name}' is not available in the memory spaces."
            )

        memory_space = self.explorer.memory_spaces[model_name]
        self.explorations[model_name] = Exploration()

        self._explore_first_config(model_name=model_name)

        index = math.ceil(len(memory_space) / 3)

        for memory in [memory_space[index], memory_space[-1]]:
            try:
                self.update_exploration(memory_mb=memory, model_name=model_name)

            except SamplingError as e:
                logger.error(e)
                raise

    def _explore_first_config(self, model_name: str = "None"):
        if model_name not in self.explorer.memory_spaces:
            raise ValueError(
                f"Model name '{model_name}' is not available in the memory spaces."
            )

        memory_space = self.explorer.memory_spaces[model_name]

        while len(memory_space) >= 3:
            try:
                self.update_exploration(
                    memory_mb=int(memory_space[0]), model_name=model_name
                )

            except NotEnoughMemory as e:
                logger.info(
                    f"Trying with new memories. {self.explorer.invoker._function_name}: {memory_space[0]}MB for model: {model_name}"
                )
                self.memory_spaces[model_name] = np.array(
                    [mem for mem in memory_space if mem >= memory_space[0] + 128],
                    dtype=int,
                )

            except SamplingError as e:
                logger.error(e)
                raise

            else:
                break

        if len(memory_space) <= 3:
            raise NoMemoryLeft()

    def update_exploration(self, memory_mb: int, model_name: str = "None"):
        logger.info(
            f"Exploring memory configuration: {memory_mb} MB for {self.explorer.invoker._function_name} for model: {model_name}"
        )
        try:
            durations = self.explorer.explore_multi_threading(
                num_of_invocations=self._profiling_iterations,
                num_of_threads=self._profiling_iterations,
                memory_mb=memory_mb,
                model_name=model_name,
            )

        except SamplingError as e:
            logger.error(e)
            raise

        durations = self._explore_dynamically(durations=durations)

        subsample = [
            Sample(memory_mb=memory_mb, duration_ms=duration) for duration in durations
        ]

        if model_name not in self.explorations or self.explorations[model_name] is None:
            raise ValueError(
                "Exploration object is not initialized. Call exploration_init() first."
            )
        self.explorations[model_name].add_sample(subsample)

        logger.info(
            f"Finished exploring memory configuration: {memory_mb} MB for {self.explorer.invoker._function_name}: {durations} ms for model: {model_name}"
        )

    def _explore_dynamically(self, durations: list):
        if len(durations) < self._profiling_iterations:
            raise ValueError(
                "The number of durations is less than the number of profiling iterations."
            )

        dynamic_exploration_count = 0
        min_cv = np.std(durations, ddof=1) / np.mean(durations)

        while dynamic_exploration_count < 8 and min_cv > 0.05:
            try:
                result = self.explorer._explore()

            except SamplingError as e:
                logger.error(e)
                raise

            dynamic_exploration_count += 1

            values = durations.copy()

            for i in range(len(durations)):
                value = values[i]
                values[i] = result
                cv = np.std(values, ddof=1) / np.mean(values)

                if min_cv > cv:
                    min_cv = cv
                    durations = values.copy()

                values[i] = value

        return durations
