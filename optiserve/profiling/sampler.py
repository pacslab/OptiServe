"""Adaptive memory-configuration sampling.

:class:`Sampler` drives the :class:`~optiserve.profiling.explorer.Explorer` to
collect duration samples across memory sizes, seeding a few points and adding
targeted samples to keep the per-configuration measurement stable.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from optiserve.exceptions import NoMemoryLeft, NotEnoughMemory, SamplingError
from optiserve.logging import get_logger
from optiserve.profiling.explorer import Explorer
from optiserve.profiling.sample import Exploration, Sample

logger = get_logger(__name__)


class Sampler:
    def __init__(
        self,
        explorer: Explorer,
        profiling_iterations: int,
        *,
        memory_floor_step_mb: int = 128,
        cv_threshold: float = 0.05,
        max_dynamic_samples: int = 8,
    ):
        self.explorations: Dict[str, Exploration] = {}
        self.explorer = explorer
        # Shared by reference with the Explorer: pruning here removes infeasible
        # small memories from the search space the optimizer also sees.
        self.memory_spaces = explorer.memory_spaces
        self._profiling_iterations = profiling_iterations
        self._memory_floor_step_mb = memory_floor_step_mb
        self._cv_threshold = cv_threshold
        self._max_dynamic_samples = max_dynamic_samples

    def exploration_init(self, model_name: str = "None") -> None:
        if model_name not in self.explorer.memory_spaces:
            raise ValueError(f"Model '{model_name}' is not in the memory spaces.")

        self.explorations[model_name] = Exploration()
        self._explore_first_config(model_name=model_name)

        # Re-read the space AFTER _explore_first_config, which may have pruned it.
        memory_space = self.memory_spaces[model_name]
        index = math.ceil(len(memory_space) / 3)
        for memory in [memory_space[index], memory_space[-1]]:
            try:
                self.update_exploration(memory_mb=int(memory), model_name=model_name)
            except SamplingError as exc:
                logger.error(exc)
                raise

    def _explore_first_config(self, model_name: str = "None") -> None:
        """Explore the smallest feasible memory, raising the floor whenever the
        current smallest is too small to run the function."""
        if model_name not in self.explorer.memory_spaces:
            raise ValueError(f"Model '{model_name}' is not in the memory spaces.")

        while True:
            memory_space = self.memory_spaces[model_name]
            if len(memory_space) < 3:
                raise NoMemoryLeft()
            try:
                self.update_exploration(
                    memory_mb=int(memory_space[0]), model_name=model_name
                )
                return
            except NotEnoughMemory:
                floor = int(memory_space[0]) + self._memory_floor_step_mb
                logger.info(
                    "Raising memory floor to >= %d MB for %s (model=%s)",
                    floor,
                    self.explorer.function_name,
                    model_name,
                )
                self.memory_spaces[model_name] = np.array(
                    [m for m in memory_space if m >= floor], dtype=int
                )
            except SamplingError as exc:
                logger.error(exc)
                raise

    def update_exploration(self, memory_mb: int, model_name: str = "None") -> None:
        logger.info(
            "Exploring %d MB for %s (model=%s)",
            memory_mb,
            self.explorer.function_name,
            model_name,
        )
        try:
            durations = self.explorer.explore_multi_threading(
                num_of_invocations=self._profiling_iterations,
                num_of_threads=self._profiling_iterations,
                memory_mb=memory_mb,
                model_name=model_name,
            )
        except SamplingError as exc:
            logger.error(exc)
            raise

        durations = self._explore_dynamically(durations=durations)

        subsample = [
            Sample(memory_mb=memory_mb, duration_ms=d) for d in durations
        ]
        if model_name not in self.explorations or self.explorations[model_name] is None:
            raise ValueError("Exploration not initialized; call exploration_init().")
        self.explorations[model_name].add_sample(subsample)
        logger.info(
            "Finished %d MB for %s: %s ms",
            memory_mb,
            self.explorer.function_name,
            durations,
        )

    def _explore_dynamically(self, durations: list) -> list:
        """Reduce measurement noise: while the coefficient of variation exceeds
        the threshold, take extra samples and substitute each in place of the
        position that most reduces the CV.

        NOTE (methodology): this replaces measured durations with lower-variance
        substitutes to stabilize the per-configuration estimate. It is a
        deliberate noise-reduction step carried over from the thesis; confirm it
        matches the intended experimental protocol before relying on the raw
        per-invocation numbers.
        """
        if len(durations) < self._profiling_iterations:
            raise ValueError(
                "Fewer durations than profiling iterations."
            )

        count = 0
        min_cv = np.std(durations, ddof=1) / np.mean(durations)
        while count < self._max_dynamic_samples and min_cv > self._cv_threshold:
            try:
                result = self.explorer._explore()
            except SamplingError as exc:
                logger.error(exc)
                raise
            count += 1

            candidate = durations.copy()
            for i in range(len(durations)):
                original = candidate[i]
                candidate[i] = result
                cv = np.std(candidate, ddof=1) / np.mean(candidate)
                if cv < min_cv:
                    min_cv = cv
                    durations = candidate.copy()
                candidate[i] = original
        return durations
