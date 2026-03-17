"""Online active-learning fit of a :class:`~optiserve.modeling.parametric.ParamFunction`.

- :class:`Objective` is the acquisition function: it combines modeled cost with
  a Gaussian "knowledge" penalty so already-sampled memory regions are
  deprioritized, and defines the confidence-based termination signal.
- :class:`Optimizer` runs the sample→fit→select loop until the model is
  confident around the cost optimum (or a sample budget is hit).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.stats as stats

from optiserve.exceptions import NoMemoryLeft, NotEnoughMemory
from optiserve.logging import get_logger
from optiserve.modeling.parametric import ParamFunction
from optiserve.profiling.sampler import Sampler

logger = get_logger(__name__)


class Objective:
    def __init__(
        self,
        param_function: ParamFunction,
        memory_space: np.ndarray,
        termination_threshold: float = 3,
        model_name: Optional[str] = None,
        knowledge_sigma_mb: float = 200.0,
    ):
        self.param_function = param_function
        self.memory_space = memory_space
        self.knowledge_values = {x: 0 for x in memory_space}
        self.termination_threshold = termination_threshold
        self.model_name = model_name
        self._sigma = knowledge_sigma_mb

    def reset(self) -> None:
        self.param_function.params = None
        self.knowledge_values = {x: 0 for x in self.memory_space}

    def get_knowledge(self, memories: np.ndarray) -> np.ndarray:
        knowledge = np.array([self.knowledge_values[m] for m in memories])
        return 1.0 + knowledge

    def update_knowledge(self, memory_mb: int) -> None:
        for memory in self.knowledge_values:
            self.knowledge_values[memory] += stats.norm.pdf(
                memory, memory_mb, self._sigma
            ) / stats.norm.pdf(memory_mb, memory_mb, self._sigma)

    def get_values(self, memories: np.ndarray) -> np.ndarray:
        real_cost = self.param_function(memories) * memories
        return real_cost * self.get_knowledge(memories)

    @property
    def termination_value(self) -> float:
        knowledge = self.get_knowledge(self.memory_space)
        cost = self.param_function(self.memory_space) * self.memory_space
        return knowledge[np.argmin(cost)]


class Optimizer:
    def __init__(
        self,
        objectives: Dict[str, Objective],
        sampler: Sampler,
        max_total_sample_count: int = 30,
        memory_floor_step_mb: int = 128,
    ):
        self.objectives = objectives
        self.sampler = sampler
        self._max_total_sample_count = max_total_sample_count
        self._memory_floor_step_mb = memory_floor_step_mb

    def _is_terminated(self, model_name: str) -> bool:
        sample_count = len(self.sampler.explorations[model_name])
        termination_value = self.objectives[model_name].termination_value
        logger.debug(
            "samples=%d/%d termination=%.3f/%.3f",
            sample_count,
            self._max_total_sample_count,
            termination_value,
            self.objectives[model_name].termination_threshold,
        )
        return (
            sample_count > self._max_total_sample_count
            or termination_value > self.objectives[model_name].termination_threshold
        )

    def _initialize(self, model_name: str = "None") -> None:
        self.sampler.exploration_init(model_name=model_name)
        exploration = self.sampler.explorations[model_name]
        for memory in set(exploration.memories):
            self.objectives[model_name].update_knowledge(memory)
        try:
            self.objectives[model_name].param_function.fit(exploration)
        except RuntimeError as exc:
            logger.error(exc.args[0])
            raise RuntimeError("Could not fit the parametric function.")

    def _update(self, memory_mb: int, model_name: str) -> None:
        try:
            self.sampler.update_exploration(memory_mb, model_name=model_name)
        except NotEnoughMemory:
            # This memory is infeasible; prune it (and everything smaller) from
            # the shared search space so it is not selected again. (Previously an
            # unread local was assigned, so the same memory could recur.)
            space = self.sampler.memory_spaces[model_name]
            floor = int(memory_mb) + self._memory_floor_step_mb
            logger.warning(
                "%s: %dMB infeasible; raising floor to >= %dMB",
                self.sampler.explorer.function_name,
                memory_mb,
                floor,
            )
            self.sampler.memory_spaces[model_name] = np.array(
                [m for m in space if m >= floor], dtype=int
            )
            return

        self.objectives[model_name].update_knowledge(memory_mb)
        try:
            self.objectives[model_name].param_function.fit(
                self.sampler.explorations[model_name]
            )
        except RuntimeError as exc:
            logger.error(exc.args[0])
            raise RuntimeError("Could not fit the parametric function.")

    def _select_next_memory_to_explore(self, model_name: str) -> int:
        explored = set(self.sampler.explorations[model_name].memories)
        memory_space = set(self.sampler.memory_spaces[model_name])
        remainder = np.array(list(memory_space - explored), dtype=int)
        if len(remainder) == 0:
            raise NoMemoryLeft()
        values = self.objectives[model_name].get_values(remainder)
        return remainder[np.argmin(values)]

    def start(self, model_name: str = "None") -> None:
        self._initialize(model_name=model_name)
        while not self._is_terminated(model_name=model_name):
            memory = self._select_next_memory_to_explore(model_name=model_name)
            self._update(memory_mb=memory, model_name=model_name)
