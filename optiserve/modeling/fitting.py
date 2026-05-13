"""Online active-learning fit of a :class:`~optiserve.modeling.parametric.ParamFunction`.

- :class:`Objective` is the acquisition function: it combines modeled cost with
  a Gaussian "knowledge" penalty so already-sampled memory regions are
  deprioritized, and defines the confidence-based termination signal.
- :class:`Optimizer` runs the sample→fit→select loop until the model is
  confident around the cost optimum (or a sample budget is hit).
"""

from __future__ import annotations

from collections.abc import Callable

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
        model_name: str | None = None,
        knowledge_sigma_mb: float = 200.0,
        *,
        space_provider: Callable[[], np.ndarray] | None = None,
    ):
        self.param_function = param_function
        self._memory_space = np.asarray(memory_space)
        # The sampler prunes infeasible (too-small) memories by *replacing* the
        # array in its `memory_spaces` dict. Holding the original array here
        # meant the acquisition function and the termination check kept scoring
        # memories the sampler had already ruled out — including the argmin the
        # termination threshold is measured at, so the confidence-based stop
        # could never fire. `space_provider` re-reads the live space instead.
        self._space_provider = space_provider
        self.knowledge_values = dict.fromkeys(self._memory_space, 0)
        self.termination_threshold = termination_threshold
        self.model_name = model_name
        self._sigma = knowledge_sigma_mb

    @property
    def memory_space(self) -> np.ndarray:
        if self._space_provider is not None:
            return np.asarray(self._space_provider())
        return self._memory_space

    @memory_space.setter
    def memory_space(self, value: np.ndarray) -> None:
        self._memory_space = np.asarray(value)

    def reset(self) -> None:
        self.param_function.params = None
        self.knowledge_values = dict.fromkeys(self.memory_space, 0)

    def get_knowledge(self, memories: np.ndarray) -> np.ndarray:
        # `.get(m, 0)`: the live memory space is re-read from the sampler, so a
        # memory can appear here that was not present when knowledge_values was
        # built. Unknown memory == no knowledge yet.
        knowledge = np.array([self.knowledge_values.get(m, 0) for m in memories])
        return 1.0 + knowledge

    def update_knowledge(self, memory_mb: int) -> None:
        """Add a Gaussian bump of "we have measured near here" around ``memory_mb``.

        Vectorized: the previous implementation made one scalar
        ``scipy.stats.norm.pdf`` call per memory in the space, i.e. 2 881 calls
        per sample at the default bounds — 0.71 s of pure Python per update
        against 0.0005 s vectorized (~1 300x), repeated for every sample of
        every model variant. The vectorized form is bit-identical (verified by
        ``np.array_equal`` over the full default space), so this is purely a
        cost reduction.
        """
        if not self.knowledge_values:
            return
        memories = np.fromiter(
            self.knowledge_values.keys(), dtype=float, count=len(self.knowledge_values)
        )
        increments = stats.norm.pdf(memories, memory_mb, self._sigma) / stats.norm.pdf(
            memory_mb, memory_mb, self._sigma
        )
        for memory, increment in zip(self.knowledge_values, increments, strict=True):
            self.knowledge_values[memory] += increment

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
        objectives: dict[str, Objective],
        sampler: Sampler,
        max_total_sample_count: int = 30,
        memory_floor_step_mb: int = 128,
    ):
        self.objectives = objectives
        self.sampler = sampler
        self._max_total_sample_count = max_total_sample_count
        self._memory_floor_step_mb = memory_floor_step_mb

    def _is_terminated(self, model_name: str) -> bool:
        # Count distinct *memory configurations*, not individual duration
        # observations. `len(Exploration)` is the number of Samples, which is
        # configurations x profiling_iterations — so the default budget of 20
        # with 4 iterations stopped after 5 configurations, not 20, and the
        # sample budget silently scaled with the iteration count.
        sample_count = len(set(self.sampler.explorations[model_name].memories))
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
            raise RuntimeError("Could not fit the parametric function.") from exc

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
            self.objectives[model_name].param_function.fit(self.sampler.explorations[model_name])
        except RuntimeError as exc:
            logger.error(exc.args[0])
            raise RuntimeError("Could not fit the parametric function.") from exc

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
