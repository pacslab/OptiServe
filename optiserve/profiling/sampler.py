"""Adaptive memory-configuration sampling.

:class:`Sampler` drives the :class:`~optiserve.profiling.explorer.Explorer` to
collect duration samples across memory sizes, seeding a few points and adding
targeted samples to keep the per-configuration measurement stable.

Two production concerns are handled here without changing the sampling maths:

* **Resumability.** Pass a
  :class:`~optiserve.profiling.state.CheckpointStore` and every completed
  configuration is persisted, so an interrupted multi-hour run continues instead
  of restarting. The default store remembers nothing, so callers that do not opt
  in get byte-identical behaviour.
* **Observability.** Every sample, prune and resume emits an event through the
  hook registry (see :mod:`optiserve.observability`), which is how a finished
  run can be explained after the fact.
"""

from __future__ import annotations

import math

import numpy as np

from optiserve.exceptions import NoMemoryLeft, NotEnoughMemory, SamplingError
from optiserve.logging import get_logger
from optiserve.observability.events import EventName
from optiserve.observability.hooks import HookRegistry
from optiserve.observability.hooks import hooks as default_hooks
from optiserve.profiling.explorer import Explorer
from optiserve.profiling.sample import Exploration, Sample
from optiserve.profiling.state import (
    CheckpointStore,
    NullCheckpointStore,
    ProfilingState,
    run_id_for,
)

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
        noise_reduction: bool = True,
        checkpoint_store: CheckpointStore | None = None,
        hooks: HookRegistry | None = None,
    ):
        self.explorations: dict[str, Exploration] = {}
        self.explorer = explorer
        # Shared by reference with the Explorer: pruning here removes infeasible
        # small memories from the search space the optimizer also sees.
        self.memory_spaces = explorer.memory_spaces
        self._profiling_iterations = profiling_iterations
        self._memory_floor_step_mb = memory_floor_step_mb
        self._cv_threshold = cv_threshold
        self._max_dynamic_samples = max_dynamic_samples
        self._noise_reduction = noise_reduction
        self._checkpoints: CheckpointStore = checkpoint_store or NullCheckpointStore()
        self._hooks = hooks if hooks is not None else default_hooks
        self._run_ids: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def run_id(self, model_name: str) -> str:
        """Stable id for this (function, model, parameters) run."""
        if model_name not in self._run_ids:
            self._run_ids[model_name] = run_id_for(
                self.explorer.function_name,
                model_name,
                profiling_iterations=self._profiling_iterations,
                cv_threshold=self._cv_threshold,
                max_dynamic_samples=self._max_dynamic_samples,
                payload=getattr(self.explorer, "payload", None),
            )
        return self._run_ids[model_name]

    def _checkpoint(self, model_name: str) -> None:
        exploration = self.explorations.get(model_name)
        if exploration is None:
            return
        state = ProfilingState(
            run_id=self.run_id(model_name),
            function_name=self.explorer.function_name,
            model_name=model_name,
            samples=[
                Sample(memory_mb=int(m), duration_ms=float(d))
                for m, d in zip(exploration.memories, exploration.durations, strict=True)
            ],
            memory_space=[int(m) for m in self.memory_spaces[model_name]],
        )
        self._checkpoints.save(state)
        self._hooks.emit(
            EventName.CHECKPOINT_SAVED,
            function=self.explorer.function_name,
            model=model_name,
            run_id=state.run_id,
            samples=len(state.samples),
        )

    def _try_resume(self, model_name: str) -> bool:
        """Restore a previous run's samples and pruned memory space.

        Returns ``True`` when the run was resumed, in which case the caller
        skips the (expensive) seeding phase.
        """
        state = self._checkpoints.load(self.run_id(model_name))
        if state is None or not state.samples:
            return False

        self.explorations[model_name] = Exploration(list(state.samples))
        if state.memory_space:
            self.memory_spaces[model_name] = np.array(state.memory_space, dtype=int)
        logger.info(
            "Resumed profiling of %s (model=%s) from %d samples",
            self.explorer.function_name,
            model_name,
            len(state.samples),
        )
        self._hooks.emit(
            EventName.CHECKPOINT_RESUMED,
            function=self.explorer.function_name,
            model=model_name,
            run_id=state.run_id,
            samples=len(state.samples),
        )
        return True

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def exploration_init(self, model_name: str = "None") -> None:
        if model_name not in self.explorer.memory_spaces:
            raise ValueError(f"Model '{model_name}' is not in the memory spaces.")

        if self._try_resume(model_name):
            return

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
                self.update_exploration(memory_mb=int(memory_space[0]), model_name=model_name)
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
                self._hooks.emit(
                    EventName.MEMORY_PRUNED,
                    function=self.explorer.function_name,
                    model=model_name,
                    floor_mb=floor,
                    remaining=len(self.memory_spaces[model_name]),
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

        subsample = [Sample(memory_mb=memory_mb, duration_ms=d) for d in durations]
        if model_name not in self.explorations or self.explorations[model_name] is None:
            raise ValueError("Exploration not initialized; call exploration_init().")
        self.explorations[model_name].add_sample(subsample)
        logger.info(
            "Finished %d MB for %s: %s ms",
            memory_mb,
            self.explorer.function_name,
            durations,
        )
        self._hooks.emit(
            EventName.SAMPLE_RECORDED,
            function=self.explorer.function_name,
            model=model_name,
            memory_mb=int(memory_mb),
            observations=len(durations),
            mean_duration_ms=float(np.mean(durations)) if len(durations) else None,
        )
        self._checkpoint(model_name)

    def _explore_dynamically(self, durations: list) -> list:
        """Reduce measurement noise: while the coefficient of variation exceeds
        the threshold, take extra samples and substitute each in place of the
        position that most reduces the CV.

        METHODOLOGY WARNING. This is selection on the outcome variable, not
        noise reduction. Each extra invocation is kept only if substituting it
        for one of the existing observations *lowers* the coefficient of
        variation, so the retained sample is conditioned on the value it takes.
        Over 4 000 simulated replications the step removes roughly 74 % of the
        sample variance and shifts the retained mean by about +1.3 %: the fitted
        curve therefore looks far more precise than the measurements justify,
        and is slightly biased. It is enabled by default because the published
        results were produced with it — set
        ``ProfilingConfig(noise_reduction=False)`` for any new measurement
        campaign, and state which setting a result was produced under.
        """
        if not self._noise_reduction:
            return durations

        if len(durations) < self._profiling_iterations:
            raise ValueError("Fewer durations than profiling iterations.")

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
