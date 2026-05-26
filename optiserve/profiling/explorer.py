"""Function profiling orchestration.

:class:`Explorer` sets a Lambda's memory/model configuration, forces a cold
start, then invokes it repeatedly (concurrently) and returns the billed
durations. It owns the per-model memory search spaces.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
from tqdm import tqdm

from optiserve.aws.lambda_client import ConfigManager, Invoker
from optiserve.aws.log_parser import LogParser
from optiserve.exceptions import InvocationError
from optiserve.logging import get_logger
from optiserve.observability.events import EventName
from optiserve.observability.hooks import HookRegistry
from optiserve.observability.hooks import hooks as default_hooks

logger = get_logger(__name__)

#: Lambda's current maximum configurable memory. Configurations above this are
#: rejected by the API, so a search space must never contain one.
LAMBDA_MAX_MEMORY_MB = 10240
#: Lambda's minimum configurable memory.
LAMBDA_MIN_MEMORY_MB = 128


class Explorer:
    def __init__(
        self,
        function_name: str,
        max_invocations: int,
        boto_session: boto3.Session,
        payload: str | None = None,
        memory_bounds: tuple[int, int] | list[tuple[int, int]] = (128, 3008),
        available_models: list[str] | None = None,
        memory_space_step: int = 1,
        *,
        hooks: HookRegistry | None = None,
    ):
        self.function_name = function_name
        self.available_models = available_models
        self.payload = payload
        self.memory_bounds = memory_bounds
        self._hooks = hooks if hooks is not None else default_hooks

        self.log_parser = LogParser()
        self.config_manager = ConfigManager(
            function_name=function_name, boto_session=boto_session, hooks=self._hooks
        )
        self.invoker = Invoker(
            function_name=function_name,
            max_invocations=max_invocations,
            boto_session=boto_session,
            # Read the configuration through the ConfigManager's cache: without
            # this, every invocation costs an extra GetFunctionConfiguration on
            # the control plane, purely to build a log line.
            config_provider=self.config_manager.current_config,
            hooks=self._hooks,
        )

        self.memory_spaces: dict[str, np.ndarray] = self._build_memory_spaces(
            memory_bounds, available_models, memory_space_step
        )

    @staticmethod
    def _build_memory_spaces(
        memory_bounds: tuple[int, int] | list[tuple[int, int]],
        available_models: list[str] | None,
        step: int,
    ) -> dict[str, np.ndarray]:
        def space(bounds: tuple[int, int]) -> np.ndarray:
            low, high = int(bounds[0]), int(bounds[1])
            if low < LAMBDA_MIN_MEMORY_MB:
                raise ValueError(
                    f"memory lower bound {low} is below Lambda's minimum "
                    f"({LAMBDA_MIN_MEMORY_MB} MB)."
                )
            if high > LAMBDA_MAX_MEMORY_MB:
                raise ValueError(
                    f"memory upper bound {high} exceeds Lambda's maximum "
                    f"({LAMBDA_MAX_MEMORY_MB} MB)."
                )
            if high < low:
                raise ValueError(f"memory bounds {bounds} are inverted.")
            # Inclusive of `high`: `range` excludes its stop value, so the
            # caller's stated maximum memory was silently never profiled. The
            # thesis default of (128, 3008) reads as "up to 3008" precisely
            # because of that off-by-one; with an inclusive bound (128, 3008)
            # now means what it says.
            return np.array(sorted(set(range(low, high + 1, step))), dtype=int)

        if isinstance(memory_bounds, list) and available_models is not None:
            return {
                model: space(bounds)
                for model, bounds in zip(available_models, memory_bounds, strict=False)
            }
        if isinstance(memory_bounds, tuple):
            return {"None": space(memory_bounds)}
        return {}

    def _invoke_once(self) -> float:
        """Invoke the function once and return its billed duration (ms)."""
        if self.payload is None:
            raise InvocationError("No payload provided.")
        try:
            exec_log = self.invoker.invoke_to_get_duration(payload=self.payload)
            duration = self.log_parser.parse_function_execution_time(log=exec_log)
        except InvocationError as exc:
            logger.error(exc)
            self._hooks.emit(
                EventName.INVOCATION_FAILED,
                function=self.function_name,
                reason=type(exc).__name__,
            )
            raise

        if duration is None:
            # parse_function_execution_time returns the billed duration, or the
            # duration carried on an application error — which is None when the
            # log had no REPORT to read it from. Letting that through would put
            # a None into the sample list, where numpy turns the whole
            # exploration into NaN and the curve fit fails far from the cause.
            raise InvocationError(
                "Invocation produced no billed duration; the log had no parsable REPORT line."
            )

        self._hooks.emit(
            EventName.INVOCATION_COMPLETED,
            function=self.function_name,
            duration_ms=float(duration),
        )
        return float(duration)

    def _configure(self, memory_mb: int | None, model_name: str | None) -> None:
        """Apply a memory/model configuration and force a cold start with a
        throwaway warm-up invocation."""
        self.config_manager.set_config(memory_mb=memory_mb, model_name=model_name)
        self._invoke_once()  # cold-start warm-up (result discarded)

    def _explore(
        self,
        memory_mb: int | None = None,
        model_name: str | None = None,
    ) -> float:
        if memory_mb is not None or model_name is not None:
            self._configure(memory_mb, model_name)
        return self._invoke_once()

    def explore_multi_threading(
        self,
        num_of_invocations: int,
        num_of_threads: int,
        memory_mb: int | None = None,
        model_name: str | None = None,
    ) -> list[float]:
        if memory_mb is not None or model_name is not None:
            self._configure(memory_mb, model_name)

        results: list[float] = []
        first_error: Exception | None = None
        with ThreadPoolExecutor(max_workers=num_of_threads) as executor:
            futures = [executor.submit(self._invoke_once) for _ in range(num_of_invocations)]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error(exc)
                    first_error = _preferred_error(first_error, exc)

        if first_error is not None:
            raise first_error
        return results

    def explore_all_memories(self, num_of_invocations: int) -> None:
        """Brute-force sweep every memory in every model space (results are not
        returned — data is harvested afterward from CloudWatch)."""
        for model_name, memory_space in self.memory_spaces.items():
            for memory_mb in tqdm(
                memory_space,
                desc="Processing",
                bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]",
            ):
                self.explore_multi_threading(
                    num_of_invocations, num_of_invocations, int(memory_mb), model_name
                )


def _preferred_error(current: Exception | None, candidate: Exception) -> Exception:
    """Pick which of two concurrent invocation errors to propagate.

    ``as_completed`` yields in completion order, so "first error seen" was a
    thread race: with one thread hitting an out-of-memory condition and another
    a transient throttle, whichever finished first decided whether the sampler
    *pruned the memory size* or *aborted the run*. Those are very different
    outcomes to leave to scheduling order.

    ``NotEnoughMemory`` (and its ``FunctionTimeout`` subclass) is a statement
    about the configuration under test and is therefore preferred: it is the
    signal the sampler acts on, and it stays true regardless of what the other
    threads saw.
    """
    from optiserve.exceptions import NotEnoughMemory

    if current is None:
        return candidate
    if isinstance(candidate, NotEnoughMemory) and not isinstance(current, NotEnoughMemory):
        return candidate
    return current
