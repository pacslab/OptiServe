"""Function profiling orchestration.

:class:`Explorer` sets a Lambda's memory/model configuration, forces a cold
start, then invokes it repeatedly (concurrently) and returns the billed
durations. It owns the per-model memory search spaces.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import boto3
import numpy as np
from tqdm import tqdm

from optiserve.aws.lambda_client import ConfigManager, Invoker
from optiserve.aws.log_parser import LogParser
from optiserve.exceptions import InvocationError
from optiserve.logging import get_logger

logger = get_logger(__name__)


class Explorer:
    def __init__(
        self,
        function_name: str,
        max_invocations: int,
        boto_session: boto3.Session,
        payload: Optional[str] = None,
        memory_bounds: Union[Tuple[int, int], List[Tuple[int, int]]] = (128, 3009),
        available_models: Optional[List[str]] = None,
        memory_space_step: int = 1,
    ):
        self.function_name = function_name
        self.available_models = available_models
        self.payload = payload
        self.memory_bounds = memory_bounds

        self.log_parser = LogParser()
        self.config_manager = ConfigManager(
            function_name=function_name, boto_session=boto_session
        )
        self.invoker = Invoker(
            function_name=function_name,
            max_invocations=max_invocations,
            boto_session=boto_session,
        )

        self.memory_spaces: Dict[str, np.ndarray] = self._build_memory_spaces(
            memory_bounds, available_models, memory_space_step
        )

    @staticmethod
    def _build_memory_spaces(
        memory_bounds, available_models, step
    ) -> Dict[str, np.ndarray]:
        def space(bounds: Tuple[int, int]) -> np.ndarray:
            return np.array(
                sorted(set(range(bounds[0], bounds[1], step))), dtype=int
            )

        if isinstance(memory_bounds, list) and available_models is not None:
            return {
                model: space(bounds)
                for model, bounds in zip(available_models, memory_bounds)
            }
        if isinstance(memory_bounds, tuple):
            return {"None": space(memory_bounds)}
        return {}

    def _invoke_once(self) -> Optional[float]:
        """Invoke the function once and return its billed duration (ms)."""
        if self.payload is None:
            raise InvocationError("No payload provided.")
        try:
            exec_log = self.invoker.invoke_to_get_duration(payload=self.payload)
            return self.log_parser.parse_function_execution_time(log=exec_log)
        except InvocationError as exc:
            logger.error(exc)
            raise

    def _configure(self, memory_mb: Optional[int], model_name: Optional[str]) -> None:
        """Apply a memory/model configuration and force a cold start with a
        throwaway warm-up invocation."""
        self.config_manager.set_config(memory_mb=memory_mb, model_name=model_name)
        self._invoke_once()  # cold-start warm-up (result discarded)

    def _explore(
        self,
        memory_mb: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Optional[float]:
        if memory_mb is not None or model_name is not None:
            self._configure(memory_mb, model_name)
        return self._invoke_once()

    def explore_multi_threading(
        self,
        num_of_invocations: int,
        num_of_threads: int,
        memory_mb: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> List[float]:
        if memory_mb is not None or model_name is not None:
            self._configure(memory_mb, model_name)

        results: List[float] = []
        first_error: Optional[Exception] = None
        with ThreadPoolExecutor(max_workers=num_of_threads) as executor:
            futures = [
                executor.submit(self._invoke_once) for _ in range(num_of_invocations)
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - re-raised after draining
                    logger.error(exc)
                    if first_error is None:
                        first_error = exc

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
