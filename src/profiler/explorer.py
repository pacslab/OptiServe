import boto3
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from src.profiler.invoker import Invoker
from src.profiler.cost_calculator import CostCalculator
from src.profiler.config_manager import ConfigManager
from src.utils.logger import logger
from src.exceptions import InvocationError
from src.analytics.log_parser import LogParser
from typing import List, Optional, Tuple, Union, Dict


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
        self.available_models = available_models

        self.log_parser = LogParser()
        self.config_manager = ConfigManager(
            function_name=function_name, boto_session=boto_session
        )
        self.invoker = Invoker(
            function_name=function_name,
            max_invocations=max_invocations,
            boto_session=boto_session,
        )
        self.cost_calculator = CostCalculator(
            function_name=function_name,
        )
        self.payload = payload
        self.memory_bounds = memory_bounds
        self.memory_spaces: Dict[str, np.ndarray] = {}
        if isinstance(memory_bounds, List) and available_models is not None:
            for model_name, bounds in zip(available_models, memory_bounds):
                self.memory_spaces[model_name] = np.array(
                    sorted(list(set(range(bounds[0], bounds[1], memory_space_step)))),
                    dtype=int,
                )
        elif isinstance(memory_bounds, Tuple):
            self.memory_spaces["None"] = np.array(
                sorted(list(set(range(memory_bounds[0], memory_bounds[1], memory_space_step)))),
                dtype=int,
            )

        # self._memory_config_mb = 0

    def _explore(
        self,
        memory_mb: Optional[int] = None,
        enable_cost_calculation: bool = True,
        model_name: Optional[str] = None,
    ):
        if memory_mb is not None or model_name is not None:
            self.config_manager.set_config(memory_mb=memory_mb, model_name=model_name)
            self._memory_config_mb = memory_mb

            # Cold start
            self._explore(
                enable_cost_calculation=enable_cost_calculation, model_name=model_name
            )

        try:
            if self.payload is None:
                raise InvocationError("No payload provided.")
            exec_log = self.invoker.invoke_to_get_duration(payload=self.payload)
            exec_time = self.log_parser.parse_function_execution_time(log=exec_log)

        except InvocationError as e:
            logger.error(e)
            # e_duration_ms = e.duration_ms
            # if (
            #     enable_cost_calculation
            #     and self._memory_config_mb is not None
            #     and e_duration_ms is not None
            # ):
            #     self.cost += self.cost_calculator.calculate_cost(
            #         memory_mb=self._memory_config_mb, duration_ms=e_duration_ms
            #     )
            raise

        else:
            # if (
            #     enable_cost_calculation
            #     and self._memory_config_mb is not None
            #     and isinstance(exec_time, (int, float, np.ndarray))
            # ):
            #     self.cost += self.cost_calculator.calculate_cost(
            #         memory_mb=self._memory_config_mb, duration_ms=exec_time
            #     )
            return exec_time

    def explore_multi_threading(
        self,
        num_of_invocations: int,
        num_of_threads: int,
        memory_mb: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        if memory_mb is not None or model_name is not None:
            self.config_manager.set_config(memory_mb=memory_mb, model_name=model_name)
            self._memory_config_mb = memory_mb

            # Cold start
            self.explore_multi_threading(
                num_of_invocations=num_of_invocations, num_of_threads=num_of_threads
            )

        error = None
        results = []

        with ThreadPoolExecutor(max_workers=num_of_threads) as executor:
            futures = [
                executor.submit(
                    self._explore,
                    memory_mb=None,
                    enable_cost_calculation=False,
                )
                for _ in range(num_of_invocations)
            ]

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(e)

                    if error is None:
                        error = e

                    # if self._memory_config_mb is not None and isinstance(
                    #     e, InvocationError
                    # ):
                    #     e_duration_ms = e.duration_ms
                    #     if e_duration_ms is not None:
                    #         self.cost += self.cost_calculator.calculate_cost(
                    #             self._memory_config_mb, e_duration_ms
                    #         )

                    continue

        if error:
            raise error

        # if self._memory_config_mb is not None:
        #     self.cost += np.sum(
        #         self.cost_calculator.calculate_cost(
        #             self._memory_config_mb, np.array(results)
        #         )
        #     )

        return results

    def explore_all_memories(self, num_of_invocations: int):
        for model_name, memory_space in self.memory_spaces.items():
            for memory_mb in tqdm(
                memory_space,
                desc="Processing",
                bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]",
            ):
                _ = self.explore_multi_threading(
                    num_of_invocations, num_of_invocations, memory_mb, model_name
                )
