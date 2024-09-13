import boto3
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


from .invoker import Invoker
from .cost_calculator import CostCalculator
from .config_manager import ConfigManager
from src.utils.logger import logger
from src.exceptions import InvocationError
from src.analytics.log_parser import LogParser


class Profiler:
    def __init__(
        self,
        function_name: str,
        max_invocations: int,
        boto_session: boto3.Session,
        payload: str = None,
        memory_bounds: tuple = (128, 3009),
    ):
        self.log_parser = LogParser()
        self.config_manager = ConfigManager(
            function_name=function_name,
            boto_session=boto_session
        )
        self.invoker = Invoker(
            function_name=function_name,
            max_invocations=max_invocations,
            boto_session=boto_session
        )
        self.cost_calculator = CostCalculator(
            function_name=function_name,
        )
        self.payload = payload
        self.memory_bounds = memory_bounds
        self.memory_space = np.array(list(set(range(*memory_bounds))), dtype=int)
        
        self.cost = 0
        self._memory_config_mb = 0
        
        
    def _explore(self, memory_mb: int = None, enable_cost_calculation: bool = True):
        if memory_mb:
            self.config_manager.set_config(memory_mb=memory_mb)
            self._memory_config_mb = memory_mb
            
            # Cold start
            self._explore(enable_cost_calculation=enable_cost_calculation)
            
        try:
            exec_log = self.invoker.invoke_to_get_duration(payload=self.payload)
            exec_time = self.log_parser.parse_execution_time(log=exec_log)
            
        except InvocationError as e:
            logger.error(e)
            if enable_cost_calculation:
                self.cost += self.cost_calculator.calculate_cost(
                    memory_mb=self._memory_config_mb,
                    duration_ms=e.duration_ms
                )
            raise
        
        else:
            if enable_cost_calculation:
                self.cost += self.cost_calculator.calculate_cost(
                    memory_mb=self._memory_config_mb,
                    duration_ms=exec_time
                )
            return exec_time
    
    
    def explore_multi_threading(self, num_of_invocations: int, num_of_threads: int, memory_mb: int = None):
        if memory_mb:
            self.config_manager.set_config(memory_mb=memory_mb)
            self._memory_config_mb = memory_mb
            
            # Cold start
            self.explore_multi_threading(num_of_invocations=num_of_invocations, num_of_threads=num_of_threads)
            
        error = None
        results = []
        
        with ThreadPoolExecutor(max_workers=num_of_threads) as executor:
            futures = [
                executor.submit(self._explore, memory_mb=None, enable_cost_calculation=False)
                for _ in range(num_of_invocations)
            ]
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except InvocationError as e:
                    logger.error(e)
                    
                    if error is None:
                        error = e
                    
                    self.cost += self.cost_calculator.calculate_cost(
                        self._memory_config_mb,
                        e.duration_ms
                    )
                    
                    continue
        
        if error:
            raise error
        
        self.cost += np.sum(
            self.cost_calculator.calculate_cost(
                self._memory_config_mb, np.array(results)
            )
        )
        
        return results