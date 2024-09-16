import numpy as np
import boto3


from src.profiler.explorer import Explorer
from src.utils.exploration import Exploration
from src.utils.sampler import Sampler
from src.optimizer.objective import Objective
from src.optimizer.optimizer import Optimizer
from src.optimizer.parametric_function import ParamFunction


class FunctionPerformanceModeling:
    def __init__(self,
                 function_name: str,
                 max_invocations: int = 5,
                 memory_bounds: tuple = (128, 3009),
                 region_name: str = 'us-east-1',
                 knowledge_termination_threshold: int = 3,
                 profiling_iterations: int = 4,
                 max_total_sample_count: int = 20,
                 payload: str = '{"key1": "value1"}',
                ):
        if not function_name:
            raise ValueError("Function name is required.")

        self.explorer = Explorer(
            function_name=function_name,
            max_invocations=max_invocations,
            memory_bounds=memory_bounds,
            boto_session=boto3.Session(region_name=region_name),
            payload=payload
        )

        self.param_function = ParamFunction()

        self.objective = Objective(
            param_function=self.param_function,
            memory_space=self.explorer.memory_space,
            termination_threshold=knowledge_termination_threshold
        )

        self.sampler = Sampler(
            explorer=self.explorer,
            profiling_iterations=profiling_iterations
        )

        self.optimizer = Optimizer(
            objective=self.objective,
            sampler=self.sampler,
            max_total_sample_count=max_total_sample_count
        )
        
        self._explored = False

     
    def run(self):
        if not self._explored:
            self.optimizer.start()
            self._explored = True
            
    
    def get_optimal_memory(self, latency_constraint_threshold_ms: float = None):
        if not self._explored:
            self.run()

        return self.param_function.minimize(
            self.explorer.memory_space,
            latency_constraint_threshold_ms=latency_constraint_threshold_ms
        )
        
    
    def get_performance_model_as_function(self):
        if not self._explored:
            self.run()

        return self.param_function.function
     