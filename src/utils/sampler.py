import math
import numpy as np


from src.profiler.explorer import Explorer
from src.utils.logger import logger
from src.exceptions import (
    SamplingError,
    NotEnoughMemory,
    NoMemoryLeft,

)
from .sample import Sample
from .exploration import Exploration


class Sampler:
    def __init__(
        self,
        explorer: Explorer,
        profiling_iterations: int,
    ):
        self.exploration = None
        self.explorer = explorer
        self.memory_space = explorer.memory_space
        self._profiling_iterations = profiling_iterations
    
    
    def exploration_init(self):
        self.exploration = Exploration()
        
        self._explore_first_config()
        
        index = math.ceil(len(self.memory_space) / 3)
        
        for memory in [self.memory_space[index], self.memory_space[-1]]:
            try:
                self.update_exploration(memory_mb=memory)
            
            except SamplingError as e:
                logger.error(e)
                raise

    
    def _explore_first_config(self):
        while len(self.memory_space) >= 3:
            try:
                self.update_exploration(memory_mb=int(self.memory_space[0]))
                
            except NotEnoughMemory as e:
                logger.info(f'Trying with new memories. {self.explorer.invoker._function_name}: {self.memory_space[0]}MB')
                self.memory_space = np.array(
                    [
                        mem
                        for mem in self.memory_space
                        if mem >= self.memory_space[0] + 128
                    ],
                    dtype=int,
                )
            
            except SamplingError as e:
                logger.error(e)
                raise
            
            else:
                break
            
        if len(self.memory_space) <= 3:
            raise NoMemoryLeft()
    
    
    def update_exploration(self, memory_mb: int):
        logger.info(f"Exploring memory configuration: {memory_mb} MB for {self.explorer.invoker._function_name}")
        try:
            durations = self.explorer.explore_multi_threading(
                num_of_invocations=self._profiling_iterations,
                num_of_threads=self._profiling_iterations,
                memory_mb=memory_mb
            )
            
        except SamplingError as e:
            logger.error(e)
            raise
        
        
        durations = self._explore_dynamically(durations=durations)
        
        subsample = [Sample(memory_mb=memory_mb, duration_ms=duration) for duration in durations]

        self.exploration.add_sample(subsample)
        
        logger.info(f"Finished exploring memory configuration: {memory_mb} MB for {self.explorer.invoker._function_name}: {durations} ms")
        
        
    def _explore_dynamically(self, durations: list):
        if len(durations) < self._profiling_iterations:
            raise ValueError("The number of durations is less than the number of profiling iterations.")
        
        
        dynamic_exploration_count = 0
        min_cv = np.std(durations, ddof=1) / np.mean(durations)
        
        while (
            dynamic_exploration_count < 8 and
            min_cv > 0.05
        ):
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
        
        