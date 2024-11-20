import numpy as np

from src.optimizer.objective import Objective
from src.utils.sampler import Sampler
from src.utils.logger import logger
from src.exceptions import (
    NotEnoughMemory,
    NoMemoryLeft,
)


class Optimizer:
    def __init__(
        self,
        objective: Objective,
        sampler: Sampler,
        max_total_sample_count: int = 30,
    ):
        self.objective = objective
        self.sampler = sampler
        self._max_total_sample_count = max_total_sample_count
        
    
    @property
    def _is_terminated(self):
        sample_count = len(self.sampler.exploration)
        print('Sample Count: ', sample_count)
        print('Max Sample Count: ', self._max_total_sample_count)
        print('Memories: ', self.sampler.exploration.memories)
        print('Duration: ', self.sampler.exploration.durations)
        termination_value = self.objective.termination_value
        print('Value and Threshold: ', termination_value, self.objective.termination_threshold)
        return (sample_count > self._max_total_sample_count or termination_value > self.objective.termination_threshold)
        
    
    def _initialize(self):
        self.sampler.exploration_init()
        
        exploration = self.sampler.exploration
        
        print(f'Explored memories: {exploration.memories}')
        
        for memory in set(exploration.memories):
            self.objective.update_knowledge(memory)
        
        try:
            self.objective.param_function.fit(exploration)
        except RuntimeError as e:
            logger.error(e.args[0])
            raise RuntimeError("Could not fit the parametric function.")
        
        
    def _update(self, memory_mb: int):
        try:
            self.sampler.update_exploration(memory_mb)
        except NotEnoughMemory as e:
            logger.error(f'Trying with new memories. {self.sampler.explorer.invoker._function_name}: {memory_mb}MB')
            self.memory_space = np.array(
                [
                    mem for mem in self.sampler.memory_space if
                    mem >= self.sampler.memory_space[0] + 128
                ],
                dtype=int
            )
            return
        
        self.objective.update_knowledge(memory_mb)
        try:
            self.objective.param_function.fit(self.sampler.exploration)
        except RuntimeError as e:
            logger.error(e.args[0])
            raise RuntimeError("Could not fit the parametric function.")
        
        
    def _select_next_memory_to_explore(self):
        exploration_memories = set(self.sampler.exploration.memories)
        memory_space = set(self.sampler.memory_space)
        
        remainder_memories = np.array(
            list(memory_space - exploration_memories),
            dtype=int
        )
        
        if len(remainder_memories) == 0:
            raise NoMemoryLeft()
        
        
        values = self.objective.get_values(remainder_memories)
        
        return remainder_memories[np.argmin(values)]
    
    
    def start(self):
        self._initialize()
        
        while not self._is_terminated:
            memory = self._select_next_memory_to_explore()
            self._update(memory)
            