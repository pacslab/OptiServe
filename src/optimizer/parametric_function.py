from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from src.utils.exploration import Exploration
from src.utils.logger import logger
from src.exceptions import (
    UnfeasibleConstraint,
)


@dataclass
class ParamFunction:
    
    function: callable = lambda x, a0, a1, a2: (a0 + a1 * np.exp(-x / a2)) if a2 != 0 else a0
    bounds: tuple = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    params: any = None
    
    
    def __call__(self, x: np.ndarray):
        return self.function(x, *self.params)
    
    
    def fit(self, exploration: Exploration):
        if self.params is None:
            self.params = [exploration.durations[0] // 10] * 3
            
            
        self.params = curve_fit(
            f = self.function,
            xdata = exploration.memories,
            ydata = exploration.durations,
            maxfev = int(1e8),
            p0 = self.params,
            bounds = self.bounds
        )[0]
        
        
    def minimize(self,
                 memory_space: np.ndarray,
                 latency_constraint_threshold_ms: float = None):

        exec_time = self(memory_space)
        costs = exec_time * memory_space
        
        if latency_constraint_threshold_ms:
            try:
                feasible_memories = exec_time < latency_constraint_threshold_ms
                
                if len(feasible_memories) == 0:
                    raise UnfeasibleConstraint(f"No feasible memory configuration found for latncy requirement {latency_constraint_threshold_ms} ms.")

            except UnfeasibleConstraint as e:
                logger.warning(e)
                
            else:
                memory_space = memory_space[feasible_memories]
                costs = costs[feasible_memories]
            
        return memory_space[np.argmin(costs)]
    
    
    
    