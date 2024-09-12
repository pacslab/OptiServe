from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from src.utils.exploration import Exploration
from src.utils.logger import logger


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
        
        
    def minimize(self, memory_space: np.ndarray):
        costs = self(memory_space) * memory_space
        return memory_space[np.argmin(costs)]
    
    
    
    