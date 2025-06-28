from dataclasses import dataclass

import numpy as np
from typing import Any, Optional, Union
from scipy.optimize import curve_fit
from src.utils.exploration import Exploration
from src.utils.logger import logger
from src.exceptions import (
    UnfeasibleConstraint,
)
from pathlib import Path


def model_function(x, a0, a1, a2):
    """
    A model function that describes the relationship between memory allocation and execution time.
    This function is used for curve fitting to find the best parameters for the model.
    
    Parameters:
    - x: Memory allocation (independent variable).
    - a0: Intercept parameter.
    - a1: Exponential decay parameter.
    - a2: Scale parameter for the exponential decay.
    
    Returns:
    - Computed execution time based on the model.
    """
    return (a0 + a1 * np.exp(-x / a2)) if a2 != 0 else a0


@dataclass
class ParamFunction:
    
    """
    A class representing a parametric function for modeling the execution time of a function based on allocated memory.
    This class uses a curve fitting approach to model the relationship between memory allocation and execution time.
    It defines a callable function that can be used to predict execution time based on memory allocation.
    """

    function: callable = model_function
    bounds: tuple = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    params: Any = None


    def __call__(self, x: np.ndarray):
        return self.function(x, *self.params)

    def fit(self, exploration: Exploration):
        if self.params is None:
            self.params = [exploration.durations[0] // 10] * 3

        self.params = curve_fit(
            f=self.function,
            xdata=exploration.memories,
            ydata=exploration.durations,
            maxfev=int(1e8),
            p0=self.params,
            bounds=self.bounds,
        )[0]

    def minimize(
        self,
        memory_space: np.ndarray,
        latency_constraint_threshold_ms: Optional[float] = None,
    ):

        exec_time = self(memory_space)
        costs = exec_time * memory_space

        if latency_constraint_threshold_ms:
            try:
                feasible_memories = exec_time < latency_constraint_threshold_ms

                if not np.any(feasible_memories):
                    raise UnfeasibleConstraint(
                        f"No feasible memory configuration found for latncy requirement {latency_constraint_threshold_ms} ms."
                    )

            except UnfeasibleConstraint as e:
                logger.warning(e)

            else:
                memory_space = memory_space[feasible_memories]
                costs = costs[feasible_memories]

        return memory_space[np.argmin(costs)]
    
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParamFunction":
        import joblib

        return joblib.load(path)
    
    
    def save(self, path: Union[str, Path]) -> None:
        import joblib

        joblib.dump(self, path)
        logger.info(f"ParamFunction saved to {path}.")
