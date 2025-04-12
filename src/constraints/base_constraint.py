from abc import ABC, abstractmethod
from src.application.function import Function
from typing import List


class BaseConstraint(ABC):
    """
    Abstract base class for defining constraints in an optimization problem.
    """

    @abstractmethod
    def is_satisfied(self, solution: List[Function]) -> bool:
        pass
