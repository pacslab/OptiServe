from src.constraints.base_constraint import BaseConstraint
from abc import ABC, abstractmethod
from src.application.function import Function
from typing import List


class EndToEndAccuracyFormula(ABC):
    def __init__(self, function_names: List[str]):
        self.function_names = function_names

    def calculate_accuracy(self, function_chain: List[Function]) -> float:
        for function_name in self.function_names:
            function = next(
                (f for f in function_chain if f.name == function_name), None
            )
            if function is None:
                raise ValueError(f"Function {function_name} not found in the chain.")

        return self._accuracy_formula()

    @abstractmethod
    def _accuracy_formula(self) -> float:
        """
        Returns the end-to-end accuracy of the function chain.
        """
        pass


class AccuracyConstraint(BaseConstraint):
    def __init__(
        self,
        min_accuracy: float,
        max_accuracy: float,
        end_to_end_accuracy_formula: EndToEndAccuracyFormula,
    ):
        self.min_accuracy = min_accuracy
        self.max_accuracy = max_accuracy
        self.end_to_end_accuracy_formula = end_to_end_accuracy_formula

    def is_satisfied(self, solution: List[Function]) -> bool:
        eas = self.end_to_end_accuracy_formula.calculate_accuracy(solution)
        return self.min_accuracy <= eas <= self.max_accuracy
