from src.constraints.base_constraint import BaseConstraint
from src.application.function import Function
from typing import List


class PerformanceConstraint(BaseConstraint):
    def __init__(self, max_exec_time: float):
        self.max_exec_time = max_exec_time

    def is_satisfied(self, solution: List[Function]) -> bool:
        sum_exec_time = sum(func.execution_time for func in solution)
        return sum_exec_time <= self.max_exec_time
