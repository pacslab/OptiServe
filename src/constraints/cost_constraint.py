from src.constraints.base_constraint import BaseConstraint
from src.application.function import Function
from typing import List


class CostConstraint(BaseConstraint):
    def __init__(self, max_cost: int):
        self.max_cost = max_cost

    def is_satisfied(self, solution: List[Function]) -> bool:
        sum_cost = sum(func.cost for func in solution)
        return sum_cost <= self.max_cost
