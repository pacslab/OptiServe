from src.exceptions.optimization_error import OptimizationError


class UnfeasibleConstraint(OptimizationError):
    def __init__(self, message="One or more provided constraints are unfeasible."):
        super().__init__(message)
