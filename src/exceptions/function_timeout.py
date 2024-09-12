from .not_enough_memory import NotEnoughMemory

class FunctionTimeout(NotEnoughMemory):
    def __init__(self, message: str = "Function timed out. The execution time limit is reached", duration_ms: int = None):
        super().__init__(message, duration_ms)