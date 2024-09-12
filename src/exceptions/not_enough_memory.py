from .invocation_error import InvocationError

class NotEnoughMemory(InvocationError):
    def __init__(self, message: str = "Not enough memory configurations to explore.", duration_ms: int = None):
        super().__init__(message, duration_ms)