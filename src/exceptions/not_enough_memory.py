from src.exceptions.invocation_error import InvocationError
from typing import Optional


class NotEnoughMemory(InvocationError):
    def __init__(
        self,
        message: str = "Not enough memory configurations to explore.",
        duration_ms: Optional[int] = None,
    ):
        super().__init__(message, duration_ms)
