from src.exceptions.invocation_error import InvocationError
from typing import Optional


class MaxInvocationAttemptsReached(InvocationError):
    def __init__(
        self,
        message: str = "Maximum Max number of invocations' attempts reached.",
        duration_ms: Optional[int] = None,
    ):
        super().__init__(message, duration_ms)
