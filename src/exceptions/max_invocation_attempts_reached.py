from .invocation_error import InvocationError


class MaxInvocationAttemptsReached(InvocationError):
    def __init__(self, message: str = "Maximum Max number of invocations' attempts reached.", duration_ms: int = None):
        super().__init__(message, duration_ms)
        