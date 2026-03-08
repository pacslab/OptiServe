from typing import Optional


class InvocationError(Exception):
    def __init__(self, message: str, duration_ms: Optional[int] = None):
        super().__init__(message)
        self.duration_ms = duration_ms
