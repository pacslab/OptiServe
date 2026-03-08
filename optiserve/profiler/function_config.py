from dataclasses import dataclass
from typing import Optional


@dataclass
class FunctionConfig:

    memory_mb: int
    timeout_s: Optional[int] = None
    model_name: Optional[str] = None

    def to_string(self):
        return f"memory_mb: {self.memory_mb} timeout_s: {self.timeout_s} model_name: {self.model_name}"
