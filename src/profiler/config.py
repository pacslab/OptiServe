from dataclasses import dataclass
from typing import Optional


@dataclass
class FunctionConfig:

    memory_mb: int
    timeout_s: Optional[int] = None
    model_name: Optional[str] = None
