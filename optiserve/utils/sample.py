from dataclasses import dataclass


@dataclass
class Sample:
    memory_mb: int
    duration_ms: float