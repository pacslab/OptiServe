from dataclasses import dataclass


@dataclass
class FunctionConfig:
    memory_mb: int
    timeout_s: int | None = None
    model_name: str | None = None

    def to_string(self) -> str:
        return (
            f"memory_mb: {self.memory_mb} timeout_s: {self.timeout_s} model_name: {self.model_name}"
        )
