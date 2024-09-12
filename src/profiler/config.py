from dataclasses import dataclass

@dataclass
class FunctionConfig:
    
    memory_mb: int
    timeout_s: int