"""Central configuration for OptiServe.

Frozen dataclasses that replace magic numbers previously scattered across the
profiler, modeling, and optimizer code. Each config object carries sensible
defaults (the values used in the thesis experiments) and can be overridden per
run. Later refactor stages wire these into the components that consume them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class AWSConfig:
    """AWS environment settings for profiling and pricing."""

    region_name: str = "us-east-1"
    architecture: str = "x86_64"  # or "arm64"


@dataclass(frozen=True)
class ProfilingConfig:
    """Settings for online function profiling (Explorer/Sampler/fitting loop)."""

    memory_bounds: Tuple[int, int] = (128, 3009)
    memory_space_step: int = 1
    max_invocations: int = 5
    profiling_iterations: int = 4
    max_total_sample_count: int = 20
    payload: str = '{"key1": "value1"}'
    # Adaptive-sampling controls.
    cv_threshold: float = 0.05          # target coefficient of variation
    max_dynamic_samples: int = 8        # extra samples to reduce CV
    memory_floor_step_mb: int = 128     # bump applied when a config is OOM
    # Acquisition / termination for the fitting loop.
    knowledge_sigma_mb: float = 200.0
    knowledge_termination_threshold: float = 3.0


@dataclass(frozen=True)
class ModelingConfig:
    """Settings for the analytical application model."""

    # AWS Step Functions transition overheads (per state / per edge), in ms.
    sfn_node_delay_ms: float = 18.81
    sfn_edge_delay_ms: float = 1.0


@dataclass(frozen=True)
class OptimizationConfig:
    """Settings for the application-level greedy optimizer."""

    bcr_threshold: float = 0.1
    accuracy_penalty_weight: float = 100.0  # `w` in the model-upgrade score


@dataclass(frozen=True)
class OptiServeConfig:
    """Aggregate configuration bundling every sub-config."""

    aws: AWSConfig = field(default_factory=AWSConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
