"""Central configuration for OptiServe.

Frozen dataclasses that replace magic numbers previously scattered across the
profiler, modeling, and optimizer code. Each config object carries sensible
defaults (the values used in the thesis experiments) and can be overridden per
run. Later refactor stages wire these into the components that consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optiserve.optimization.compat import OptimizerCompat


@dataclass(frozen=True)
class AWSConfig:
    """AWS environment settings for profiling and pricing."""

    region_name: str = "us-east-1"
    architecture: str = "x86_64"  # or "arm64"


@dataclass(frozen=True)
class ProfilingConfig:
    """Settings for online function profiling (Explorer/Sampler/fitting loop)."""

    # Inclusive bounds. 3008 MB was Lambda's ceiling until 2020 and is the
    # value the published experiments profiled; the platform maximum is now
    # 10240 MB (optiserve.profiling.explorer.LAMBDA_MAX_MEMORY_MB). Raise
    # this to explore the full modern range — it is left at the historical
    # value so the default reproduces the published measurements.
    memory_bounds: tuple[int, int] = (128, 3008)
    memory_space_step: int = 1
    max_invocations: int = 5
    profiling_iterations: int = 4
    max_total_sample_count: int = 20
    payload: str = '{"key1": "value1"}'
    # Adaptive-sampling controls.
    cv_threshold: float = 0.05  # target coefficient of variation
    max_dynamic_samples: int = 8  # extra samples to reduce CV
    # Whether to run the CV-substitution step in Sampler._explore_dynamically.
    #
    # That step replaces measured durations with lower-variance substitutes,
    # which is selection on the outcome variable, not noise reduction: over
    # 4 000 simulated replications it removes ~74% of the sample variance and
    # shifts the retained mean by about +1.3%. It is ON by default because the
    # published results were produced with it; set it to False for any new
    # measurement campaign where the per-invocation numbers must stand on their
    # own, and say which setting a result was produced under.
    noise_reduction: bool = True
    memory_floor_step_mb: int = 128  # bump applied when a config is OOM
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
    # Which known optimizer defects to reproduce. Defaults to CORRECTED; the
    # golden battery and any reproduction of the published figures ask for
    # PUBLISHED explicitly. See optiserve.optimization.compat.
    compat: OptimizerCompat = OptimizerCompat.CORRECTED


@dataclass(frozen=True)
class OptiServeConfig:
    """Aggregate configuration bundling every sub-config."""

    aws: AWSConfig = field(default_factory=AWSConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
