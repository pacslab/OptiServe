"""OptiServe — modeling and optimization of serverless ML applications on AWS.

Curated public API. See ``docs/architecture.md`` for the layered design.
"""
from optiserve.config import (
    AWSConfig,
    ModelingConfig,
    OptimizationConfig,
    OptiServeConfig,
    ProfilingConfig,
)
from optiserve.cost import CostCalculator
from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.modeling.function_model import FunctionPerformanceModeling
from optiserve.modeling.parametric import ParamFunction
from optiserve.optimization.application_optimizer import ApplicationOptimizer
from optiserve.optimization.result import OptimizationResult
from optiserve.workflow import FunctionNode, ModelVariant, WorkflowGraph

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # workflow
    "WorkflowGraph",
    "FunctionNode",
    "ModelVariant",
    # modeling
    "ApplicationPerformanceModeling",
    "FunctionPerformanceModeling",
    "ParamFunction",
    # optimization
    "ApplicationOptimizer",
    "OptimizationResult",
    # cost + config
    "CostCalculator",
    "OptiServeConfig",
    "AWSConfig",
    "ProfilingConfig",
    "ModelingConfig",
    "OptimizationConfig",
]
