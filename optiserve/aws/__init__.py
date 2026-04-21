"""AWS adapter layer — the only place in OptiServe that talks to boto3.

Everything above this layer (profiling, modeling, optimization) depends on these
adapters, never on boto3 directly.
"""

from optiserve.aws.function_config import FunctionConfig
from optiserve.aws.lambda_client import ConfigManager, Invoker
from optiserve.aws.log_parser import LogParser
from optiserve.aws.logs_client import AWSApplicationLogs, AWSFunctionLogs, AWSLogs
from optiserve.aws.pricing_client import PricingClient
from optiserve.aws.session import create_client, create_session, default_botocore_config

__all__ = [
    "AWSApplicationLogs",
    "AWSFunctionLogs",
    "AWSLogs",
    "ConfigManager",
    "FunctionConfig",
    "Invoker",
    "LogParser",
    "PricingClient",
    "create_client",
    "create_session",
    "default_botocore_config",
]
