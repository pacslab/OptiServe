"""AWS Lambda control-plane and data-plane adapters.

- :class:`ConfigManager` reads/writes a function's memory, timeout, and the
  ``MODEL_NAME`` env var (used to switch which ML model a container serves).
- :class:`Invoker` invokes a function synchronously with retry/backoff and
  returns either the tail log or the response payload.

Both honor the caller-supplied ``boto3.Session`` (region/credentials).
"""
from __future__ import annotations

import base64
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError, ParamValidationError, ReadTimeoutError

from optiserve.aws.function_config import FunctionConfig
from optiserve.exceptions import (
    FunctionConfigurationError,
    FunctionTimeout,
    InvocationError,
    MaxInvocationAttemptsReached,
)
from optiserve.logging import get_logger

logger = get_logger(__name__)

# AWS service-quota code for a function's maximum timeout.
_MAX_TIMEOUT_QUOTA_CODE = "L-9FEEFFC0"


class ConfigManager:
    """Reads and mutates a live Lambda function's configuration."""

    def __init__(self, function_name: str, boto_session: boto3.Session):
        self._function_name = function_name
        self._initial_config: Optional[FunctionConfig] = None
        self._aws_lambda_client = boto_session.client("lambda")
        self._aws_quotas_client = boto_session.client("service-quotas")

    def _get_max_timeout(self) -> int:
        try:
            quota = self._aws_quotas_client.get_service_quota(
                ServiceCode="lambda", QuotaCode=_MAX_TIMEOUT_QUOTA_CODE
            )
        except ClientError:
            quota = self._aws_quotas_client.get_aws_default_service_quota(
                ServiceCode="lambda", QuotaCode=_MAX_TIMEOUT_QUOTA_CODE
            )
        return int(quota["Quota"]["Value"])

    def get_config(self) -> FunctionConfig:
        try:
            config = self._aws_lambda_client.get_function_configuration(
                FunctionName=self._function_name
            )
        except (ParamValidationError, ClientError) as exc:
            logger.debug("get_config failed: %s", exc)
            raise FunctionConfigurationError(str(exc))
        env = config.get("Environment", {}).get("Variables", {})
        return FunctionConfig(
            memory_mb=config["MemorySize"],
            timeout_s=config["Timeout"],
            model_name=env.get("MODEL_NAME"),
        )

    def set_config(
        self,
        memory_mb: Optional[int] = None,
        timeout_s: Optional[int] = None,
        model_name: Optional[str] = None,
        *,
        max_conflict_retries: int = 5,
    ):
        """Update memory / timeout / MODEL_NAME and wait for the change to
        settle. ``model_name == "None"`` (the sentinel used for non-ML
        functions) is normalized to ``None``.
        """
        if model_name == "None":
            model_name = None

        for attempt in range(max_conflict_retries + 1):
            try:
                return self._apply_config(memory_mb, timeout_s, model_name)
            except ParamValidationError as exc:
                logger.debug("set_config validation error: %s", exc)
                raise FunctionConfigurationError(str(exc))
            except ClientError as exc:
                if (
                    exc.response["Error"]["Code"] == "ResourceConflictException"
                    and attempt < max_conflict_retries
                ):
                    logger.warning(
                        "Concurrent update conflict; retry %d/%d",
                        attempt + 1,
                        max_conflict_retries,
                    )
                    time.sleep(2)
                    continue
                raise FunctionConfigurationError(str(exc))
        raise FunctionConfigurationError(
            "Exhausted retries updating function configuration."
        )

    def _apply_config(self, memory_mb, timeout_s, model_name):
        config = self._aws_lambda_client.get_function_configuration(
            FunctionName=self._function_name
        )
        current_env = dict(config.get("Environment", {}).get("Variables", {}))

        if memory_mb is None:
            memory_mb = int(config["MemorySize"])
        memory_mb = int(memory_mb)

        if self._initial_config is None:
            self._initial_config = FunctionConfig(
                memory_mb=config["MemorySize"],
                timeout_s=config["Timeout"],
                model_name=current_env.get("MODEL_NAME"),
            )

        if model_name is not None:
            current_env["MODEL_NAME"] = model_name

        timeout = self._get_max_timeout() if timeout_s is None else timeout_s

        self._aws_lambda_client.update_function_configuration(
            FunctionName=self._function_name,
            MemorySize=memory_mb,
            Timeout=timeout,
            Environment={"Variables": current_env},
        )

        # Wait for the update to reach a terminal state. The AWS waiter blocks
        # until LastUpdateStatus != InProgress; we then confirm memory matches
        # (and MODEL_NAME, only when we set it — checking it when model_name is
        # None could never converge if the function already has the env var).
        waiter = self._aws_lambda_client.get_waiter("function_updated")
        waiter.wait(FunctionName=self._function_name)
        config = self._aws_lambda_client.get_function_configuration(
            FunctionName=self._function_name
        )
        settled_env = config.get("Environment", {}).get("Variables", {})
        if config["MemorySize"] != memory_mb or (
            model_name is not None and settled_env.get("MODEL_NAME") != model_name
        ):
            raise FunctionConfigurationError(
                "Function configuration did not settle to the requested values."
            )
        return config

    def reset_config(self) -> None:
        if self._initial_config is None:
            raise FunctionConfigurationError("Initial configuration not set.")
        self.set_config(
            memory_mb=self._initial_config.memory_mb,
            timeout_s=self._initial_config.timeout_s,
            model_name=self._initial_config.model_name,
        )


class Invoker:
    """Synchronous Lambda invocation with retry/backoff."""

    def __init__(
        self, function_name: str, max_invocations: int, boto_session: boto3.Session
    ):
        self._function_name = function_name
        self._max_invocations = max_invocations
        self._aws_lambda_client = boto_session.client("lambda")

    def _invoke(self, payload: str):
        # The memory/timeout config does not change across retry attempts, so
        # fetch it once (only used for the log line and the timeout value).
        config = self._aws_lambda_client.get_function_configuration(
            FunctionName=self._function_name
        )
        memory_mb = config["MemorySize"]
        timeout_s = config["Timeout"]

        sleeping_interval = 1
        for _ in range(self._max_invocations):
            try:
                logger.info(
                    "Invoking %s (%s MB, %s s timeout)",
                    self._function_name,
                    memory_mb,
                    timeout_s,
                )
                return self._aws_lambda_client.invoke(
                    FunctionName=self._function_name,
                    LogType="Tail",
                    Payload=payload,
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "TooManyRequestsException":
                    logger.debug("Throttled; backing off %ss", sleeping_interval)
                    time.sleep(sleeping_interval)
                    sleeping_interval *= 2
                else:
                    logger.debug("Invocation ClientError: %s", exc)
                    raise InvocationError(
                        "Error invoking the Lambda function. Check that the "
                        "function name and configuration are correct."
                    )
            except ReadTimeoutError:
                logger.warning("Invocation timed out: %s", self._function_name)
                # timeout_s is in seconds; duration_ms expects milliseconds.
                raise FunctionTimeout(
                    duration_ms=int(timeout_s * 1000) if timeout_s is not None else None
                )
            except ParamValidationError as exc:
                raise InvocationError(str(exc))

        logger.warning("Max invocation attempts reached: %s", self._function_name)
        raise MaxInvocationAttemptsReached()

    def invoke_to_get_duration(self, payload: str) -> str:
        """Invoke and return the base64-decoded tail log (UTF-8)."""
        response = self._invoke(payload)
        return base64.b64decode(response["LogResult"]).decode("utf-8")

    def invoke_with_payload(self, payload: str) -> str:
        """Invoke and return the response payload (UTF-8)."""
        response = self._invoke(payload)
        return response["Payload"].read().decode("utf-8")
