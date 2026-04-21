"""AWS Lambda control-plane and data-plane adapters.

- :class:`ConfigManager` reads/writes a function's memory, timeout, and the
  ``MODEL_NAME`` env var (used to switch which ML model a container serves).
- :class:`Invoker` invokes a function synchronously with retry/backoff and
  returns either the tail log or the response payload.

Both honor the caller-supplied ``boto3.Session`` (region/credentials).

Two production concerns drive this module's shape:

**API-call amplification.** Profiling is control-plane heavy. Naively, every
memory step costs an ``UpdateFunctionConfiguration`` plus a ``GetServiceQuota``
plus one ``GetFunctionConfiguration`` *per invocation*; a 2 881-step sweep at 4
invocations each is ~14 000 avoidable control-plane calls, and Lambda's control
plane throttles long before that. The quota and the settled configuration are
therefore cached and invalidated on mutation, and :class:`Invoker` reads the
configuration through a provider instead of re-fetching it.

**Restoration.** A profiling run mutates a *live* function. If the process dies
between "set memory to 128 MB" and "restore", the function is left broken.
:meth:`ConfigManager.managed` captures the configuration on entry and restores
it in a ``finally``, so every exit path — exception, ``KeyboardInterrupt``, a
caller's ``return`` — puts the function back.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import boto3
from botocore.exceptions import ClientError, ParamValidationError, ReadTimeoutError

from optiserve.aws.function_config import FunctionConfig
from optiserve.aws.session import create_client, default_botocore_config
from optiserve.exceptions import (
    FunctionConfigurationError,
    FunctionTimeout,
    InvocationError,
    MaxInvocationAttemptsReached,
)
from optiserve.logging import get_logger
from optiserve.observability.events import EventName
from optiserve.observability.hooks import HookRegistry
from optiserve.observability.hooks import hooks as default_hooks

logger = get_logger(__name__)

# AWS service-quota code for a function's maximum timeout.
_MAX_TIMEOUT_QUOTA_CODE = "L-9FEEFFC0"
# Lambda's documented hard ceiling, used when the quota API is unavailable
# (it is not implemented by every AWS-compatible endpoint, including moto).
_LAMBDA_MAX_TIMEOUT_S = 900


class ConfigManager:
    """Reads and mutates a live Lambda function's configuration."""

    def __init__(
        self,
        function_name: str,
        boto_session: boto3.Session,
        *,
        hooks: HookRegistry | None = None,
    ) -> None:
        self._function_name = function_name
        self._initial_config: FunctionConfig | None = None
        self._aws_lambda_client = create_client(boto_session, "lambda")
        # The quota lookup is a one-shot read of a constant with a safe
        # documented fallback, so it gets a short retry budget rather than the
        # profiling default. Against an endpoint that returns a retryable error
        # for this API — moto does not implement it — ten adaptive retries with
        # backoff would stall the profiling critical path for minutes before
        # taking a fallback that was always available.
        self._aws_quotas_client = create_client(
            boto_session,
            "service-quotas",
            config=default_botocore_config(
                retries={"max_attempts": 2, "mode": "standard"},
                connect_timeout=5,
                read_timeout=10,
            ),
        )
        self._hooks = hooks if hooks is not None else default_hooks
        # Caches. `_max_timeout` never changes during a run; `_settled_config`
        # is the last configuration we observed AWS converge to.
        self._max_timeout: int | None = None
        self._settled_config: FunctionConfig | None = None

    @property
    def function_name(self) -> str:
        return self._function_name

    @property
    def initial_config(self) -> FunctionConfig | None:
        """The configuration captured before OptiServe first mutated the
        function, or ``None`` if it has not been touched yet."""
        return self._initial_config

    def _get_max_timeout(self) -> int:
        """The account's maximum Lambda timeout, fetched once per manager.

        Falls back to Lambda's documented ceiling when the Service Quotas API is
        unavailable — some AWS-compatible endpoints do not implement it, and an
        unreachable quota lookup must not abort a profiling run.
        """
        if self._max_timeout is not None:
            return self._max_timeout
        try:
            quota = self._aws_quotas_client.get_service_quota(
                ServiceCode="lambda", QuotaCode=_MAX_TIMEOUT_QUOTA_CODE
            )
        except ClientError:
            try:
                quota = self._aws_quotas_client.get_aws_default_service_quota(
                    ServiceCode="lambda", QuotaCode=_MAX_TIMEOUT_QUOTA_CODE
                )
            except Exception:
                logger.warning(
                    "Service Quotas unavailable; using the documented Lambda "
                    "maximum timeout of %ds.",
                    _LAMBDA_MAX_TIMEOUT_S,
                )
                self._max_timeout = _LAMBDA_MAX_TIMEOUT_S
                return self._max_timeout
        except Exception:
            logger.warning(
                "Service Quotas unavailable; using the documented Lambda maximum timeout of %ds.",
                _LAMBDA_MAX_TIMEOUT_S,
            )
            self._max_timeout = _LAMBDA_MAX_TIMEOUT_S
            return self._max_timeout

        self._max_timeout = int(quota["Quota"]["Value"])
        return self._max_timeout

    @staticmethod
    def _to_function_config(config: dict) -> FunctionConfig:
        env = config.get("Environment", {}).get("Variables", {})
        return FunctionConfig(
            memory_mb=config["MemorySize"],
            timeout_s=config["Timeout"],
            model_name=env.get("MODEL_NAME"),
        )

    def get_config(self) -> FunctionConfig:
        try:
            config = self._aws_lambda_client.get_function_configuration(
                FunctionName=self._function_name
            )
        except (ParamValidationError, ClientError) as exc:
            logger.debug("get_config failed: %s", exc)
            raise FunctionConfigurationError(str(exc)) from exc
        self._settled_config = self._to_function_config(config)
        return self._settled_config

    def current_config(self) -> FunctionConfig:
        """The function's configuration, served from cache when known.

        :class:`Invoker` calls this before every invocation; without the cache
        that is one control-plane request per data-plane request.
        """
        if self._settled_config is not None:
            return self._settled_config
        return self.get_config()

    def set_config(
        self,
        memory_mb: int | None = None,
        timeout_s: int | None = None,
        model_name: str | None = None,
        *,
        clear_model_name: bool = False,
        max_conflict_retries: int = 5,
    ) -> dict:
        """Update memory / timeout / MODEL_NAME and wait for the change to
        settle. ``model_name == "None"`` (the sentinel used for non-ML
        functions) is normalized to ``None``.
        """
        if model_name == "None":
            model_name = None

        for attempt in range(max_conflict_retries + 1):
            try:
                return self._apply_config(
                    memory_mb, timeout_s, model_name, clear_model_name=clear_model_name
                )
            except ParamValidationError as exc:
                logger.debug("set_config validation error: %s", exc)
                raise FunctionConfigurationError(str(exc)) from exc
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
                raise FunctionConfigurationError(str(exc)) from exc
        raise FunctionConfigurationError("Exhausted retries updating function configuration.")

    def _apply_config(
        self,
        memory_mb: int | None,
        timeout_s: int | None,
        model_name: str | None,
        *,
        clear_model_name: bool = False,
    ) -> dict:
        config = self._aws_lambda_client.get_function_configuration(
            FunctionName=self._function_name
        )
        current_env = dict(config.get("Environment", {}).get("Variables", {}))

        if memory_mb is None:
            memory_mb = int(config["MemorySize"])
        memory_mb = int(memory_mb)

        if self._initial_config is None:
            self._initial_config = self._to_function_config(config)

        if clear_model_name:
            # Restoring a function that had no MODEL_NAME to begin with. Setting
            # the variable to None is not expressible in the API, so the only
            # faithful restore is to drop the key entirely — otherwise the
            # function keeps whichever model the profiler last probed.
            current_env.pop("MODEL_NAME", None)
        elif model_name is not None:
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
        model_settled = (
            "MODEL_NAME" not in settled_env
            if clear_model_name
            else (model_name is None or settled_env.get("MODEL_NAME") == model_name)
        )
        if config["MemorySize"] != memory_mb or not model_settled:
            self._settled_config = None
            raise FunctionConfigurationError(
                "Function configuration did not settle to the requested values."
            )

        self._settled_config = self._to_function_config(config)
        self._hooks.emit(
            EventName.CONFIG_APPLIED,
            function=self._function_name,
            memory_mb=self._settled_config.memory_mb,
            timeout_s=self._settled_config.timeout_s,
            model=self._settled_config.model_name,
        )
        return config

    def reset_config(self) -> None:
        """Put the function back exactly as it was found.

        Includes removing ``MODEL_NAME`` when the function did not have it
        before: ``set_config(model_name=None)`` is "leave it alone", so a
        function that started without the variable used to be left carrying
        whichever model the profiler last probed.
        """
        if self._initial_config is None:
            raise FunctionConfigurationError("Initial configuration not set.")
        self.set_config(
            memory_mb=self._initial_config.memory_mb,
            timeout_s=self._initial_config.timeout_s,
            model_name=self._initial_config.model_name,
            clear_model_name=self._initial_config.model_name is None,
        )
        self._hooks.emit(
            EventName.CONFIG_RESTORED,
            function=self._function_name,
            memory_mb=self._initial_config.memory_mb,
            model=self._initial_config.model_name,
        )

    @contextmanager
    def managed(self) -> Iterator[ConfigManager]:
        """Capture the live configuration on entry, restore it on every exit.

        Profiling mutates a *production* function. Without this, any crash
        between the first ``set_config`` and the final ``reset_config`` leaves
        the function on whatever memory size the sweep happened to be probing.

            with ConfigManager(name, session).managed() as manager:
                manager.set_config(memory_mb=512)
                ...                      # restored even if this raises

        Restoration failures are logged, never raised, so they cannot mask the
        original exception that caused the exit.
        """
        # Eager capture: reset_config must work even if the body never mutated
        # anything, and the captured value must predate the first mutation.
        if self._initial_config is None:
            self._initial_config = self.get_config()
        try:
            yield self
        finally:
            try:
                self.reset_config()
            except Exception:
                logger.exception(
                    "Failed to restore %s to %s. Restore it manually before "
                    "the function is used in production.",
                    self._function_name,
                    self._initial_config,
                )


class Invoker:
    """Synchronous Lambda invocation with retry/backoff."""

    def __init__(
        self,
        function_name: str,
        max_invocations: int,
        boto_session: boto3.Session,
        *,
        config_provider: Callable[[], FunctionConfig] | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        self._function_name = function_name
        self._max_invocations = max_invocations
        self._aws_lambda_client = create_client(boto_session, "lambda")
        # Where the current memory/timeout comes from. Defaults to a
        # self-managed one-shot fetch; the profiler passes the ConfigManager's
        # cached accessor so an invocation costs no control-plane call.
        self._config_provider = config_provider or self._fetch_config
        self._cached_config: FunctionConfig | None = None
        self._hooks = hooks if hooks is not None else default_hooks

    def _fetch_config(self) -> FunctionConfig:
        if self._cached_config is None:
            config = self._aws_lambda_client.get_function_configuration(
                FunctionName=self._function_name
            )
            env = config.get("Environment", {}).get("Variables", {})
            self._cached_config = FunctionConfig(
                memory_mb=config["MemorySize"],
                timeout_s=config["Timeout"],
                model_name=env.get("MODEL_NAME"),
            )
        return self._cached_config

    def invalidate(self) -> None:
        """Drop the cached configuration (call after mutating the function)."""
        self._cached_config = None

    def _invoke(self, payload: str) -> Any:
        # The memory/timeout config does not change across retry attempts, so
        # read it once — and through the provider, so a profiling sweep does not
        # pay a GetFunctionConfiguration for every single invocation.
        config = self._config_provider()
        memory_mb = config.memory_mb
        timeout_s = config.timeout_s

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
                    self._hooks.emit(
                        EventName.INVOCATION_FAILED,
                        function=self._function_name,
                        memory_mb=memory_mb,
                        reason="throttled",
                        backoff_s=sleeping_interval,
                    )
                    time.sleep(sleeping_interval)
                    sleeping_interval *= 2
                else:
                    logger.debug("Invocation ClientError: %s", exc)
                    raise InvocationError(
                        "Error invoking the Lambda function. Check that the "
                        "function name and configuration are correct."
                    ) from exc
            except ReadTimeoutError as exc:
                logger.warning("Invocation timed out: %s", self._function_name)
                # timeout_s is in seconds; duration_ms expects milliseconds.
                raise FunctionTimeout(
                    duration_ms=int(timeout_s * 1000) if timeout_s is not None else None
                ) from exc
            except ParamValidationError as exc:
                raise InvocationError(str(exc)) from exc

        logger.warning("Max invocation attempts reached: %s", self._function_name)
        raise MaxInvocationAttemptsReached()

    def invoke_to_get_duration(self, payload: str) -> str:
        """Invoke and return the base64-decoded tail log (UTF-8)."""
        response = self._invoke(payload)
        log_result = response.get("LogResult")
        if log_result is None:
            raise InvocationError(
                "Lambda returned no tail log; invoke with LogType='Tail' against "
                "an endpoint that supports it."
            )
        return base64.b64decode(log_result).decode("utf-8")

    def invoke_with_payload(self, payload: str) -> str:
        """Invoke and return the response payload (UTF-8)."""
        response = self._invoke(payload)
        return response["Payload"].read().decode("utf-8")
