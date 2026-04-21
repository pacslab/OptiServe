"""boto3 session and client construction.

Centralizes session creation so region, profile, retry policy and endpoint
overrides live in one place and every AWS adapter is built the same way.

Two behaviours matter for production use:

* **Adaptive retries.** Profiling drives Lambda's control plane far harder than
  a typical application: a memory sweep issues thousands of
  ``UpdateFunctionConfiguration`` calls, which is exactly the shape that trips
  AWS throttling. botocore's ``adaptive`` retry mode adds client-side rate
  limiting on top of retries, so the SDK backs off before the API does.
* **Endpoint override.** ``endpoint_url`` (or the standard ``AWS_ENDPOINT_URL``
  environment variable) points every client at a local fake — moto or
  LocalStack — which is the seam the offline evaluation stack uses. No library
  code needs to know whether AWS is real.
"""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.config import Config

__all__ = ["create_client", "create_session", "default_botocore_config"]

# Control-plane calls are cheap but heavily throttled; data-plane invocations are
# slow but rarely throttled. One conservative default serves both, and callers
# that need something else pass their own Config.
_DEFAULT_MAX_ATTEMPTS = 10
_DEFAULT_CONNECT_TIMEOUT_S = 10
_DEFAULT_READ_TIMEOUT_S = 900  # a Lambda may legitimately run for 15 minutes


def default_botocore_config(**overrides: Any) -> Config:
    """The botocore ``Config`` every OptiServe client uses unless told otherwise."""
    settings: dict[str, Any] = {
        "retries": {"max_attempts": _DEFAULT_MAX_ATTEMPTS, "mode": "adaptive"},
        "connect_timeout": _DEFAULT_CONNECT_TIMEOUT_S,
        "read_timeout": _DEFAULT_READ_TIMEOUT_S,
        "user_agent_extra": "optiserve",
    }
    settings.update(overrides)
    return Config(**settings)


def create_session(
    region_name: str | None = None, profile_name: str | None = None
) -> boto3.Session:
    """Create a boto3 Session.

    Credentials come from the standard AWS credential chain (environment,
    shared config, or instance role) — OptiServe never reads or stores them.
    """
    return boto3.Session(region_name=region_name, profile_name=profile_name)


def create_client(
    session: boto3.Session,
    service_name: str,
    *,
    region_name: str | None = None,
    endpoint_url: str | None = None,
    config: Config | None = None,
) -> Any:
    """Build a service client with OptiServe's defaults applied.

    ``endpoint_url`` falls back to ``AWS_ENDPOINT_URL`` so the whole stack can be
    redirected at a local mock with one environment variable. An empty value is
    treated as unset, which lets compose files disable an inherited override
    with ``AWS_ENDPOINT_URL: ""``.
    """
    if endpoint_url is None:
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL") or None

    return session.client(
        service_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        config=config or default_botocore_config(),
    )
