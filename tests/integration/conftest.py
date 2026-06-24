"""Fixtures for the moto-backed integration suite.

The same tests run against **two backends**, chosen by environment:

* no ``AWS_ENDPOINT_URL`` — in-process ``moto.mock_aws()`` (the default, and
  what a bare ``pytest -m integration`` uses); or
* ``AWS_ENDPOINT_URL`` set — a real **moto server** over HTTP, which is what the
  compose stack and CI use.

Running both matters: the in-process mock patches botocore from the inside, so
it cannot catch a client that was built with the wrong endpoint, the wrong
region, or a stale cached client. The server path exercises the seam OptiServe
actually ships — every client is constructed through ``create_client``, which
honours ``AWS_ENDPOINT_URL``, so no library code knows whether AWS is real.

moto covers Lambda's control plane and CloudWatch Logs, which is most of the
surface OptiServe uses. It does **not** implement Lambda ``Invoke`` without
Docker, nor the Pricing or Service Quotas APIs — those three seams are stubbed
explicitly by the tests that need them, and named as such rather than pretended.
"""

from __future__ import annotations

import io
import os
import time
import urllib.error
import urllib.request
import zipfile

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto", reason="moto is required for the integration suite")

#: Captured at import time: the root ``conftest`` scrubs ``AWS_ENDPOINT_URL``
#: from the environment of every test, so the value has to be read before any
#: fixture runs.
SERVER_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL") or None


def _reset_server(endpoint: str) -> None:
    """Wipe a moto server's state so tests stay independent.

    The in-process mock gets a fresh backend per ``mock_aws()`` context; a
    long-lived server does not, and leaked functions or log groups would make
    these tests order-dependent.
    """
    request = urllib.request.Request(f"{endpoint.rstrip('/')}/moto-api/reset", method="POST")
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except urllib.error.URLError as exc:  # pragma: no cover - CI plumbing
        pytest.skip(f"moto server at {endpoint} is unreachable: {exc}")


def _wait_for_server(endpoint: str, attempts: int = 30) -> None:  # pragma: no cover
    for _ in range(attempts):
        try:
            urllib.request.urlopen(f"{endpoint.rstrip('/')}/moto-api/", timeout=2).close()
            return
        except urllib.error.URLError:
            time.sleep(1)
    pytest.skip(f"moto server at {endpoint} did not become ready")


@pytest.fixture(scope="session")
def aws_backend() -> str:
    """Which backend this run uses; also waits for a server to come up."""
    if SERVER_ENDPOINT:
        _wait_for_server(SERVER_ENDPOINT)
        return "server"
    return "in-process"


@pytest.fixture
def aws(aws_backend, monkeypatch):
    """A mocked AWS account. Nothing here can reach the real API."""
    if aws_backend == "server":
        # Put the endpoint back (the root conftest removed it) so every client
        # OptiServe builds is redirected at the server.
        monkeypatch.setenv("AWS_ENDPOINT_URL", SERVER_ENDPOINT)
        _reset_server(SERVER_ENDPOINT)
        yield
    else:
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        with moto.mock_aws():
            yield


@pytest.fixture
def session(aws):
    return boto3.Session(region_name="us-east-1")


def _zip_stub() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("lambda_function.py", "def lambda_handler(event, ctx):\n    return {}\n")
    return buffer.getvalue()


@pytest.fixture
def deployed_function(session):
    """Create a real (mocked) Lambda function and return a factory for it."""
    lambda_client = session.client("lambda")
    iam = session.client("iam")
    try:
        role = iam.create_role(RoleName="optiserve-test", AssumeRolePolicyDocument="{}")["Role"][
            "Arn"
        ]
    except iam.exceptions.EntityAlreadyExistsException:  # pragma: no cover - server reuse
        role = iam.get_role(RoleName="optiserve-test")["Role"]["Arn"]

    def create(name="inference", memory=512, timeout=30, environment=None):
        # Idempotent: a test that simulates two consecutive processes (e.g. an
        # interrupted run resuming) calls this twice against the same account.
        try:
            lambda_client.create_function(
                FunctionName=name,
                Runtime="python3.11",
                Role=role,
                Handler="lambda_function.lambda_handler",
                Code={"ZipFile": _zip_stub()},
                MemorySize=memory,
                Timeout=timeout,
                Environment={"Variables": dict(environment or {})},
            )
        except lambda_client.exceptions.ResourceConflictException:
            lambda_client.update_function_configuration(
                FunctionName=name,
                MemorySize=memory,
                Timeout=timeout,
                Environment={"Variables": dict(environment or {})},
            )
        return name

    return create


@pytest.fixture
def log_group(session):
    """Create a CloudWatch log group and return a writer for REPORT lines."""
    logs = session.client("logs")

    def create(name, events, stream="stream-1"):
        logs.create_log_group(logGroupName=name)
        logs.create_log_stream(logGroupName=name, logStreamName=stream)
        # Timestamps must be recent: CloudWatch (and moto) reject log events
        # older than the retention horizon, so a hard-coded past timestamp is
        # accepted by PutLogEvents and then never returned by any query.
        now_ms = int(time.time() * 1000)
        logs.put_log_events(
            logGroupName=name,
            logStreamName=stream,
            logEvents=[{"timestamp": now_ms + i, "message": m} for i, m in enumerate(events)],
        )
        return name

    return create
