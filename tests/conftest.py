"""Shared pytest fixtures.

Three jobs:

1. **Make it impossible for a test to reach real AWS.** Every test runs with
   scrubbed credentials and the metadata endpoint disabled, so a missing mock
   surfaces as an authentication error in *that* test rather than as a surprise
   API call against someone's account.
2. **Make the process-wide hook registry test-safe.** ``optiserve.observability
   .hooks`` is a module-level singleton; leaking a sink between tests would make
   assertions order-dependent.
3. **Auto-mark by location** — ``tests/integration/`` gets ``integration`` and
   ``tests/golden/`` gets ``golden``, so ``-m "not integration"`` needs no
   per-file bookkeeping.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden"

# The golden case builders are imported by module name from several test files.
# Doing this here replaces the manual sys.path.insert that was duplicated in
# tests/test_evaluation.py and the implicit rootdir insertion the golden tests
# relied on.
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))


_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_REGION": "us-east-1",
    # Never fall back to an instance role or a developer's ~/.aws profile.
    "AWS_EC2_METADATA_DISABLED": "true",
    "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    "AWS_CONFIG_FILE": os.devnull,
}

# Variables that must be *absent*, not empty: botocore treats AWS_PROFILE="" as
# a profile literally named "", and raises ProfileNotFound.
_AWS_ENV_UNSET = ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ENDPOINT_URL")


def pytest_collection_modifyitems(config, items):
    """Mark tests by directory so ``-m`` selection needs no per-file decorators."""
    for item in items:
        path = Path(str(item.fspath))
        if "integration" in path.parts:
            item.add_marker(pytest.mark.integration)
        if "golden" in path.parts:
            item.add_marker(pytest.mark.golden)


@pytest.fixture(autouse=True)
def isolated_aws_environment(monkeypatch):
    """Scrub AWS configuration for every test.

    Integration tests opt back in to a *mocked* endpoint by setting
    ``AWS_ENDPOINT_URL`` themselves; nothing here can reach real AWS.
    """
    for key, value in _AWS_ENV.items():
        monkeypatch.setenv(key, value)
    for key in _AWS_ENV_UNSET:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def clean_hook_registry():
    """Reset the process-wide observability registry around every test."""
    from optiserve.observability.hooks import hooks

    hooks.clear()
    yield
    hooks.clear()


@pytest.fixture
def recorded_events():
    """An :class:`InMemorySink` attached to the process-wide registry."""
    from optiserve.observability import InMemorySink, hooks

    sink = InMemorySink()
    hooks.add(sink)
    yield sink
    hooks.remove(sink)


@pytest.fixture
def fixed_pricing():
    """Lambda unit prices with no AWS call behind them.

    Every offline test injects these so the analytical layer never constructs a
    live pricing client. Values are the published us-east-1 x86_64 on-demand
    prices, so costs in tests are on a realistic scale.
    """
    return {"compute": 0.0000166667, "request": 0.0000002}


@pytest.fixture
def synthetic_exploration():
    """Factory for an :class:`Exploration` sampled from a known curve."""
    import numpy as np

    from optiserve.modeling.parametric import model_function
    from optiserve.profiling.sample import Exploration, Sample

    def build(a0=120.0, a1=3000.0, a2=300.0, memories=None):
        memories = np.arange(128, 3009, 192) if memories is None else np.asarray(memories)
        return Exploration(
            [
                Sample(memory_mb=int(m), duration_ms=float(model_function(m, a0, a1, a2)))
                for m in memories
            ]
        )

    return build
