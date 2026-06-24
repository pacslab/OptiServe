"""Integration: the control-plane lifecycle against a mocked AWS account.

Unit tests pin the adapter's logic; these pin that it speaks the API correctly —
that ``UpdateFunctionConfiguration`` + the ``function_updated`` waiter really do
converge, and that the restore path really does put a function back.

This is the failure mode that matters most in production: a profiling run leaves
a live function on 128 MB with the wrong model loaded.
"""

import pytest

from optiserve.aws.lambda_client import ConfigManager
from optiserve.exceptions import FunctionConfigurationError
from optiserve.observability import EventName

pytestmark = pytest.mark.integration


def _config_of(session, name):
    response = session.client("lambda").get_function_configuration(FunctionName=name)
    return (
        response["MemorySize"],
        response["Timeout"],
        response.get("Environment", {}).get("Variables", {}).get("MODEL_NAME"),
    )


def test_reads_the_live_configuration(session, deployed_function):
    deployed_function(memory=1024, timeout=60, environment={"MODEL_NAME": "resnet-18"})
    config = ConfigManager("inference", session).get_config()
    assert (config.memory_mb, config.timeout_s, config.model_name) == (1024, 60, "resnet-18")


def test_applies_memory_and_model_and_waits_for_them_to_settle(session, deployed_function):
    deployed_function(memory=512, environment={"MODEL_NAME": "resnet-18"})
    manager = ConfigManager("inference", session)

    manager.set_config(memory_mb=3008, timeout_s=120, model_name="resnet-101")

    assert _config_of(session, "inference") == (3008, 120, "resnet-101")


def test_managed_restores_memory_timeout_and_model_after_a_crash(session, deployed_function):
    deployed_function(memory=512, timeout=30, environment={"MODEL_NAME": "resnet-18"})
    manager = ConfigManager("inference", session)
    before = _config_of(session, "inference")

    with pytest.raises(RuntimeError), manager.managed():
        for memory in (1024, 2048, 3008):
            manager.set_config(memory_mb=memory, model_name="resnet-101")
        assert _config_of(session, "inference")[0] == 3008
        raise RuntimeError("sweep aborted")

    assert _config_of(session, "inference") == before


def test_managed_removes_model_name_the_function_never_had(session, deployed_function):
    deployed_function(memory=512, environment={})
    manager = ConfigManager("inference", session)

    with manager.managed():
        manager.set_config(memory_mb=1024, model_name="yolov10n")
        assert _config_of(session, "inference")[2] == "yolov10n"

    memory, _, model = _config_of(session, "inference")
    assert (memory, model) == (512, None)


def test_configuration_is_cached_between_invocations(session, deployed_function):
    """`current_config` must not issue a control-plane call per data-plane
    invocation — that doubles API volume across a memory sweep."""
    deployed_function(memory=512)
    manager = ConfigManager("inference", session)
    manager.set_config(memory_mb=1024)

    first = manager.current_config()
    second = manager.current_config()
    assert first == second == manager.current_config()
    assert first.memory_mb == 1024

    # A mutation must invalidate it rather than serve a stale value.
    manager.set_config(memory_mb=2048)
    assert manager.current_config().memory_mb == 2048


def test_unknown_function_raises_a_typed_error(session, aws):
    with pytest.raises(FunctionConfigurationError):
        ConfigManager("does-not-exist", session).get_config()


def test_lifecycle_emits_apply_and_restore_events(session, deployed_function, recorded_events):
    deployed_function(memory=512, environment={"MODEL_NAME": "a"})
    manager = ConfigManager("inference", session)

    with manager.managed():
        manager.set_config(memory_mb=1024, model_name="b")

    assert recorded_events.count(EventName.CONFIG_APPLIED) >= 1
    assert recorded_events.count(EventName.CONFIG_RESTORED) == 1
    applied = recorded_events.of(EventName.CONFIG_APPLIED)[0]
    assert applied.attributes["memory_mb"] == 1024
