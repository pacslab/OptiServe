"""Unit tests for the AWS adapter layer — no network, no moto.

These pin behaviours that used to be wrong in ways that are invisible until you
point OptiServe at a real account: OOM that never fires, a restore that does not
restore, a pricing lookup that misses its SKU on page two, and a Logs Insights
query that is abandoned rather than stopped.
"""

import json
from unittest.mock import Mock

import pytest

from optiserve.aws.lambda_client import ConfigManager, Invoker
from optiserve.aws.log_parser import LogParser
from optiserve.aws.logs_client import AWSApplicationLogs, AWSFunctionLogs
from optiserve.aws.pricing_client import PricingClient
from optiserve.exceptions import (
    FunctionConfigurationError,
    FunctionTimeout,
    InvocationError,
    LogParsingError,
    NotEnoughMemory,
)

# --------------------------------------------------------------------------- #
# LogParser — OOM detection against realistic AWS output
# --------------------------------------------------------------------------- #
_REPORT = (
    "REPORT RequestId: a\tDuration: {d}.00 ms\tBilled Duration: {d} ms\t"
    "Memory Size: {size} MB\tMax Memory Used: {used} MB\t"
)


def test_real_aws_oom_is_detected_even_though_usage_is_clamped():
    """AWS clamps 'Max Memory Used' at 'Memory Size', so the numeric comparison
    can never fire on a real OOM. The runtime marker is the signal that does."""
    log = (
        "START RequestId: a Version: $LATEST\n"
        "RequestId: a Error: Runtime exited with error: signal: killed\n"
        "Runtime.ExitError\n"
        "END RequestId: a\n" + _REPORT.format(d=4100, size=512, used=512)
    )
    with pytest.raises(NotEnoughMemory) as excinfo:
        LogParser().parse_function_execution_time(log)
    assert excinfo.value.duration_ms == 4100


@pytest.mark.parametrize(
    "marker",
    ["Runtime exited with error: signal: killed", "Runtime.OutOfMemory", "MemoryError"],
)
def test_each_oom_marker_prunes_the_memory_size(marker):
    log = f"{marker}\nEND RequestId: a\n" + _REPORT.format(d=100, size=512, used=512)
    with pytest.raises(NotEnoughMemory):
        LogParser().parse_function_execution_time(log)


def test_a_healthy_invocation_at_exactly_the_limit_is_not_an_oom():
    # No marker: using every configured megabyte is legal, not a failure.
    log = _REPORT.format(d=100, size=512, used=512)
    assert LogParser().parse_function_execution_time(log) == 100.0


def test_multiline_application_error_still_yields_a_billed_duration():
    log = (
        "START RequestId: a\n[ERROR] ValueError: bad input\n"
        'Traceback (most recent call last):\n  File "x.py", line 3\n'
        "END RequestId: a\n" + _REPORT.format(d=90, size=512, used=100)
    )
    assert LogParser().parse_function_execution_time(log) == 90


def test_a_log_without_a_report_line_is_a_parsing_error():
    with pytest.raises(LogParsingError):
        LogParser().parse_function_execution_time("START RequestId: a\nEND RequestId: a")


# --------------------------------------------------------------------------- #
# ConfigManager — restoration and API-call amplification
# --------------------------------------------------------------------------- #
def _lambda_client(memory=512, timeout=3, env=None):
    client = Mock()
    state = {
        "MemorySize": memory,
        "Timeout": timeout,
        "Environment": {"Variables": dict(env or {})},
    }

    def get_config(FunctionName):
        return {
            "MemorySize": state["MemorySize"],
            "Timeout": state["Timeout"],
            "Environment": {"Variables": dict(state["Environment"]["Variables"])},
        }

    def update_config(FunctionName, MemorySize, Timeout, Environment):
        state["MemorySize"] = MemorySize
        state["Timeout"] = Timeout
        state["Environment"] = {"Variables": dict(Environment["Variables"])}
        return state

    client.get_function_configuration.side_effect = get_config
    client.update_function_configuration.side_effect = update_config
    client.get_waiter.return_value = Mock(wait=Mock())
    return client, state


def _manager(client, quotas=None):
    manager = ConfigManager.__new__(ConfigManager)
    manager._function_name = "f"
    manager._initial_config = None
    manager._aws_lambda_client = client
    manager._aws_quotas_client = quotas or Mock(
        get_service_quota=Mock(return_value={"Quota": {"Value": 900}})
    )
    from optiserve.observability.hooks import hooks

    manager._hooks = hooks
    manager._max_timeout = None
    manager._settled_config = None
    return manager


def test_managed_restores_the_function_after_an_exception():
    client, state = _lambda_client(memory=512, timeout=30, env={"MODEL_NAME": "resnet-18"})
    manager = _manager(client)

    with pytest.raises(RuntimeError), manager.managed():
        manager.set_config(memory_mb=3008, model_name="resnet-101")
        assert state["MemorySize"] == 3008
        raise RuntimeError("profiling blew up")

    assert state["MemorySize"] == 512
    assert state["Timeout"] == 30
    assert state["Environment"]["Variables"]["MODEL_NAME"] == "resnet-18"


def test_managed_restores_on_keyboard_interrupt():
    client, state = _lambda_client(memory=512)
    manager = _manager(client)
    with pytest.raises(KeyboardInterrupt), manager.managed():
        manager.set_config(memory_mb=2048)
        raise KeyboardInterrupt
    assert state["MemorySize"] == 512


def test_restore_removes_model_name_when_the_function_never_had_one():
    # set_config(model_name=None) means "leave it alone", so a function that
    # started without MODEL_NAME used to keep whatever the profiler last probed.
    client, state = _lambda_client(memory=512, env={})
    manager = _manager(client)

    with manager.managed():
        manager.set_config(memory_mb=1024, model_name="resnet-50")
        assert state["Environment"]["Variables"]["MODEL_NAME"] == "resnet-50"

    assert "MODEL_NAME" not in state["Environment"]["Variables"]


def test_restore_failure_does_not_mask_the_original_exception(caplog):
    """If the restore itself fails, the caller must still see why the run died —
    and be told, loudly, that the function was left mutated."""
    import logging

    client, _ = _lambda_client()
    manager = _manager(client)

    with (
        caplog.at_level(logging.ERROR, logger="optiserve"),
        pytest.raises(ValueError, match="the real failure"),
        manager.managed(),
    ):
        manager.set_config(memory_mb=1024)
        client.update_function_configuration.side_effect = RuntimeError("AWS down")
        raise ValueError("the real failure")

    assert any("Failed to restore" in record.getMessage() for record in caplog.records), (
        "an un-restored production function must be reported"
    )


def test_the_service_quota_is_fetched_once_not_per_configuration():
    client, _ = _lambda_client()
    quotas = Mock(get_service_quota=Mock(return_value={"Quota": {"Value": 900}}))
    manager = _manager(client, quotas)

    for memory in (256, 512, 1024, 2048):
        manager.set_config(memory_mb=memory)

    assert quotas.get_service_quota.call_count == 1


def test_quota_api_unavailable_falls_back_to_the_documented_maximum():
    client, state = _lambda_client()
    quotas = Mock()
    quotas.get_service_quota.side_effect = RuntimeError("not implemented here")
    manager = _manager(client, quotas)

    manager.set_config(memory_mb=1024)
    assert state["Timeout"] == 900


def test_settling_check_rejects_a_configuration_that_did_not_apply():
    client, _ = _lambda_client()
    client.update_function_configuration.side_effect = lambda **kwargs: None  # no-op
    manager = _manager(client)
    with pytest.raises(FunctionConfigurationError):
        manager.set_config(memory_mb=2048)


# --------------------------------------------------------------------------- #
# Invoker — one control-plane call per sweep, not per invocation
# --------------------------------------------------------------------------- #
def test_invoker_reads_configuration_through_the_provider():
    from optiserve.aws.function_config import FunctionConfig

    client = Mock()
    client.invoke.return_value = {"LogResult": b"", "Payload": Mock(read=lambda: b"{}")}
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return FunctionConfig(memory_mb=512, timeout_s=30)

    invoker = Invoker.__new__(Invoker)
    invoker._function_name = "f"
    invoker._max_invocations = 3
    invoker._aws_lambda_client = client
    invoker._config_provider = provider
    invoker._cached_config = None
    from optiserve.observability.hooks import hooks

    invoker._hooks = hooks

    for _ in range(10):
        invoker._invoke("{}")

    # Ten invocations, zero GetFunctionConfiguration calls on the control plane.
    assert client.get_function_configuration.call_count == 0
    assert calls["n"] == 10
    assert client.invoke.call_count == 10


def test_missing_tail_log_is_reported_not_silently_decoded():
    client = Mock()
    client.invoke.return_value = {"Payload": Mock(read=lambda: b"{}")}
    invoker = Invoker.__new__(Invoker)
    invoker._function_name = "f"
    invoker._max_invocations = 1
    invoker._aws_lambda_client = client
    invoker._config_provider = lambda: Mock(memory_mb=512, timeout_s=30)
    invoker._cached_config = None
    from optiserve.observability.hooks import hooks

    invoker._hooks = hooks

    with pytest.raises(InvocationError, match="no tail log"):
        invoker.invoke_to_get_duration("{}")


# --------------------------------------------------------------------------- #
# PricingClient — pagination, session reuse, tier selection
# --------------------------------------------------------------------------- #
def _product(group, usd_values):
    return json.dumps(
        {
            "product": {"attributes": {"group": group}},
            "terms": {
                "OnDemand": {
                    f"term{i}": {"priceDimensions": {f"rate{i}": {"pricePerUnit": {"USD": value}}}}
                    for i, value in enumerate(usd_values)
                }
            },
        }
    )


def _pricing_client(pages, can_paginate=False):
    client = PricingClient.__new__(PricingClient)
    fake = Mock()
    fake.can_paginate.return_value = can_paginate
    if can_paginate:
        fake.get_paginator.return_value = Mock(paginate=Mock(return_value=pages))
    else:
        fake.get_products.side_effect = pages
    client._client = fake
    return client, fake


def test_pricing_walks_past_the_first_page():
    """GetProducts caps at 100 products per page and AWSLambda publishes far
    more, so the target SKU is routinely not on page one."""
    pages = [
        {"PriceList": [_product("AWS-Lambda-Storage-Duration", ["0.1"])] * 100, "NextToken": "t1"},
        {
            "PriceList": [_product("AWS-Lambda-Duration", ["0.0000166667", "0.00002"])],
            "NextToken": "t2",
        },
        {"PriceList": [_product("AWS-Lambda-Requests", ["0.0000002"])]},
    ]
    client, fake = _pricing_client(pages)
    units = client.get_lambda_pricing_units("us-east-1")

    assert units == {"compute": 0.00002, "request": 0.0000002}
    assert fake.get_products.call_count == 3


def test_pricing_uses_the_paginator_when_botocore_offers_one():
    pages = [
        {"PriceList": [_product("AWS-Lambda-Duration", ["0.00002"])]},
        {"PriceList": [_product("AWS-Lambda-Requests", ["0.0000002"])]},
    ]
    client, fake = _pricing_client(pages, can_paginate=True)
    assert client.get_lambda_pricing_units("us-east-1")["compute"] == 0.00002
    fake.get_paginator.assert_called_once_with("get_products")


def test_arm_architecture_selects_the_arm_sku():
    pages = [
        {
            "PriceList": [
                _product("AWS-Lambda-Duration", ["0.00002"]),
                _product("AWS-Lambda-Duration-ARM", ["0.000016"]),
                _product("AWS-Lambda-Requests-ARM", ["0.0000002"]),
            ]
        }
    ]
    client, _ = _pricing_client(pages)
    units = client.get_lambda_pricing_units("us-east-1", architecture="arm64")
    assert units["compute"] == 0.000016


def test_pricing_reports_which_sku_it_could_not_find():
    from optiserve.exceptions import CostCalculationError

    pages = [{"PriceList": [_product("AWS-Lambda-Requests", ["0.0000002"])]}]
    client, _ = _pricing_client(pages)
    with pytest.raises(CostCalculationError, match="AWS-Lambda-Duration"):
        client.get_lambda_pricing_units("us-east-1")


# --------------------------------------------------------------------------- #
# Logs clients
# --------------------------------------------------------------------------- #
def _logs(cls, **kwargs):
    instance = cls.__new__(cls)
    instance._aws_logs_client = Mock()
    instance.log_parser = LogParser()
    instance._total_logs_limit = 10
    instance._poll_attempts = 3
    instance._poll_interval_s = 0.0
    for key, value in kwargs.items():
        setattr(instance, key, value)
    return instance


def test_a_failed_insights_query_raises_immediately_instead_of_polling_out():
    logs = _logs(AWSFunctionLogs, _function_name="f", _log_group_name="/aws/lambda/f")
    logs._aws_logs_client.start_query.return_value = {"queryId": "q"}
    logs._aws_logs_client.get_query_results.return_value = {"status": "Failed", "results": []}

    with pytest.raises(LogParsingError, match="Failed"):
        logs._run_query("/aws/lambda/f", "fields @message", 0, 1)
    logs._aws_logs_client.stop_query.assert_called_once_with(queryId="q")


def test_a_timed_out_query_is_stopped_not_abandoned():
    """An abandoned Insights query keeps consuming the account's small
    concurrent-query quota."""
    logs = _logs(AWSFunctionLogs, _function_name="f", _log_group_name="/aws/lambda/f")
    logs._aws_logs_client.start_query.return_value = {"queryId": "q"}
    logs._aws_logs_client.get_query_results.return_value = {"status": "Running", "results": []}

    with pytest.raises(FunctionTimeout):
        logs._run_query("/aws/lambda/f", "fields @message", 0, 1)
    logs._aws_logs_client.stop_query.assert_called_once_with(queryId="q")


def test_truncated_results_are_reported(caplog):
    import logging

    logs = _logs(AWSFunctionLogs, _function_name="f", _log_group_name="/aws/lambda/f")
    logs._aws_logs_client.start_query.return_value = {"queryId": "q"}
    logs._aws_logs_client.get_query_results.return_value = {
        "status": "Complete",
        "results": [[] for _ in range(10)],  # exactly at the limit
    }
    with caplog.at_level(logging.WARNING, logger="optiserve"):
        logs._run_query("/aws/lambda/f", "fields @message", 0, 1)
    assert any("truncated" in record.getMessage() for record in caplog.records)


def test_step_function_executions_outside_the_window_are_dropped():
    """A start event without its end used to produce a duration of minus the
    epoch timestamp and poison every aggregate computed from it."""
    logs = _logs(
        AWSApplicationLogs, _application_name="app", _log_group_name="/aws/vendedlogs/states/app"
    )

    def event(execution, kind, timestamp):
        message = json.dumps(
            {
                "type": kind,
                "execution_arn": f"arn:aws:states:us-east-1:1:execution:app:{execution}",
                "event_timestamp": timestamp,
            }
        )
        return [{"field": "@message", "value": message}]

    logs._aws_logs_client.start_query.return_value = {"queryId": "q"}
    logs._aws_logs_client.get_query_results.return_value = {
        "status": "Complete",
        "results": [
            event("complete", "ExecutionStarted", 1_700_000_000.0),
            event("complete", "ExecutionSucceeded", 1_700_000_002.5),
            event("straddling", "ExecutionSucceeded", 1_700_000_003.0),
        ],
    }

    executions = logs.get_logs(start_time=0, end_time=1)
    assert set(executions) == {"complete"}
    assert executions["complete"]["d"] == pytest.approx(2.5)


def test_function_logs_group_by_model_marker():
    logs = _logs(
        AWSFunctionLogs, _function_name="f", _log_group_name="/aws/lambda/f", docker_deploy=True
    )
    logs._aws_logs_client.start_query.return_value = {"queryId": "q"}
    logs._aws_logs_client.get_query_results.return_value = {
        "status": "Complete",
        "results": [
            [
                {"field": "@logStream", "value": "s1"},
                {"field": "@timestamp", "value": "1"},
                {
                    "field": "@message",
                    "value": "Model: resnet-50 - LogStream: s1 - Starting execution",
                },
            ],
            [
                {"field": "@logStream", "value": "s1"},
                {"field": "@timestamp", "value": "2"},
                {"field": "@message", "value": _REPORT.format(d=120, size=512, used=100)},
            ],
        ],
    }
    grouped = logs.get_logs(start_time=0, end_time=1)
    assert set(grouped) == {"resnet-50"}
    assert grouped["resnet-50"][0]["Duration"] == 120.0
