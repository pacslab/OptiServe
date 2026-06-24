"""Integration: CloudWatch Logs harvesting against a mocked account.

moto implements Logs Insights (``StartQuery``/``GetQueryResults``), so this
exercises the real query strings OptiServe sends and the real parsing of what
comes back — the layer that turns a deployed function's REPORT lines into
profiling samples.
"""

import json
import time

import pytest

from optiserve.aws.logs_client import AWSApplicationLogs, AWSFunctionLogs

pytestmark = pytest.mark.integration

# CloudWatch Logs Insights StartQuery takes epoch *seconds*, not milliseconds
# (moto models this faithfully). Query a window that brackets "now".
_START = int(time.time()) - 3600
_END = int(time.time()) + 3600


def _report(duration, memory=512, used=100):
    return (
        f"REPORT RequestId: r\tDuration: {duration}.00 ms\t"
        f"Billed Duration: {duration} ms\tMemory Size: {memory} MB\t"
        f"Max Memory Used: {used} MB\t"
    )


def test_function_report_metrics_are_recovered(session, log_group):
    log_group("/aws/lambda/inference", [_report(120), _report(140), _report(160)])

    logs = AWSFunctionLogs(boto_session=session, function_name="inference", poll_interval_s=0.0)
    events = logs.get_logs(start_time=_START, end_time=_END)

    assert isinstance(events, list)
    assert sorted(event["Billed Duration"] for event in events) == [120.0, 140.0, 160.0]
    assert all(event["Memory Size"] == 512.0 for event in events)


def test_function_logs_as_a_dataframe(session, log_group):
    log_group("/aws/lambda/inference", [_report(120), _report(140)])

    logs = AWSFunctionLogs(boto_session=session, function_name="inference", poll_interval_s=0.0)
    frame = logs.get_logs_df(start_time=_START, end_time=_END)

    assert len(frame) == 2
    assert {"Duration", "Billed Duration", "Memory Size", "LogStream"} <= set(frame.columns)


def test_step_function_durations_are_computed_from_paired_events(session, aws):
    logs_client = session.client("logs")
    group = "/aws/vendedlogs/states/pipeline"
    logs_client.create_log_group(logGroupName=group)
    logs_client.create_log_stream(logGroupName=group, logStreamName="s")

    def event(execution, kind, timestamp):
        return json.dumps(
            {
                "type": kind,
                "execution_arn": f"arn:aws:states:us-east-1:1:execution:pipeline:{execution}",
                "event_timestamp": timestamp,
            }
        )

    now_ms = int(time.time() * 1000)
    logs_client.put_log_events(
        logGroupName=group,
        logStreamName="s",
        logEvents=[
            {"timestamp": now_ms, "message": event("e1", "ExecutionStarted", 1000.0)},
            {"timestamp": now_ms + 1, "message": event("e1", "ExecutionSucceeded", 1002.5)},
            # An execution whose start fell outside the window: reporting it
            # would yield a duration of +1002 s of epoch nonsense.
            {"timestamp": now_ms + 2, "message": event("e2", "ExecutionSucceeded", 1003.0)},
        ],
    )

    logs = AWSApplicationLogs(
        boto_session=session, application_name="pipeline", poll_interval_s=0.0
    )
    executions = logs.get_logs(start_time=_START, end_time=_END)

    assert set(executions) == {"e1"}
    assert executions["e1"]["d"] == pytest.approx(2.5)


def test_missing_time_window_is_rejected(session, log_group):
    log_group("/aws/lambda/inference", [_report(120)])
    logs = AWSFunctionLogs(boto_session=session, function_name="inference")
    with pytest.raises(ValueError):
        logs.get_logs(start_time=None, end_time=_END)
