"""CloudWatch Logs Insights adapters.

- :class:`AWSFunctionLogs` reconstructs per-invocation REPORT metrics for one
  Lambda (optionally grouped by ML model via a custom marker log).
- :class:`AWSApplicationLogs` reconstructs Step Functions execution durations.

The shared "start query then poll for results" loop lives once in the
:class:`AWSLogs` base class.

**Time units.** ``start_time`` and ``end_time`` are epoch *seconds*, matching
CloudWatch's ``StartQuery`` API — not the epoch milliseconds that Lambda log
events themselves carry. Mixing the two is silent: a millisecond window is a
window ending in the year 65 000, which returns everything, and a second-valued
timestamp compared against milliseconds returns nothing.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import boto3
import pandas as pd

from optiserve.aws.log_parser import LogParser
from optiserve.aws.session import create_client, create_session
from optiserve.exceptions import FunctionTimeout, LogParsingError
from optiserve.logging import get_logger

logger = get_logger(__name__)


class AWSLogs(ABC):
    """Base class holding the CloudWatch Logs client and the query loop."""

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        total_logs_limit: int = 10000,
        poll_attempts: int = 30,
        poll_interval_s: float = 2.0,
    ):
        if boto_session is None:
            boto_session = create_session()
        self._aws_logs_client = create_client(boto_session, "logs")
        self.log_parser = LogParser()
        self._total_logs_limit = total_logs_limit
        self._poll_attempts = poll_attempts
        self._poll_interval_s = poll_interval_s

    def _run_query(
        self, log_group_name: str, query_string: str, start_time: int, end_time: int
    ) -> list[list]:
        """Start a Logs Insights query and poll until it completes, returning
        the raw ``results`` rows.

        Three failure modes are distinguished, because they need different
        responses and used to be conflated into one ``FunctionTimeout``:

        * the query **failed or was cancelled** server-side — polling on is
          pointless, so it raises immediately;
        * the query **did not finish** inside the polling budget — it is
          explicitly stopped rather than left running (an abandoned Insights
          query keeps consuming the account's concurrent-query quota, and the
          quota is small); and
        * the query **hit the row limit** — the results are silently truncated,
          which would otherwise show up much later as missing profiling data.
        """
        response = self._aws_logs_client.start_query(
            logGroupName=log_group_name,
            queryString=query_string,
            startTime=start_time,
            endTime=end_time,
            limit=self._total_logs_limit,
        )
        query_id = response["queryId"]

        try:
            for _ in range(self._poll_attempts):
                response = self._aws_logs_client.get_query_results(queryId=query_id)
                status = response["status"]
                if status == "Complete":
                    results = response["results"]
                    if len(results) >= self._total_logs_limit:
                        logger.warning(
                            "Logs Insights returned %d rows, at the configured limit "
                            "of %d: results are truncated and profiling data may be "
                            "missing. Narrow the time window or raise "
                            "total_logs_limit.",
                            len(results),
                            self._total_logs_limit,
                        )
                    return results
                if status in ("Failed", "Cancelled", "Timeout"):
                    raise LogParsingError(
                        f"CloudWatch Logs Insights query {query_id} ended with status {status!r}."
                    )
                time.sleep(self._poll_interval_s)
        except Exception:
            self._stop_query(query_id)
            raise

        self._stop_query(query_id)
        raise FunctionTimeout("Could not get the logs in time.")

    def _stop_query(self, query_id: str) -> None:
        """Best-effort cancellation of a still-running query."""
        try:
            self._aws_logs_client.stop_query(queryId=query_id)
        except Exception:
            logger.debug("stop_query(%s) failed", query_id, exc_info=True)

    @abstractmethod
    def get_logs(self, start_time: int, end_time: int) -> Any:
        """Fetch logs for the window. Times are epoch **seconds** (see module doc)."""
        ...


class AWSFunctionLogs(AWSLogs):
    """Per-invocation REPORT metrics for a single Lambda function."""

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        function_name: str | None = None,
        total_logs_limit: int = 10000,
        docker_deploy: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(boto_session=boto_session, total_logs_limit=total_logs_limit, **kwargs)
        if function_name is None:
            raise ValueError("function_name must be provided")
        self._function_name = function_name
        self._log_group_name = f"/aws/lambda/{self._function_name}"
        self.docker_deploy = docker_deploy

    # Custom marker the profiled Lambda prints so we can group logs by model:
    #   "Model: {model_name} - LogStream: {log_stream_id} - Starting execution"
    _MODEL_MARKER = re.compile(r"Model:\s*(.*?)\s*-\s*LogStream:\s*(.*?)\s*-\s*Starting execution")

    def get_logs(self, start_time: int, end_time: int) -> list[dict] | dict[str, list[dict]]:
        """Return per-invocation metric dicts.

        If a model marker is found, returns ``{model_name: [events]}``; otherwise
        a flat ``[events]`` list.
        """
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided")

        if self.docker_deploy:
            query_string = (
                "fields @timestamp, @message, @logStream | filter type = "
                "'platform.report' or (logger = 'root' and message like "
                "/Starting execution/) | sort @timestamp asc"
            )
        else:
            query_string = (
                "fields @timestamp, @message, @logStream | filter @message like "
                "'REPORT' | sort @timestamp desc"
            )

        results = self._run_query(self._log_group_name, query_string, start_time, end_time)

        # Group events by log stream.
        stream_logs: dict[str, list[dict]] = {}
        for row in results:
            event = {item["field"]: item["value"] for item in row}
            parsed = self.log_parser.parse_function_profiling_logs(event.get("@message", ""))
            parsed["Timestamp"] = event.get("@timestamp", "")
            parsed["LogStream"] = event.get("@logStream", "")
            parsed["Ptr"] = event.get("@ptr", "")
            parsed["RawMessage"] = event.get("@message", "")
            stream_logs.setdefault(event.get("@logStream", "unknown"), []).append(parsed)

        # Assign a model name per stream from the marker log, keep metric rows.
        logs_by_model: dict[str, list[dict]] = {}
        for events in stream_logs.values():
            model_name = "unknown"
            for ev in events:
                match = self._MODEL_MARKER.search(ev.get("RawMessage", ""))
                if match:
                    model_name = match.group(1)
                    break
            relevant = [ev for ev in events if "Duration" in ev]
            logs_by_model.setdefault(model_name, []).extend(relevant)

        if set(logs_by_model.keys()) == {"unknown"}:
            return [ev for events in logs_by_model.values() for ev in events]
        return logs_by_model

    def get_logs_df(
        self, start_time: int, end_time: int, model_name: str | None = None
    ) -> pd.DataFrame:
        """Return the logs as a DataFrame. When grouped by model and no
        ``model_name`` is given, all models are concatenated with a ``Model``
        column."""
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided")

        logs = self.get_logs(start_time=start_time, end_time=end_time)

        if isinstance(logs, dict):
            if model_name is not None and model_name in logs:
                return pd.DataFrame(logs[model_name])
            frames = []
            for model, events in logs.items():
                frame = pd.DataFrame(events)
                frame["Model"] = model
                frames.append(frame)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return pd.DataFrame(logs)


class AWSApplicationLogs(AWSLogs):
    """Step Functions execution durations from vended CloudWatch logs."""

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        application_name: str | None = None,
        total_logs_limit: int = 10000,
        **kwargs: Any,
    ) -> None:
        super().__init__(boto_session=boto_session, total_logs_limit=total_logs_limit, **kwargs)
        if application_name is None:
            raise ValueError("application_name must be provided")
        self._application_name = application_name
        self._log_group_name = f"/aws/vendedlogs/states/{self._application_name}"

    def get_logs(self, start_time: int, end_time: int) -> dict[str, dict[str, float]]:
        """Return ``{execution_name: {'s': start, 'e': end, 'd': duration}}``."""
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided")

        query_string = (
            "fields @timestamp, @message | filter type = 'ExecutionStarted' or "
            "type = 'ExecutionSucceeded' | sort id desc"
        )
        results = self._run_query(self._log_group_name, query_string, start_time, end_time)

        executions: dict[str, dict[str, float]] = defaultdict(
            lambda: {"s": 0.0, "e": 0.0, "d": 0.0}
        )
        for row in results:
            fields = {item["field"]: item["value"] for item in row}
            message = json.loads(fields["@message"])
            if message["type"] in ("ExecutionStarted", "ExecutionSucceeded"):
                execution = message["execution_arn"].split(":")[-1]
                timestamp = float(message["event_timestamp"])
                key = "s" if message["type"] == "ExecutionStarted" else "e"
                executions[execution][key] = timestamp

        # An execution that straddles the query window contributes only one of
        # its two events, leaving the other at the 0.0 default — which produced
        # a duration of +/- the epoch timestamp (order 1e12 ms) and silently
        # poisoned every aggregate computed from these numbers. Report only
        # executions the window fully contains, and say how many were dropped.
        complete = {
            name: value
            for name, value in executions.items()
            if value["s"] > 0.0 and value["e"] > 0.0
        }
        dropped = len(executions) - len(complete)
        if dropped:
            logger.warning(
                "Ignoring %d Step Functions execution(s) not fully contained in "
                "the query window (missing a start or end event).",
                dropped,
            )

        for value in complete.values():
            value["d"] = value["e"] - value["s"]
        return complete
