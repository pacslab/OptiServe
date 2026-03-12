"""CloudWatch Logs Insights adapters.

- :class:`AWSFunctionLogs` reconstructs per-invocation REPORT metrics for one
  Lambda (optionally grouped by ML model via a custom marker log).
- :class:`AWSApplicationLogs` reconstructs Step Functions execution durations.

The shared "start query then poll for results" loop lives once in the
:class:`AWSLogs` base class.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Union

import boto3
import pandas as pd

from optiserve.aws.log_parser import LogParser
from optiserve.aws.session import create_session
from optiserve.exceptions import FunctionTimeout
from optiserve.logging import get_logger

logger = get_logger(__name__)


class AWSLogs(ABC):
    """Base class holding the CloudWatch Logs client and the query loop."""

    def __init__(
        self,
        boto_session: Optional[boto3.Session] = None,
        total_logs_limit: int = 10000,
        poll_attempts: int = 30,
        poll_interval_s: float = 2.0,
    ):
        if boto_session is None:
            boto_session = create_session()
        self._aws_logs_client = boto_session.client("logs")
        self.log_parser = LogParser()
        self._total_logs_limit = total_logs_limit
        self._poll_attempts = poll_attempts
        self._poll_interval_s = poll_interval_s

    def _run_query(
        self, log_group_name: str, query_string: str, start_time: int, end_time: int
    ) -> List[list]:
        """Start a Logs Insights query and poll until it completes, returning
        the raw ``results`` rows. Raises ``FunctionTimeout`` if the query does
        not finish within the polling budget."""
        response = self._aws_logs_client.start_query(
            logGroupName=log_group_name,
            queryString=query_string,
            startTime=start_time,
            endTime=end_time,
            limit=self._total_logs_limit,
        )
        query_id = response["queryId"]

        for _ in range(self._poll_attempts):
            response = self._aws_logs_client.get_query_results(queryId=query_id)
            if response["status"] == "Complete":
                return response["results"]
            time.sleep(self._poll_interval_s)

        raise FunctionTimeout("Could not get the logs in time.")

    @abstractmethod
    def get_logs(self, start_time: int, end_time: int):
        ...


class AWSFunctionLogs(AWSLogs):
    """Per-invocation REPORT metrics for a single Lambda function."""

    def __init__(
        self,
        boto_session: Optional[boto3.Session] = None,
        function_name: Optional[str] = None,
        total_logs_limit: int = 10000,
        docker_deploy: bool = False,
        **kwargs,
    ):
        super().__init__(
            boto_session=boto_session, total_logs_limit=total_logs_limit, **kwargs
        )
        if function_name is None:
            raise ValueError("function_name must be provided")
        self._function_name = function_name
        self._log_group_name = f"/aws/lambda/{self._function_name}"
        self.docker_deploy = docker_deploy

    # Custom marker the profiled Lambda prints so we can group logs by model:
    #   "Model: {model_name} - LogStream: {log_stream_id} - Starting execution"
    _MODEL_MARKER = re.compile(
        r"Model:\s*(.*?)\s*-\s*LogStream:\s*(.*?)\s*-\s*Starting execution"
    )

    def get_logs(
        self, start_time: int, end_time: int
    ) -> Union[List[dict], Dict[str, List[dict]]]:
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

        results = self._run_query(
            self._log_group_name, query_string, start_time, end_time
        )

        # Group events by log stream.
        stream_logs: Dict[str, List[dict]] = {}
        for row in results:
            event = {item["field"]: item["value"] for item in row}
            parsed = self.log_parser.parse_function_profiling_logs(
                event.get("@message", "")
            )
            parsed["Timestamp"] = event.get("@timestamp", "")
            parsed["LogStream"] = event.get("@logStream", "")
            parsed["Ptr"] = event.get("@ptr", "")
            parsed["RawMessage"] = event.get("@message", "")
            stream_logs.setdefault(event.get("@logStream", "unknown"), []).append(parsed)

        # Assign a model name per stream from the marker log, keep metric rows.
        logs_by_model: Dict[str, List[dict]] = {}
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
        self, start_time: int, end_time: int, model_name: Optional[str] = None
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
            return (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )
        return pd.DataFrame(logs)


class AWSApplicationLogs(AWSLogs):
    """Step Functions execution durations from vended CloudWatch logs."""

    def __init__(
        self,
        boto_session: Optional[boto3.Session] = None,
        application_name: Optional[str] = None,
        total_logs_limit: int = 10000,
        **kwargs,
    ):
        super().__init__(
            boto_session=boto_session, total_logs_limit=total_logs_limit, **kwargs
        )
        if application_name is None:
            raise ValueError("application_name must be provided")
        self._application_name = application_name
        self._log_group_name = f"/aws/vendedlogs/states/{self._application_name}"

    def get_logs(
        self, start_time: int, end_time: int
    ) -> Dict[str, Dict[str, float]]:
        """Return ``{execution_name: {'s': start, 'e': end, 'd': duration}}``."""
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided")

        query_string = (
            "fields @timestamp, @message | filter type = 'ExecutionStarted' or "
            "type = 'ExecutionSucceeded' | sort id desc"
        )
        results = self._run_query(
            self._log_group_name, query_string, start_time, end_time
        )

        executions: Dict[str, Dict[str, float]] = defaultdict(
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

        for value in executions.values():
            value["d"] = value["e"] - value["s"]
        return dict(executions)
