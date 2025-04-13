import boto3
import time
import re

import pandas as pd

from src.analytics.aws_logs import AWSLogs
from src.exceptions import (
    FunctionTimeout,
    MaxInvocationAttemptsReached,
)
from typing import Optional


class AWSFunctionLogs(AWSLogs):
    def __init__(
        self,
        boto_session: Optional[boto3.Session] = None,
        function_name: Optional[str] = None,
        total_logs_limit: int = 10000,
        docker_deploy: bool = False,
    ):
        if boto_session is None:
            boto_session = boto3.Session()
        super().__init__(boto_session=boto_session)
        if function_name is None:
            raise ValueError("function_name must be provided")

        self._function_name = function_name
        self._log_group_name = f"/aws/lambda/{self._function_name}"
        self._total_logs_limit = total_logs_limit
        self._max_invocation_attempts = 5
        self._sleep_interval = 1
        self.docker_deploy = docker_deploy

    def get_logs(self, start_time: int, end_time: int):
        """
        Retrieves logs from CloudWatch within the specified time range and groups log events by model.
        It detects a custom marker log of the form:
            "Model: {model_name} - LogStream: {log_stream_id} - Starting execution"
        and then assigns all log events (including AWS runtime logs) from that log stream to that model name.
        If no marker is found in any log stream (i.e. all logs are 'unknown'),
        returns a flat list of log events.
        Otherwise, returns a dict of lists where keys are model names and values are log events.
        """
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided")

        if self.docker_deploy:
            query_string = "fields @timestamp, @message, @logStream | filter type = 'platform.report' or (logger = 'root' and message like /Starting execution/) | sort @timestamp asc"
        else:
            query_string = "fields @timestamp, @message, @logStream| filter @message like 'REPORT'| sort @timestamp desc"

        response = self._aws_logs_client.start_query(
            logGroupName=self._log_group_name,
            queryString=query_string,
            startTime=start_time,
            endTime=end_time,
            limit=self._total_logs_limit,
        )

        query_id = response["queryId"]

        try:
            attempts = 0
            while attempts < self._max_invocation_attempts:
                response = self._aws_logs_client.get_query_results(queryId=query_id)
                if response["status"] == "Complete":
                    break

                time.sleep(self._sleep_interval)
                attempts += 1

            if response["status"] != "Complete":
                raise MaxInvocationAttemptsReached()

            # Group log events by their log stream.
            stream_logs = {}
            for row in response["results"]:
                event = {item["field"]: item["value"] for item in row}
                parsed_event = self.log_parser.parse_function_profiling_logs(
                    event.get("@message", "")
                )

                parsed_event["Timestamp"] = event.get("@timestamp", "")
                parsed_event["LogStream"] = event.get("@logStream", "")
                parsed_event["Ptr"] = event.get("@ptr", "")
                parsed_event["RawMessage"] = event.get("@message", "")

                log_stream = event.get("@logStream", "unknown")
                stream_logs.setdefault(log_stream, []).append(parsed_event)

            # Regex to match the custom log marker.
            # Expected format: "Model: {model_name} - LogStream: {log_stream_id} - Starting execution"
            pattern = re.compile(
                r"Model:\s*(.*?)\s*-\s*LogStream:\s*(.*?)\s*-\s*Starting execution"
            )

            logs_by_model = {}
            for log_stream, events in stream_logs.items():
                model_name = "unknown"
                for ev in events:
                    message = ev.get("RawMessage", "")
                    match = pattern.search(message)
                    if match:
                        extracted_model = match.group(1)
                        model_name = extracted_model
                        break

                relevant_events = [event for event in events if "Duration" in event]
                logs_by_model.setdefault(model_name, []).extend(relevant_events)

            # If no model marker was found in any log stream (i.e. only "unknown" exists),
            # then return a flat list of all log events.
            if set(logs_by_model.keys()) == {"unknown"}:
                all_logs = []
                for logs in logs_by_model.values():
                    all_logs.extend(logs)
                return all_logs

            return logs_by_model

        except MaxInvocationAttemptsReached:
            raise FunctionTimeout("Could not get the logs in time.")

    def get_logs_df(
        self, start_time: int, end_time: int, model_name: Optional[str] = None
    ):
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided")

        logs = self.get_logs(start_time=start_time, end_time=end_time)

        if isinstance(logs, dict):
            if model_name is not None and model_name in logs:
                logs = pd.DataFrame(logs[model_name])
            else:
                logs = pd.DataFrame()
                for model, log_events in logs.items():
                    df = pd.DataFrame(log_events)
                    df["Model"] = model
                    logs = pd.concat([logs, df], ignore_index=True)
        else:
            logs = pd.DataFrame(logs)

        return logs
