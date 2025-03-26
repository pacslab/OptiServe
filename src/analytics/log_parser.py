import re

from src.exceptions import (
    NotEnoughMemory,
    FunctionTimeout,
    LogParsingError,
    InvocationError,
)

from src.utils.logger import logger


class LogParser:
    def __init__(self):
        self._function_log_parsing_params = [
            "Duration",
            "Billed Duration",
            "Max Memory Used",
            "Memory Size",
            "Init Duration",
        ]

        self._patterns_map = {
            "Duration": [
                r"Duration:\s*(?P<value>[0-9.]+)\s*ms",  # old logs
                r'"durationMs"\s*:\s*(?P<value>[0-9.]+)',  # new logs
            ],
            "Billed Duration": [
                r"Billed Duration:\s*(?P<value>[0-9.]+)\s*ms",
                r'"billedDurationMs"\s*:\s*(?P<value>[0-9.]+)',
            ],
            "Max Memory Used": [
                r"Max Memory Used:\s*(?P<value>[0-9.]+)\s*MB",
                r'"maxMemoryUsedMB"\s*:\s*(?P<value>[0-9.]+)',
            ],
            "Memory Size": [
                r"Memory Size:\s*(?P<value>[0-9.]+)\s*MB",
                r'"memorySizeMB"\s*:\s*(?P<value>[0-9.]+)',
            ],
            "Init Duration": [
                r"Init Duration:\s*(?P<value>[0-9.]+)\s*ms",
                r'"initDurationMs"\s*:\s*(?P<value>[0-9.]+)',
            ],
        }

    def _get_function_invocation_logs(self, log: str):
        results = {}
        for param, patterns in self._patterns_map.items():
            for pattern in patterns:
                match = re.search(pattern, log)
                if match:
                    results[param] = float(match.group("value"))
                    break

        if "Billed Duration" not in results:
            raise LogParsingError()

        logger.info(f"Invocation Results: {results}")

        if "Task timed out after" in log:
            raise FunctionTimeout(duration_ms=int(results["Billed Duration"]))

        if results["Max Memory Used"] > results["Memory Size"]:
            raise NotEnoughMemory(duration_ms=int(results["Billed Duration"]))

        error_msg = re.match(r".*\[ERROR\] (?P<error>.*)END RequestId.*", log)
        if error_msg is not None:
            raise InvocationError(
                duration_ms=int(results["Billed Duration"]), message=error_msg["error"]
            )

        return results

    def parse_function_execution_time(self, log: str):
        try:
            results = self._get_function_invocation_logs(log)

            exec_time_ms = results["Billed Duration"]

        except InvocationError as e:
            exec_time_ms = e.duration_ms

        return exec_time_ms

    def parse_function_profiling_logs(self, log: str):
        results = {}
        for param, patterns in self._patterns_map.items():
            for pattern in patterns:
                match = re.search(pattern, log)
                if match:
                    results[param] = float(match.group("value"))
                    break

        logger.info(f"Profiling Results: {results}")

        return results
