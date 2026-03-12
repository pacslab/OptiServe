"""Parsing of AWS Lambda REPORT metrics from CloudWatch / tail logs.

Supports both the classic text REPORT format and the newer JSON
``platform.report`` format emitted by container-image functions.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from optiserve.exceptions import (
    FunctionTimeout,
    InvocationError,
    LogParsingError,
    NotEnoughMemory,
)
from optiserve.logging import get_logger

logger = get_logger(__name__)

# Two regexes per metric: classic text REPORT and JSON platform.report.
_PATTERNS_MAP: Dict[str, list] = {
    "Duration": [
        r"Duration:\s*(?P<value>[0-9.]+)\s*ms",
        r'"durationMs"\s*:\s*(?P<value>[0-9.]+)',
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


class LogParser:
    """Extracts numeric REPORT metrics and detects timeout/OOM/error markers."""

    def _extract(self, log: str) -> Dict[str, float]:
        """Pull every recognizable metric out of a single log message."""
        results: Dict[str, float] = {}
        for param, patterns in _PATTERNS_MAP.items():
            for pattern in patterns:
                match = re.search(pattern, log)
                if match:
                    results[param] = float(match.group("value"))
                    break
        return results

    def _get_function_invocation_logs(self, log: str) -> Dict[str, float]:
        """Extract metrics from a full invocation REPORT, raising the typed
        error when the log indicates a timeout, OOM, or application error."""
        results = self._extract(log)

        if "Billed Duration" not in results:
            raise LogParsingError()

        logger.info("Invocation results: %s", results)

        if "Task timed out after" in log:
            raise FunctionTimeout(duration_ms=int(results["Billed Duration"]))

        if results.get("Max Memory Used", 0) > results.get("Memory Size", float("inf")):
            raise NotEnoughMemory(duration_ms=int(results["Billed Duration"]))

        error_msg = re.match(r".*\[ERROR\] (?P<error>.*)END RequestId.*", log)
        if error_msg is not None:
            raise InvocationError(
                message=error_msg["error"], duration_ms=int(results["Billed Duration"])
            )

        return results

    def parse_function_execution_time(self, log: str) -> Optional[float]:
        """Return the billed duration (ms) for an invocation.

        A plain application ``InvocationError`` is treated as a completed (if
        failed) invocation and its billed duration is returned. Timeouts and
        OOM conditions (``FunctionTimeout`` / ``NotEnoughMemory``) are NOT
        swallowed — they propagate so the sampler can prune the memory space
        (previously they were caught here, silently defeating that pruning).
        """
        try:
            results = self._get_function_invocation_logs(log)
            return results["Billed Duration"]
        except (FunctionTimeout, NotEnoughMemory):
            raise
        except InvocationError as exc:
            return exc.duration_ms

    def parse_function_profiling_logs(self, log: str) -> Dict[str, float]:
        """Extract whatever metrics are present, without validation (used when
        aggregating many CloudWatch rows)."""
        results = self._extract(log)
        logger.info("Profiling results: %s", results)
        return results
