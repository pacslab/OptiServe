"""Parsing of AWS Lambda REPORT metrics from CloudWatch / tail logs.

Supports both the classic text REPORT format and the newer JSON
``platform.report`` format emitted by container-image functions.
"""

from __future__ import annotations

import re

from optiserve.exceptions import (
    FunctionTimeout,
    InvocationError,
    LogParsingError,
    NotEnoughMemory,
)
from optiserve.logging import get_logger

logger = get_logger(__name__)

# Two regexes per metric: classic text REPORT and JSON platform.report.
_PATTERNS_MAP: dict[str, list] = {
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


# Markers Lambda actually emits when a function runs out of memory.
#
# The numeric check `Max Memory Used > Memory Size` can never fire against real
# AWS: the platform clamps the reported "Max Memory Used" at the configured
# "Memory Size", so an OOM invocation reports them as *equal*, not greater. That
# left the sampler's memory-floor pruning — the mechanism that makes profiling
# converge on a feasible range — effectively dead against a live function while
# still working in synthetic tests. These markers are the signal that does fire.
_OOM_MARKERS = (
    "Runtime exited with error: signal: killed",
    "Runtime.OutOfMemory",
    "Error: Runtime exited with error: signal: killed",
    "MemoryError",
)


class LogParser:
    """Extracts numeric REPORT metrics and detects timeout/OOM/error markers."""

    def _extract(self, log: str) -> dict[str, float]:
        """Pull every recognizable metric out of a single log message."""
        results: dict[str, float] = {}
        for param, patterns in _PATTERNS_MAP.items():
            for pattern in patterns:
                match = re.search(pattern, log)
                if match:
                    results[param] = float(match.group("value"))
                    break
        return results

    def _get_function_invocation_logs(self, log: str) -> dict[str, float]:
        """Extract metrics from a full invocation REPORT, raising the typed
        error when the log indicates a timeout, OOM, or application error."""
        results = self._extract(log)

        if "Billed Duration" not in results:
            raise LogParsingError()

        logger.info("Invocation results: %s", results)

        if "Task timed out after" in log:
            raise FunctionTimeout(duration_ms=int(results["Billed Duration"]))

        # Two independent OOM signals. The numeric comparison catches synthetic
        # and container-runtime logs; the markers catch real AWS, where the
        # reported usage is clamped at the limit and the comparison never fires.
        max_used = results.get("Max Memory Used", 0)
        memory_size = results.get("Memory Size", float("inf"))
        hit_limit = any(marker in log for marker in _OOM_MARKERS)
        if max_used > memory_size or hit_limit:
            raise NotEnoughMemory(duration_ms=int(results["Billed Duration"]))

        # DOTALL: a tail log is multi-line, so without it `.` stops at the first
        # newline and this pattern only ever matched single-line logs.
        error_msg = re.match(r".*\[ERROR\] (?P<error>.*?)END RequestId.*", log, flags=re.DOTALL)
        if error_msg is not None:
            raise InvocationError(
                message=error_msg["error"], duration_ms=int(results["Billed Duration"])
            )

        return results

    def parse_function_execution_time(self, log: str) -> float | None:
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

    def parse_function_profiling_logs(self, log: str) -> dict[str, float]:
        """Extract whatever metrics are present, without validation (used when
        aggregating many CloudWatch rows)."""
        results = self._extract(log)
        logger.info("Profiling results: %s", results)
        return results
