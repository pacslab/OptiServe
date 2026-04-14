"""Concrete event sinks.

Four, covering the situations OptiServe actually runs in:

``InMemorySink``   assertions in tests
``LoggingSink``    a human watching a profiling run in a terminal
``JsonlSink``      the durable per-run audit trail written next to the results
``EmfSink``        CloudWatch Embedded Metric Format, for runs driven from
                   inside Lambda/Fargate where stdout *is* the metrics pipeline
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

from optiserve.logging import get_logger
from optiserve.observability.events import Event

__all__ = ["EmfSink", "InMemorySink", "JsonlSink", "LoggingSink"]


class InMemorySink:
    """Keeps every event in a list. For tests and notebooks."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    # -- query helpers used by tests ---------------------------------------- #
    def names(self) -> list[str]:
        return [event.name for event in self.events]

    def of(self, name: str) -> list[Event]:
        return [event for event in self.events if event.name == name]

    def count(self, name: str) -> int:
        return sum(1 for event in self.events if event.name == name)

    def clear(self) -> None:
        with self._lock:
            self.events.clear()


class LoggingSink:
    """Renders events through the standard ``optiserve`` logger."""

    def __init__(self, level: int = logging.INFO, logger_name: str = "observability") -> None:
        self._level = level
        self._logger = get_logger(logger_name)

    def handle(self, event: Event) -> None:
        attributes = " ".join(f"{k}={v!r}" for k, v in event.attributes.items())
        if event.duration_ms is not None:
            self._logger.log(
                self._level, "%s (%.1f ms) %s", event.name, event.duration_ms, attributes
            )
        else:
            self._logger.log(self._level, "%s %s", event.name, attributes)


class JsonlSink:
    """Appends one JSON object per line — the durable trace of a run.

    Opened in append mode and flushed per event so a run killed mid-flight still
    leaves a usable trail (the exact failure mode this exists for). Writes are
    serialized because the profiler invokes Lambda from a thread pool.
    """

    def __init__(self, path: str | Path, *, stream: TextIO | None = None) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        if stream is not None:
            self._stream: TextIO | None = stream
            self._owns_stream = False
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._path.open("a", encoding="utf-8")
            self._owns_stream = True

    def handle(self, event: Event) -> None:
        if self._stream is None:
            return
        line = json.dumps(event.to_dict(), default=_fallback)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._stream is not None and self._owns_stream:
                self._stream.close()
            self._stream = None

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class EmfSink:
    """CloudWatch Embedded Metric Format.

    When OptiServe runs inside Lambda or Fargate, anything printed to stdout in
    EMF is ingested by CloudWatch as a metric with no extra API call — which
    matters because the alternative (``PutMetricData`` per sample) would add an
    API call, and a cost, to every single profiling invocation.

    ``metric_fields`` names the numeric attributes to publish as metrics;
    everything else becomes a dimension or a plain property.
    """

    def __init__(
        self,
        namespace: str = "OptiServe",
        *,
        dimensions: Sequence[str] = ("function", "model"),
        metric_fields: Iterable[str] = ("duration_ms", "memory_mb", "cost", "response_time_ms"),
        stream: TextIO | None = None,
    ) -> None:
        self._namespace = namespace
        self._dimensions = tuple(dimensions)
        self._metric_fields = tuple(metric_fields)
        self._stream = stream
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        payload: dict[str, Any] = event.to_dict()

        metrics = [
            {"Name": field, "Unit": _emf_unit(field)}
            for field in self._metric_fields
            if isinstance(payload.get(field), (int, float)) and not isinstance(payload[field], bool)
        ]
        if not metrics:
            return

        present_dimensions = [d for d in self._dimensions if isinstance(payload.get(d), str)]
        payload["_aws"] = {
            "Timestamp": int(event.timestamp * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": self._namespace,
                    "Dimensions": [present_dimensions] if present_dimensions else [[]],
                    "Metrics": metrics,
                }
            ],
        }

        line = json.dumps(payload, default=_fallback)
        with self._lock:
            if self._stream is not None:
                self._stream.write(line + "\n")
                self._stream.flush()
            else:
                print(line, flush=True)  # noqa: T201 — EMF *is* stdout by design


def _emf_unit(field: str) -> str:
    if field.endswith("_ms"):
        return "Milliseconds"
    if field.endswith("_mb") or field == "memory_mb":
        return "Megabytes"
    return "None"


def _fallback(value: object) -> str:
    """Last-resort JSON encoder — an unserializable attribute must not lose the
    whole event."""
    return repr(value)
