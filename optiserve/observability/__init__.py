"""Profiling hooks and event sinks.

Importing this package registers nothing and emits nothing — the process-wide
:data:`~optiserve.observability.hooks.hooks` registry starts empty. Opt in:

    from optiserve.observability import JsonlSink, hooks
    hooks.add(JsonlSink("output/run.jsonl"))
"""

from optiserve.observability.events import Event, EventName
from optiserve.observability.hooks import HookRegistry, Sink, hooks
from optiserve.observability.sinks import EmfSink, InMemorySink, JsonlSink, LoggingSink

__all__ = [
    "EmfSink",
    "Event",
    "EventName",
    "HookRegistry",
    "InMemorySink",
    "JsonlSink",
    "LoggingSink",
    "Sink",
    "hooks",
]
