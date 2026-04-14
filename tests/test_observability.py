"""Tests for the profiling-hook layer.

The contract that matters most: observability must never be able to break a
run. A metrics backend that throws, a sink that blocks, an attribute that will
not serialize — none of them may abort an hour-long profiling job.
"""

import io
import json
import logging

import pytest

from optiserve.observability import (
    EmfSink,
    Event,
    EventName,
    HookRegistry,
    InMemorySink,
    JsonlSink,
    LoggingSink,
)


class ExplodingSink:
    def handle(self, event):
        raise RuntimeError("metrics backend is down")


def test_registry_with_no_sinks_is_a_noop():
    registry = HookRegistry()
    assert not registry.enabled
    registry.emit(EventName.SAMPLE_RECORDED, memory_mb=512)
    with registry.timed("x") as extra:
        extra["ignored"] = True  # discarded, but must not raise


def test_a_failing_sink_cannot_break_the_run():
    good = InMemorySink()
    registry = HookRegistry([ExplodingSink(), good])

    registry.emit(EventName.INVOCATION_COMPLETED, function="f")

    # The healthy sink still received the event.
    assert good.count(EventName.INVOCATION_COMPLETED) == 1


def test_timed_records_duration_and_late_attributes():
    sink = InMemorySink()
    registry = HookRegistry([sink])

    with registry.timed(EventName.INVOCATION_COMPLETED, function="f") as extra:
        extra["billed_ms"] = 120

    event = sink.of(EventName.INVOCATION_COMPLETED)[0]
    assert event.attributes["function"] == "f"
    assert event.attributes["billed_ms"] == 120
    assert event.duration_ms is not None and event.duration_ms >= 0


def test_timed_emits_on_failure_and_reraises():
    sink = InMemorySink()
    registry = HookRegistry([sink])

    with pytest.raises(ValueError), registry.timed("op", memory_mb=512):
        raise ValueError("boom")

    event = sink.of("op")[0]
    assert event.attributes["error"] == "ValueError"
    assert event.attributes["memory_mb"] == 512


def test_timed_does_not_swallow_keyboard_interrupt():
    registry = HookRegistry([InMemorySink()])
    with pytest.raises(KeyboardInterrupt), registry.timed("op"):
        raise KeyboardInterrupt


def test_event_is_immutable_and_composable():
    event = Event(name="a", attributes={"x": 1})
    extended = event.with_attributes(y=2)
    assert dict(event.attributes) == {"x": 1}  # original untouched
    assert dict(extended.attributes) == {"x": 1, "y": 2}


def test_jsonl_sink_writes_one_object_per_line_and_flushes():
    buffer = io.StringIO()
    sink = JsonlSink("unused", stream=buffer)
    sink.handle(Event(name="a", attributes={"memory_mb": 512}, duration_ms=1.5))
    sink.handle(Event(name="b", attributes={}))

    lines = buffer.getvalue().strip().split("\n")
    assert [json.loads(line)["event"] for line in lines] == ["a", "b"]
    assert json.loads(lines[0])["duration_ms"] == 1.5


def test_jsonl_sink_survives_unserializable_attributes():
    buffer = io.StringIO()
    sink = JsonlSink("unused", stream=buffer)
    sink.handle(Event(name="a", attributes={"session": object()}))
    # The event survives with a repr rather than being lost entirely.
    assert json.loads(buffer.getvalue())["event"] == "a"


def test_emf_sink_emits_cloudwatch_metric_metadata():
    buffer = io.StringIO()
    sink = EmfSink(stream=buffer)
    sink.handle(
        Event(
            name=EventName.SAMPLE_RECORDED,
            attributes={"function": "f", "model": "resnet-50", "memory_mb": 1024},
            duration_ms=42.0,
        )
    )
    payload = json.loads(buffer.getvalue())
    metrics = payload["_aws"]["CloudWatchMetrics"][0]
    assert metrics["Namespace"] == "OptiServe"
    assert metrics["Dimensions"] == [["function", "model"]]
    assert {m["Name"] for m in metrics["Metrics"]} == {"duration_ms", "memory_mb"}
    assert {m["Unit"] for m in metrics["Metrics"]} == {"Milliseconds", "Megabytes"}


def test_emf_sink_skips_events_with_no_numeric_fields():
    buffer = io.StringIO()
    EmfSink(stream=buffer).handle(Event(name="a", attributes={"function": "f"}))
    assert buffer.getvalue() == ""


def test_logging_sink_uses_the_optiserve_logger(caplog):
    with caplog.at_level(logging.INFO, logger="optiserve"):
        HookRegistry([LoggingSink()]).emit("a.b", function="f")
    rendered = [record.getMessage() for record in caplog.records]
    assert any("a.b" in message and "function='f'" in message for message in rendered)
    assert all(record.name.startswith("optiserve.") for record in caplog.records)
