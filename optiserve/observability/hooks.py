"""Profiling hooks: the dispatch seam between OptiServe's loops and any sink.

Design constraints, in priority order:

1. **A hook can never change a run.** Sinks are called inside a ``try/except``
   that swallows everything but ``KeyboardInterrupt``/``SystemExit``. A broken
   metrics backend must not abort an hour-long, money-costing profiling job.
2. **Zero cost when unused.** With no sinks registered, :meth:`HookRegistry.emit`
   is a single truthiness check, and :meth:`~HookRegistry.timed` avoids building
   the attribute dict at all.
3. **No global mutable default.** Components take a registry by dependency
   injection and fall back to the process-wide :data:`hooks` only when the
   caller passes nothing, so tests can isolate completely.

    from optiserve.observability import EventName, HookRegistry, InMemorySink

    sink = InMemorySink()
    registry = HookRegistry([sink])
    with registry.timed(EventName.INVOCATION_COMPLETED, function="f", memory_mb=512):
        ...
    sink.names()  # ['profiling.invocation.completed']
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from optiserve.logging import get_logger
from optiserve.observability.events import Event

logger = get_logger(__name__)

__all__ = ["HookRegistry", "Sink", "hooks"]


@runtime_checkable
class Sink(Protocol):
    """Anything that can receive events. Implementations must not raise; the
    registry guards against it anyway, but a raising sink loses its own event."""

    def handle(self, event: Event) -> None:  # pragma: no cover - protocol
        ...


class HookRegistry:
    """Fan-out of :class:`~optiserve.observability.events.Event` to sinks."""

    def __init__(self, sinks: Iterable[Sink] | None = None) -> None:
        self._sinks: list[Sink] = list(sinks) if sinks else []

    # -- registration -------------------------------------------------------- #
    def add(self, sink: Sink) -> HookRegistry:
        self._sinks.append(sink)
        return self

    def remove(self, sink: Sink) -> HookRegistry:
        if sink in self._sinks:
            self._sinks.remove(sink)
        return self

    def clear(self) -> HookRegistry:
        self._sinks.clear()
        return self

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    def __len__(self) -> int:
        return len(self._sinks)

    # -- emission ------------------------------------------------------------ #
    def emit(self, name: str, *, duration_ms: float | None = None, **attributes: Any) -> None:
        """Emit an event. Cheap no-op when nothing is listening."""
        if not self._sinks:
            return
        self.dispatch(Event(name=name, attributes=attributes, duration_ms=duration_ms))

    def dispatch(self, event: Event) -> None:
        for sink in self._sinks:
            try:
                sink.handle(event)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                logger.debug("observability sink %r failed on %s", sink, event.name, exc_info=True)

    @contextmanager
    def timed(self, name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
        """Time a block and emit ``name`` with its wall-clock duration.

        The yielded dict is mutable: add attributes discovered *inside* the block
        (a measured duration, a returned status) and they land on the event.
        On an exception the event is still emitted, with ``error`` set, and the
        exception propagates unchanged.
        """
        if not self._sinks:
            yield {}
            return

        extra: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            yield extra
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.dispatch(
                Event(
                    name=name,
                    attributes={**attributes, **extra, "error": type(exc).__name__},
                    duration_ms=elapsed,
                )
            )
            raise
        else:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.dispatch(Event(name=name, attributes={**attributes, **extra}, duration_ms=elapsed))


#: Process-wide registry used when a component is constructed without one.
#: Empty by default — importing OptiServe emits nothing anywhere.
hooks = HookRegistry()
