"""The event vocabulary emitted by OptiServe's profiling and optimization loops.

A profiling run costs real money and real wall-clock time: it mutates a live
Lambda's configuration and invokes it hundreds of times. Before this module the
only record of what happened was unstructured log lines, so a run that produced
a bad curve could not be explained after the fact.

Every interesting transition now emits a typed :class:`Event`. Events are
*facts about a run*, never control flow — emitting is best-effort and a failing
sink can never fail the run (see :mod:`optiserve.observability.hooks`).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Event", "EventName"]


class EventName:
    """Canonical event names. String constants rather than an ``Enum`` so sinks
    can forward them to metric backends without conversion."""

    # -- profiling / measurement ------------------------------------------- #
    RUN_STARTED = "profiling.run.started"
    RUN_FINISHED = "profiling.run.finished"
    CONFIG_APPLIED = "profiling.config.applied"
    CONFIG_RESTORED = "profiling.config.restored"
    INVOCATION_COMPLETED = "profiling.invocation.completed"
    INVOCATION_FAILED = "profiling.invocation.failed"
    SAMPLE_RECORDED = "profiling.sample.recorded"
    MEMORY_PRUNED = "profiling.memory.pruned"
    CHECKPOINT_SAVED = "profiling.checkpoint.saved"
    CHECKPOINT_RESUMED = "profiling.checkpoint.resumed"

    # -- curve fitting ------------------------------------------------------ #
    FIT_UPDATED = "modeling.fit.updated"
    FIT_FAILED = "modeling.fit.failed"
    ACQUISITION_SELECTED = "modeling.acquisition.selected"

    # -- application optimization ------------------------------------------- #
    OPTIMIZATION_STARTED = "optimization.started"
    OPTIMIZATION_FINISHED = "optimization.finished"
    OPTIMIZATION_STEP = "optimization.step"
    MODEL_EVALUATION = "optimization.model_evaluation"
    CACHE_STATS = "optimization.cache_stats"


@dataclass(frozen=True)
class Event:
    """One observation from a run.

    ``attributes`` carries the dimensional payload (function name, memory size,
    model variant, strategy, …). Keep values primitive — sinks serialize them to
    JSON or to a metrics backend without custom encoders.
    """

    name: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    timestamp: float = field(default_factory=time.time)

    def with_attributes(self, **extra: Any) -> Event:
        """Return a copy carrying additional attributes (the event is frozen)."""
        merged = dict(self.attributes)
        merged.update(extra)
        return Event(
            name=self.name,
            attributes=merged,
            duration_ms=self.duration_ms,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": self.name,
            "timestamp": self.timestamp,
            **dict(self.attributes),
        }
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
        return payload
