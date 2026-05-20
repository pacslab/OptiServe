"""Durable state for a profiling run, so an interrupted run can be resumed.

Profiling a single ML function across its memory space costs hundreds of Lambda
invocations and tens of minutes; profiling a workflow's worth of model variants
costs hours and real money. Before this module a run held everything in memory:
a dropped connection, a throttling storm or a ``Ctrl-C`` at minute fifty threw
away every sample.

A checkpoint records the samples already collected and the memory space as it
has been pruned, keyed by a *run id* that includes a fingerprint of the run's
parameters. Resuming with different parameters is refused rather than silently
mixing incomparable measurements into one curve — the failure mode that would
quietly corrupt a published result.

    store = JsonCheckpointStore("output/checkpoints")
    sampler = Sampler(explorer, profiling_iterations=4, checkpoint_store=store)
    sampler.exploration_init("resnet-50")   # resumes if a checkpoint exists
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from optiserve.logging import get_logger
from optiserve.profiling.sample import Sample

logger = get_logger(__name__)

__all__ = [
    "CheckpointStore",
    "JsonCheckpointStore",
    "NullCheckpointStore",
    "ProfilingState",
    "run_id_for",
]

#: Bumped when the on-disk shape changes; older checkpoints are ignored, not
#: misread.
STATE_VERSION = 1


def run_id_for(function_name: str, model_name: str, **parameters: Any) -> str:
    """A stable id for one (function, model, parameters) profiling run.

    The parameter fingerprint is part of the id, so changing the payload,
    iteration count or memory bounds starts a *new* run rather than appending
    incomparable samples to the old one.
    """
    fingerprint = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, default=repr).encode("utf-8")
    ).hexdigest()[:12]
    safe_function = "".join(c if c.isalnum() or c in "-_." else "_" for c in function_name)
    safe_model = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    return f"{safe_function}__{safe_model}__{fingerprint}"


@dataclass
class ProfilingState:
    """Everything needed to continue a partially completed profiling run."""

    run_id: str
    function_name: str
    model_name: str
    samples: list[Sample] = field(default_factory=list)
    memory_space: list[int] = field(default_factory=list)
    version: int = STATE_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "function_name": self.function_name,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "memory_space": [int(m) for m in self.memory_space],
            "samples": [
                {"memory_mb": int(s.memory_mb), "duration_ms": float(s.duration_ms)}
                for s in self.samples
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProfilingState:
        return cls(
            run_id=payload["run_id"],
            function_name=payload["function_name"],
            model_name=payload["model_name"],
            samples=[
                Sample(memory_mb=int(s["memory_mb"]), duration_ms=float(s["duration_ms"]))
                for s in payload.get("samples", [])
            ],
            memory_space=[int(m) for m in payload.get("memory_space", [])],
            version=int(payload.get("version", 0)),
            created_at=float(payload.get("created_at", time.time())),
            updated_at=float(payload.get("updated_at", time.time())),
        )


@runtime_checkable
class CheckpointStore(Protocol):
    """Persistence for :class:`ProfilingState`."""

    def load(self, run_id: str) -> ProfilingState | None:  # pragma: no cover - protocol
        ...

    def save(self, state: ProfilingState) -> None:  # pragma: no cover - protocol
        ...

    def delete(self, run_id: str) -> None:  # pragma: no cover - protocol
        ...


class NullCheckpointStore:
    """The default: remembers nothing. Keeps checkpointing strictly opt-in, so
    existing callers see byte-identical behaviour."""

    def load(self, run_id: str) -> ProfilingState | None:
        return None

    def save(self, state: ProfilingState) -> None:
        return None

    def delete(self, run_id: str) -> None:
        return None


class JsonCheckpointStore:
    """One JSON file per run, written atomically.

    Atomicity matters: the process being killed mid-write is the exact scenario
    checkpoints exist for, and a truncated file would turn a recoverable
    interruption into a corrupt one. Writes go to a temp file in the same
    directory and are then ``os.replace``d, which is atomic on POSIX and Windows.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        return self._directory / f"{run_id}.json"

    def load(self, run_id: str) -> ProfilingState | None:
        path = self.path_for(run_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable checkpoint %s", path, exc_info=True)
            return None

        if int(payload.get("version", 0)) != STATE_VERSION:
            logger.warning(
                "Ignoring checkpoint %s written by an incompatible version (%s != %s)",
                path,
                payload.get("version"),
                STATE_VERSION,
            )
            return None
        return ProfilingState.from_dict(payload)

    def save(self, state: ProfilingState) -> None:
        state.updated_at = time.time()
        path = self.path_for(state.run_id)
        handle, tmp_name = tempfile.mkstemp(dir=str(self._directory), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(state.to_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            # Never leave a stray temp file behind, whatever went wrong.
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def delete(self, run_id: str) -> None:
        with contextlib.suppress(OSError):
            self.path_for(run_id).unlink()

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self._directory.glob("*.json"))


def samples_from(memories: Sequence[int], durations: Iterable[float]) -> list[Sample]:
    """Zip a memory sequence and its durations into :class:`Sample` objects."""
    return [
        Sample(memory_mb=int(m), duration_ms=float(d))
        for m, d in zip(memories, durations, strict=False)
    ]
