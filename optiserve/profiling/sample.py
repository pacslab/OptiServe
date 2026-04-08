"""Value objects for profiling data: a single :class:`Sample` and an
:class:`Exploration` container of samples for one function/model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    """One profiling observation: a duration measured at a memory size."""

    memory_mb: int
    duration_ms: float


class Exploration:
    """An ordered-by-memory collection of samples for a single function/model.

    Exposes numpy views used by the curve-fitting code. ``costs`` is the
    proportional surrogate ``duration * memory`` (not dollars — real pricing
    lives in :mod:`optiserve.cost`).
    """

    def __init__(self, samples: list[Sample] | None = None):
        self._samples: list[Sample] = list(samples) if samples else []

    @property
    def memories(self) -> np.ndarray:
        return np.array([s.memory_mb for s in self._samples], dtype=np.int32)

    @property
    def durations(self) -> np.ndarray:
        return np.array([s.duration_ms for s in self._samples], dtype=np.float32)

    @property
    def costs(self) -> np.ndarray:
        return self.durations * self.memories

    def add_sample(self, sample: Sample | list[Sample]) -> None:
        if isinstance(sample, Sample):
            self._samples.append(sample)
        elif isinstance(sample, list):
            self._samples.extend(sample)
        else:
            raise ValueError(f"Invalid sample type: {type(sample)}")
        self._samples.sort(key=lambda s: s.memory_mb)

    def __len__(self) -> int:
        return len(self._samples)
