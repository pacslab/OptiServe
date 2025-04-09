import numpy as np

from src.utils.sample import Sample
from typing import Optional, List, Union


class Exploration:
    """Class for storing and analyzing samples of memory and duration for a single function."""

    def __init__(self, samples: Optional[List] = None):
        if samples is None:
            samples = []
        self._samples = samples

    @property
    def memories(self):
        return np.array([sample.memory_mb for sample in self._samples], dtype=np.int32)

    @property
    def durations(self):
        return np.array(
            [sample.duration_ms for sample in self._samples], dtype=np.float32
        )

    @property
    def costs(self):
        return np.array(self.durations * self.memories)

    def add_sample(self, sample: Union[Sample, List]):
        if isinstance(sample, Sample):
            self._samples.append(sample)
        elif isinstance(sample, list):
            self._samples.extend(sample)
        else:
            raise ValueError(f"Invalid sample type: {type(sample)}")

        self._samples.sort(key=lambda sample: sample.memory_mb)

    def __len__(self):
        return len(self._samples)
