"""Memoization for the analytical application model.

The greedy optimizer and the brute-force ground-truth sweep both work by
evaluating the *same* analytical model thousands of times over a configuration
space. Each evaluation runs the full graph reduction (``get_simple_dag``) and
then enumerates every simple path (``get_avg_rt``) — and the reduction is
re-derived from scratch every time even when the memory/variant configuration
being scored is one the model has already seen.

Two properties make memoization safe here:

* the model's outputs are a **pure function** of the per-node ``rt``/``mem``
  vectors, the fixed topology, the delay model and the execution counts; and
* the cache key is the *exact float tuple*, so a hit returns the bit-identical
  value the recomputation would have produced. No tolerance, no rounding, no
  numerical drift — which is what lets a published result stay reproducible.

The cache is bounded and reports hit statistics, because an unbounded dict over
a brute-force sweep of 10^6 configurations is itself a production incident.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

__all__ = ["CacheStats", "EvaluationCache"]

#: Chosen so a full BPBC/BCPC sweep on the benchmark applications fits, while
#: the worst case stays a few tens of MB rather than unbounded.
DEFAULT_MAX_ENTRIES = 200_000


@dataclass
class CacheStats:
    """Hit/miss counters, exposed so a run can report its own effectiveness."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
        }


class EvaluationCache:
    """A bounded LRU map from an exact configuration key to a computed value.

    ``None`` is a legal cached value, so lookups use a sentinel rather than
    ``get(...) is None`` — an evaluation that legitimately returns ``None`` must
    not be recomputed forever.
    """

    _MISS = object()

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()
        self.stats = CacheStats()

    def lookup(self, key: Hashable) -> tuple[bool, Any]:
        """``(found, value)``, counting the hit or the miss.

        Returned as a pair rather than a sentinel default because ``None`` is a
        legal cached value — an evaluation that legitimately returns ``None``
        must not be recomputed on every call.
        """
        value = self._entries.get(key, self._MISS)
        if value is self._MISS:
            self.stats.misses += 1
            return False, None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return True, value

    def get(self, key: Hashable, default: Any = None) -> Any:
        found, value = self.lookup(key)
        return value if found else default

    def contains(self, key: Hashable) -> bool:
        """Membership test that does not disturb the hit/miss counters."""
        return key in self._entries

    def put(self, key: Hashable, value: Any) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    def reset_stats(self) -> None:
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"EvaluationCache(entries={len(self._entries)}/{self._max_entries}, "
            f"hit_rate={self.stats.hit_rate:.2%})"
        )


class NullEvaluationCache:
    """Always misses. Lets callers keep one code path when caching is off."""

    def __init__(self) -> None:
        self.stats = CacheStats()

    def lookup(self, key: Hashable) -> tuple[bool, Any]:
        return False, None

    def get(self, key: Hashable, default: Any = None) -> Any:
        return default

    def contains(self, key: Hashable) -> bool:
        return False

    def put(self, key: Hashable, value: Any) -> None:
        return None

    def clear(self) -> None:
        return None

    def reset_stats(self) -> None:
        self.stats = CacheStats()

    def __len__(self) -> int:
        return 0


def make_cache(enabled: bool, max_entries: int | None = None) -> Any:
    """Factory returning either a real cache or the always-miss stand-in."""
    if not enabled:
        return NullEvaluationCache()
    return EvaluationCache(max_entries or DEFAULT_MAX_ENTRIES)
