"""Short-term memory: a small, decaying working set.

Capacity is the point. A bounded buffer forces a decision every time something
new arrives — keep it, or let it fall out toward consolidation — which is what
makes long-term memory selective rather than a transcript.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from .base import (
    DEFAULT_WEIGHTS,
    MemoryKind,
    MemoryRecord,
    RetrievalWeights,
    keyword_relevance,
    rank,
    render_records,
)

__all__ = ["ShortTermMemory"]


@dataclass
class ShortTermMemory:
    """FIFO working memory with salience decay.

    Two ways out: pushed off the end by newer material, or faded below
    ``salience_floor``. Either way the record is *returned*, never dropped on the
    floor — the consolidator decides whether it earns a place in long-term.
    """

    capacity: int = 12
    decay_rate: float = 0.12
    salience_floor: float = 0.15
    weights: RetrievalWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)
    relevance_fn: Callable[[str, MemoryRecord], float] = keyword_relevance
    _records: deque[MemoryRecord] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self.capacity = max(1, int(self.capacity))

    # -- writing -----------------------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Insert ``record``; use :meth:`add_returning_evicted` if you need the spill."""
        self.add_returning_evicted(record)
        return record

    def add_returning_evicted(
        self, record: MemoryRecord
    ) -> tuple[MemoryRecord, list[MemoryRecord]]:
        self._records.append(record)
        evicted: list[MemoryRecord] = []
        while len(self._records) > self.capacity:
            evicted.append(self._records.popleft())
        return record, evicted

    def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        importance: float = 0.5,
        tick: int = 0,
        tags: Sequence[str] = (),
        **metadata: object,
    ) -> tuple[MemoryRecord, list[MemoryRecord]]:
        record = MemoryRecord(
            content=content,
            kind=kind,
            importance=importance,
            tick=tick,
            tags=tuple(tags),
            metadata=dict(metadata),
        )
        return self.add_returning_evicted(record)

    # -- time --------------------------------------------------------------

    def decay(self) -> list[MemoryRecord]:
        """Age everything one tick and return whatever faded out."""
        faded: list[MemoryRecord] = []
        survivors: deque[MemoryRecord] = deque()
        for record in self._records:
            record.decay(self.decay_rate)
            (survivors if record.salience >= self.salience_floor else faded).append(record)
        self._records = survivors
        return faded

    # -- reading -----------------------------------------------------------

    def recall(
        self, query: str, *, limit: int = 5, tick: int | None = None
    ) -> list[MemoryRecord]:
        now = tick if tick is not None else self.latest_tick
        results = rank(
            self._records,
            query,
            now_tick=now,
            limit=limit,
            weights=self.weights,
            relevance_fn=self.relevance_fn,
        )
        for record in results:
            record.touch(now)
        return results

    def recent(self, limit: int | None = None) -> list[MemoryRecord]:
        records = list(self._records)
        return records if limit is None else records[-limit:]

    def all(self) -> list[MemoryRecord]:
        return list(self._records)

    def drain(self) -> list[MemoryRecord]:
        """Empty the buffer, returning everything (used when a run ends)."""
        records = list(self._records)
        self._records.clear()
        return records

    @property
    def latest_tick(self) -> int:
        return max((r.tick for r in self._records), default=0)

    def render(self, limit: int | None = None) -> str:
        return render_records(self.recent(limit))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MemoryRecord]:
        return iter(self._records)
