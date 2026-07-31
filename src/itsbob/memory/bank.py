"""The memory bank: short-term and long-term wired together.

Everything the character remembers goes in here, and everything it recalls comes
out of here. The interesting part is the seam between the two stores —
consolidation — which is where a bounded working set turns into a selective
permanent record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from ..config import MemorySettings
from .base import MemoryKind, MemoryRecord, render_records
from .long_term import LongTermMemory
from .short_term import ShortTermMemory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..llm.router import LLMRouter

__all__ = ["MemoryBank", "ConsolidationReport"]


@dataclass
class ConsolidationReport:
    tick: int
    promoted: list[MemoryRecord] = field(default_factory=list)
    discarded: list[MemoryRecord] = field(default_factory=list)
    reflections: list[MemoryRecord] = field(default_factory=list)

    @property
    def touched(self) -> int:
        return len(self.promoted) + len(self.discarded) + len(self.reflections)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"tick {self.tick}: promoted {len(self.promoted)}, "
            f"discarded {len(self.discarded)}, reflected {len(self.reflections)}"
        )


class MemoryBank:
    """Facade over :class:`ShortTermMemory` + :class:`LongTermMemory`."""

    def __init__(
        self,
        settings: MemorySettings | None = None,
        *,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        router: "LLMRouter | None" = None,
    ) -> None:
        self.settings = settings or MemorySettings()
        self.short_term = short_term or ShortTermMemory(
            capacity=self.settings.short_term_capacity,
            decay_rate=self.settings.short_term_decay,
            salience_floor=self.settings.salience_floor,
        )
        self.long_term = long_term or LongTermMemory(self.settings.database)
        self.router = router
        self._last_reflection_tick = 0

    # -- writing -----------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        importance: float = 0.5,
        tick: int = 0,
        tags: Sequence[str] = (),
        **metadata: object,
    ) -> MemoryRecord:
        """Record something. Anything pushed out of the working set is consolidated."""
        record, evicted = self.short_term.remember(
            content,
            kind=kind,
            importance=importance,
            tick=tick,
            tags=tags,
            **metadata,
        )
        if evicted:
            self._consolidate_records(evicted, tick)
        return record

    def memorize(self, record: MemoryRecord) -> MemoryRecord:
        """Write straight to long-term, bypassing the working set."""
        return self.long_term.add(record)

    # -- reading -----------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        tick: int | None = None,
        include_long_term: bool = True,
    ) -> list[MemoryRecord]:
        """Best ``limit`` memories across both stores, working set first.

        Short-term wins ties on purpose: what just happened outranks a matching
        memory from fifty ticks ago.
        """
        limit = limit or self.settings.recall_limit
        results = self.short_term.recall(query, limit=limit, tick=tick)
        if include_long_term and len(results) < limit:
            seen = {r.id for r in results}
            for record in self.long_term.recall(
                query, limit=limit - len(results), tick=tick
            ):
                if record.id not in seen:
                    results.append(record)
        return results[:limit]

    def working_set(self, limit: int | None = None) -> list[MemoryRecord]:
        return self.short_term.recent(limit)

    def render_context(self, query: str = "", *, limit: int = 6, tick: int = 0) -> str:
        """Prompt-ready memory block."""
        records = (
            self.recall(query, limit=limit, tick=tick)
            if query
            else self.working_set(limit)
        )
        return render_records(records)

    # -- time --------------------------------------------------------------

    def tick(self, tick: int) -> ConsolidationReport:
        """Advance one tick: decay the working set, consolidate, maybe reflect."""
        faded = self.short_term.decay()
        report = self._consolidate_records(faded, tick)
        if self._should_reflect(tick):
            report.reflections.extend(self.reflect(tick))
            self._last_reflection_tick = tick
        return report

    def flush(self, tick: int) -> ConsolidationReport:
        """Consolidate the entire working set (end of run, or a night's sleep)."""
        return self._consolidate_records(self.short_term.drain(), tick)

    def _consolidate_records(
        self, records: Sequence[MemoryRecord], tick: int
    ) -> ConsolidationReport:
        report = ConsolidationReport(tick=tick)
        for record in records:
            if self._should_promote(record):
                self.long_term.add(record)
                report.promoted.append(record)
            else:
                report.discarded.append(record)
        return report

    def _should_promote(self, record: MemoryRecord) -> bool:
        """Keep what mattered, what was revisited, or what was concluded.

        Rehearsal counts: a memory recalled more than once has proven useful
        even if it looked mundane when written.
        """
        if record.importance >= self.settings.promotion_threshold:
            return True
        if record.access_count >= 2:
            return True
        return record.kind in (MemoryKind.REFLECTION, MemoryKind.FACT)

    def _should_reflect(self, tick: int) -> bool:
        interval = self.settings.reflection_interval
        if interval <= 0 or self.router is None:
            return False
        return tick - self._last_reflection_tick >= interval

    # -- reflection --------------------------------------------------------

    def reflect(self, tick: int, *, max_insights: int = 3) -> list[MemoryRecord]:
        """Ask the LLM to draw conclusions from recent memory.

        Reflections are stored as first-class memories with high importance, so
        they survive consolidation and shape later recalls — the mechanism by
        which the character develops a point of view instead of just a log.
        Returns ``[]`` if there is no router or the call fails; reflection is a
        luxury, never a hard dependency.
        """
        if self.router is None:
            return []
        source = self.short_term.recent(10) or self.long_term.all(limit=10)
        if len(source) < 3:
            return []

        from ..llm.base import LLMRequest, system, user

        prompt = (
            "Recent memories:\n"
            f"{render_records(source)}\n\n"
            f"Draw at most {max_insights} short, concrete insights about patterns, "
            "preferences, or causes. Do not restate a memory verbatim.\n"
            'Reply as JSON: {"insights": ["...", "..."]}'
        )
        try:
            payload, _ = self.router.complete_json(
                LLMRequest(
                    messages=[
                        system(
                            "You distill an agent's memories into durable insights. "
                            "Terse, specific, no preamble."
                        ),
                        user(prompt),
                    ],
                    temperature=0.4,
                    max_tokens=600,
                ),
                purpose="reflection",
                default={"insights": []},
            )
        except Exception:
            return []

        insights = payload.get("insights") or []
        if isinstance(insights, str):
            insights = [insights]

        stored: list[MemoryRecord] = []
        for text in list(insights)[:max_insights]:
            text = str(text).strip()
            if not text:
                continue
            record = MemoryRecord(
                content=text,
                kind=MemoryKind.REFLECTION,
                importance=0.8,
                tick=tick,
                tags=("reflection",),
            )
            self.long_term.add(record)
            stored.append(record)
        return stored

    # -- misc --------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {"short_term": len(self.short_term), "long_term": len(self.long_term)}

    def close(self) -> None:
        self.long_term.close()
