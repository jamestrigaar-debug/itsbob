"""Deciding what was worth remembering, after the fact.

The agent has a ``remember`` tool and is told to use it, but relying on that
alone loses most of what matters: a model focused on answering a question does
not reliably stop to note that the user mentioned, in passing, that they have
moved cities. So a second cheap pass reads the finished turn and extracts
durable facts.

Two rules do most of the work:

**Durable, not momentary.** "I'm on the 14:05 train" is true for an hour;
"I commute from Reading" is true for years. Only the second is worth a row.

**Deduplicate against what is already known.** Recall runs first and the
existing memories go into the extraction prompt, because the failure mode of
an automatic writer is fifty near-identical rows about the same preference,
each recalled with equal confidence.

This runs on the cheapest tier. It is a background chore, and paying premium
prices for it would undo the point of the ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..llm.base import LLMRequest, system, user
from ..memory.base import MemoryKind, MemoryRecord
from ..router.tiers import Tier

__all__ = ["MemoryWriter", "ExtractedMemory"]

_SYSTEM = (
    "You extract durable facts from a conversation for an assistant's long-term "
    "memory. Return only things that will still be true and still be useful "
    "weeks from now.\n\n"
    "WRITE: stable preferences, decisions and their reasons, people and "
    "relationships, recurring problems, where things live, standing constraints, "
    "commitments with dates.\n"
    "DO NOT WRITE: anything true only today, the assistant's own actions, "
    "restatements of the conversation, tool output, or anything already listed as "
    "known below.\n\n"
    "Each fact must be one self-contained sentence, understandable with no other "
    "context, written in the third person about the user.\n"
    'Reply as strict JSON: {"memories": [{"content": "...", "kind": "fact|decision|'
    'observation", "importance": 0.0-1.0, "tags": ["short", "labels"]}]}\n'
    "An empty list is the correct answer for most conversations. Prefer writing "
    "nothing to writing something marginal."
)


@dataclass
class ExtractedMemory:
    content: str
    kind: MemoryKind = MemoryKind.FACT
    importance: float = 0.6
    tags: tuple[str, ...] = ()

    def to_record(self) -> MemoryRecord:
        return MemoryRecord(
            content=self.content,
            kind=self.kind,
            importance=self.importance,
            tags=self.tags,
            metadata={"source": "auto-extracted"},
        )


@dataclass
class MemoryWriter:
    """Post-turn extraction into long-term memory."""

    brain: Any
    store: Any
    tier: Tier = Tier.C
    max_per_turn: int = 4
    #: Skip extraction entirely for turns this short — a greeting has nothing
    #: in it, and the call costs more than the miss.
    min_chars: int = 24
    #: Cosine/lexical near-duplicates above this are treated as already known.
    duplicate_threshold: float = 0.92
    enabled: bool = True
    errors: int = 0
    last_error: str | None = None

    def extract(self, *, message: str, answer: str, known: Sequence[Any] = ()) -> list[ExtractedMemory]:
        if not self.enabled or len(message.strip()) < self.min_chars:
            return []

        known_lines = "\n".join(
            f"- {getattr(getattr(k, 'record', k), 'content', str(k))}" for k in list(known)[:12]
        )
        prompt = (
            f"User said:\n{message.strip()[:3000]}\n\n"
            f"Assistant replied:\n{answer.strip()[:2000]}\n\n"
            f"Already known (do not repeat these):\n{known_lines or '- (nothing)'}"
        )
        try:
            payload, _ = self.brain.complete_json(
                self.tier,
                LLMRequest(
                    messages=[system(_SYSTEM), user(prompt)],
                    temperature=0.0,
                    max_tokens=700,
                ),
                purpose="memory.extract",
                default={"memories": []},
            )
        except Exception as exc:  # noqa: BLE001 - never let bookkeeping break a turn
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return []

        out: list[ExtractedMemory] = []
        for item in (payload.get("memories") or [])[: self.max_per_turn]:
            if isinstance(item, str):
                item = {"content": item}
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            kind = MemoryKind.coerce(item.get("kind"))
            out.append(
                ExtractedMemory(
                    content=content,
                    kind=kind,
                    importance=_clamp(item.get("importance", 0.6)),
                    tags=tuple(str(t).strip().lower() for t in (item.get("tags") or []) if str(t).strip()),
                )
            )
        return out

    def write(self, *, message: str, answer: str, known: Sequence[Any] = ()) -> list[MemoryRecord]:
        """Extract and store, skipping anything already in memory."""
        stored: list[MemoryRecord] = []
        for extracted in self.extract(message=message, answer=answer, known=known):
            if self._is_duplicate(extracted.content):
                continue
            record = extracted.to_record()
            self.store.add(record)
            stored.append(record)
        return stored

    def _is_duplicate(self, content: str) -> bool:
        """Near-duplicate check via the store's own recall.

        Reuses recall rather than a separate similarity pass so the definition
        of "the same memory" is identical to the definition of "would be
        retrieved instead of" — which is the property that actually matters.
        """
        try:
            hits = self.store.search(content, limit=1, touch=False)
        except Exception:  # noqa: BLE001
            return False
        if not hits:
            return False
        existing = hits[0].record.content.strip().lower()
        candidate = content.strip().lower()
        if existing == candidate:
            return True
        # A high vector score against the top hit means recall would surface the
        # old row for anything that would surface the new one.
        return hits[0].vector_score >= self.duplicate_threshold


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.6
