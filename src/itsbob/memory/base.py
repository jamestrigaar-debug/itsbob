"""Memory records and the scoring that decides what gets recalled."""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

__all__ = [
    "MemoryKind",
    "MemoryRecord",
    "MemoryStore",
    "RetrievalWeights",
    "tokenize",
    "keyword_relevance",
    "score_record",
]


class MemoryKind(str, Enum):
    OBSERVATION = "observation"  # something the world did
    ACTION = "action"  # something the agent did
    DECISION = "decision"  # why it did it
    REFLECTION = "reflection"  # a conclusion drawn from other memories
    FACT = "fact"  # knowledge about the world
    PREFERENCE = "preference"  # what the user likes, wants, or always does
    DIALOGUE = "dialogue"  # something said or heard

    @classmethod
    def coerce(cls, value: "str | MemoryKind | None") -> "MemoryKind":
        """Accept the word a model reaches for, not only the exact token.

        Models pick the *meaning* over the enum member: real logs show
        `kind="preference"` rejected twice in one conversation, each rejection
        costing a step and a model call before it settled on "fact". Adding
        preference as a real kind fixes the common case — an assistant that
        cannot distinguish "likes dark roast coffee" from "Paris is in France"
        is missing a category it needs — and the aliases below absorb the rest.
        """
        if isinstance(value, cls):
            return value
        text = str(value or "fact").strip().lower()
        try:
            return cls(text)
        except ValueError:
            pass
        return _KIND_ALIASES.get(text, cls.FACT)


#: Near-misses seen in practice, mapped to the nearest real kind. Unknown words
#: fall back to FACT rather than raising: losing the *category* of a memory is a
#: much smaller loss than losing the memory.
_KIND_ALIASES: dict[str, MemoryKind] = {
    "preferences": MemoryKind.PREFERENCE,
    "like": MemoryKind.PREFERENCE,
    "likes": MemoryKind.PREFERENCE,
    "opinion": MemoryKind.PREFERENCE,
    "taste": MemoryKind.PREFERENCE,
    "habit": MemoryKind.PREFERENCE,
    "event": MemoryKind.OBSERVATION,
    "observations": MemoryKind.OBSERVATION,
    "note": MemoryKind.FACT,
    "knowledge": MemoryKind.FACT,
    "info": MemoryKind.FACT,
    "information": MemoryKind.FACT,
    "facts": MemoryKind.FACT,
    "task": MemoryKind.ACTION,
    "activity": MemoryKind.ACTION,
    "conclusion": MemoryKind.REFLECTION,
    "insight": MemoryKind.REFLECTION,
    "conversation": MemoryKind.DIALOGUE,
    "message": MemoryKind.DIALOGUE,
    "choice": MemoryKind.DECISION,
}


@dataclass
class MemoryRecord:
    """One remembered thing.

    ``importance`` is assigned at write time and never decays — it is how much
    the memory *mattered*. ``salience`` is short-term attention and decays every
    tick; it governs eviction from working memory, not long-term retrieval.
    """

    content: str
    kind: MemoryKind = MemoryKind.OBSERVATION
    importance: float = 0.5
    tick: int = 0
    tags: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    last_access_tick: int = 0
    access_count: int = 0
    #: Left unset, salience starts as a function of importance — a memory that
    #: mattered when it was formed also holds attention longer.
    salience: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.importance = _clamp(self.importance)
        if self.salience is None:
            self.salience = 0.5 + 0.5 * self.importance
        self.salience = _clamp(self.salience)
        self.tags = tuple(dict.fromkeys(t.strip().lower() for t in self.tags if t.strip()))
        if not self.last_access_tick:
            self.last_access_tick = self.tick

    def touch(self, tick: int) -> "MemoryRecord":
        self.last_access_tick = max(self.last_access_tick, tick)
        self.access_count += 1
        self.salience = _clamp(self.salience + 0.1)
        return self

    def decay(self, rate: float) -> float:
        self.salience = _clamp(self.salience * (1.0 - rate))
        return self.salience

    def render(self) -> str:
        tags = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"(t{self.tick}, {self.kind.value}{tags}) {self.content}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind.value,
            "importance": self.importance,
            "tick": self.tick,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "last_access_tick": self.last_access_tick,
            "access_count": self.access_count,
            "salience": self.salience,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            content=data["content"],
            kind=MemoryKind(data.get("kind", "observation")),
            importance=float(data.get("importance", 0.5)),
            tick=int(data.get("tick", 0)),
            tags=tuple(data.get("tags") or ()),
            id=data.get("id") or uuid.uuid4().hex,
            created_at=float(data.get("created_at", time.time())),
            last_access_tick=int(data.get("last_access_tick", 0)),
            access_count=int(data.get("access_count", 0)),
            salience=(
                float(data["salience"]) if data.get("salience") is not None else None
            ),
            metadata=dict(data.get("metadata") or {}),
        )


@runtime_checkable
class MemoryStore(Protocol):
    """Minimal contract shared by short- and long-term stores."""

    def add(self, record: MemoryRecord) -> MemoryRecord: ...

    def recall(
        self, query: str, *, limit: int = 5, tick: int | None = None
    ) -> list[MemoryRecord]: ...

    def all(self) -> list[MemoryRecord]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class RetrievalWeights:
    """Relevance / importance / recency mix, à la generative-agent retrieval."""

    relevance: float = 1.0
    importance: float = 0.6
    recency: float = 0.4
    #: Per-tick recency decay; 0.97 means ~50% weight after 23 ticks.
    recency_decay: float = 0.97


DEFAULT_WEIGHTS = RetrievalWeights()

_WORD_RE = re.compile(r"[a-z0-9']+")

# Small and hand-picked: the corpus here is a few hundred short sentences, so a
# heavyweight stoplist removes more signal than noise.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by
    with without about into over under again is are was were be been being am do does
    did doing have has had having i me my we our you your he him his she her it its
    they them their what which who whom when where why how all any both each few more
    most other some such no nor not only own same so too very s t can will just don
    should now
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def keyword_relevance(query: str, record: MemoryRecord) -> float:
    """Overlap between query and record, in ``[0, 1]``.

    A bag-of-words proxy for embedding similarity. It is deliberately the
    dumbest thing that works: swap in a real embedder by passing a different
    ``relevance_fn`` to the stores.
    """
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(record.content)) | set(record.tags)
    if not d:
        return 0.0
    overlap = len(q & d)
    if not overlap:
        return 0.0
    # Length-normalized so a long memory can't win on surface area alone.
    return overlap / math.sqrt(len(q) * len(d))


def score_record(
    record: MemoryRecord,
    query: str,
    *,
    now_tick: int,
    weights: RetrievalWeights = DEFAULT_WEIGHTS,
    relevance_fn=keyword_relevance,
) -> float:
    relevance = relevance_fn(query, record) if query else 0.0
    age = max(0, now_tick - record.last_access_tick)
    recency = weights.recency_decay**age
    return (
        weights.relevance * relevance
        + weights.importance * record.importance
        + weights.recency * recency
    )


def rank(
    records: Iterable[MemoryRecord],
    query: str,
    *,
    now_tick: int,
    limit: int = 5,
    weights: RetrievalWeights = DEFAULT_WEIGHTS,
    relevance_fn=keyword_relevance,
) -> list[MemoryRecord]:
    scored = [
        (
            score_record(
                r, query, now_tick=now_tick, weights=weights, relevance_fn=relevance_fn
            ),
            r,
        )
        for r in records
    ]
    # Stable tiebreak on tick keeps seeded runs reproducible.
    scored.sort(key=lambda pair: (-pair[0], -pair[1].tick, pair[1].id))
    return [record for _, record in scored[:limit]]


def render_records(records: Sequence[MemoryRecord]) -> str:
    return "\n".join(f"- {r.render()}" for r in records) if records else "- (nothing)"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
