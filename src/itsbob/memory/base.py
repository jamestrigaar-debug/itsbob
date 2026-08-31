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
    "Horizon",
    "SHORT_TTL_SECONDS",
    "VITAL_GRACE",
    "VITAL_IMPORTANCE",
    "short_ttl_for",
    "MemoryKind",
    "MemoryRecord",
    "Subject",
    "MemoryStore",
    "RetrievalWeights",
    "tokenize",
    "keyword_relevance",
    "score_record",
]


class Subject(str, Enum):
    """Who a memory is *about*.

    This exists because of a real and embarrassing failure: asked for its own
    favourite films, the assistant answered with five of them and the extractor
    wrote every one down as "the user's favourite film is ...". Recall then fed
    those back as facts about the person, who had never mentioned a single one.

    Attribution is not a nicety. A memory store that cannot say whose opinion it
    is holding will, given enough turns, replace the user with the assistant.
    """

    USER = "user"  # the person this assistant works for
    SELF = "bob"  # the assistant's own tastes, habits, state, conclusions
    WORLD = "world"  # neither: the machine, a project, a place, a fact

    @classmethod
    def coerce(cls, value: "str | Subject | None") -> "Subject":
        if isinstance(value, cls):
            return value
        text = str(value or "user").strip().lower()
        try:
            return cls(text)
        except ValueError:
            return _SUBJECT_ALIASES.get(text, cls.USER)

    @property
    def label(self) -> str:
        return {"user": "the user", "bob": "you (the assistant)", "world": "the world"}[
            self.value
        ]


#: First person from the model means the model, second person means the user —
#: which is exactly backwards from how the extraction prompt is phrased, so both
#: readings are mapped explicitly rather than guessed at.
_SUBJECT_ALIASES: dict[str, "Subject"] = {}


class Horizon(str, Enum):
    """How long a memory is meant to survive.

    Short-horizon rows are the working set: what is going on right now, what was
    just tried, what the current thread is about. They are pruned by count and by
    clock, so a week of them cannot silently become the corpus. Long-horizon rows
    are the ones worth having in a year.
    """

    SHORT = "short"
    LONG = "long"

    @classmethod
    def coerce(cls, value: "str | Horizon | None") -> "Horizon":
        """Unstated means short.

        This default is the whole policy. Writing everything down permanently
        is not a good memory, it is a transcript: after a few weeks recall is
        picking between forty near-identical rows, each surfacing with equal
        confidence, and the one you wanted is no likelier than the rest. So a
        memory starts in the working set and *earns* permanence — by being
        recalled again, or by being kept on purpose with an explicit
        ``remember``. Being *scored* important at the moment of writing is not
        one of the ways: see :func:`short_ttl_for`.
        """
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        if text in ("long", "long_term", "long-term", "permanent", "durable", "forever"):
            return cls.LONG
        return cls.SHORT


#: How long an ordinary short-horizon memory lives before the working set
#: drops it. Long enough to carry a conversation, short enough that a busy
#: afternoon does not become the corpus.
SHORT_TTL_SECONDS = 6 * 3600.0

#: Scored at least this highly at the moment of writing.
VITAL_IMPORTANCE = 0.85

#: ...and so given this much longer to prove it. Six hours becomes seven days.
VITAL_GRACE = 28.0


def short_ttl_for(importance: float) -> float:
    """How long a short-horizon memory gets before the working set drops it.

    Importance buys *time*, never permanence. That distinction is the whole
    design, and it exists because the two obvious rules are both wrong.

    Promoting on a high score alone means the writer decides permanence at the
    one moment it is least able to: everything just said looks like it matters,
    and a model asked to rate what it has written says so nearly every time.
    That is what fills the store with rows nobody reads.

    But expiring everything at six hours regardless is worse in a rarer and
    more costly way. "Allergic to penicillin" scored 0.95 and never asked about
    again that afternoon is gone by evening, and the one memory whose loss
    actually matters is the one thrown away.

    So a high score does not make a memory permanent — it keeps it alive long
    enough to be recalled, and being recalled is what earns permanence. If it
    is never useful in a week, it was not worth keeping.
    """
    return SHORT_TTL_SECONDS * (VITAL_GRACE if importance >= VITAL_IMPORTANCE else 1.0)


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


_SUBJECT_ALIASES.update(
    {
        "me": Subject.SELF,
        "myself": Subject.SELF,
        "self": Subject.SELF,
        "assistant": Subject.SELF,
        "bob": Subject.SELF,
        "agent": Subject.SELF,
        "i": Subject.SELF,
        "you": Subject.USER,
        "the user": Subject.USER,
        "owner": Subject.USER,
        "human": Subject.USER,
        "person": Subject.USER,
        "environment": Subject.WORLD,
        "machine": Subject.WORLD,
        "laptop": Subject.WORLD,
        "system": Subject.WORLD,
        "project": Subject.WORLD,
        "other": Subject.WORLD,
        "general": Subject.WORLD,
        "none": Subject.WORLD,
    }
)


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
    #: Who this is about. Defaults to the user because most memories are, but
    #: the default is exactly what the extractor must not be allowed to coast
    #: on — see :class:`Subject`.
    subject: Subject = Subject.USER
    #: Working set or corpus. Short-horizon rows are pruned by count and clock,
    #: and the default, because permanence is earned rather than assumed.
    horizon: Horizon = Horizon.SHORT
    #: Wall-clock expiry, or ``None`` to keep until explicitly forgotten.
    expires_at: float | None = None
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
        self.kind = MemoryKind.coerce(self.kind)
        self.subject = Subject.coerce(self.subject)
        self.horizon = Horizon.coerce(self.horizon)
        if self.expires_at is None and self.metadata.get("expires_at") is not None:
            # Older callers put the expiry in metadata. Promote it rather than
            # having two sources of truth that can disagree.
            try:
                self.expires_at = float(self.metadata["expires_at"])
            except (TypeError, ValueError):
                self.expires_at = None
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

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    def render(self) -> str:
        tags = f" [{', '.join(self.tags)}]" if self.tags else ""
        # The subject is rendered for everything that is not about the user,
        # because "about the user" is the reading a model defaults to and the
        # other two are the ones it gets wrong.
        about = "" if self.subject is Subject.USER else f", about {self.subject.value}"
        return f"(t{self.tick}, {self.kind.value}{about}{tags}) {self.content}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind.value,
            "subject": self.subject.value,
            "horizon": self.horizon.value,
            "expires_at": self.expires_at,
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
            kind=MemoryKind.coerce(data.get("kind", "observation")),
            subject=Subject.coerce(data.get("subject")),
            horizon=Horizon.coerce(data.get("horizon")),
            expires_at=(
                float(data["expires_at"]) if data.get("expires_at") is not None else None
            ),
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
