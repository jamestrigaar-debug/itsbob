"""Deciding what was worth remembering, after the fact — and *whose* it is.

The agent has a ``remember`` tool and is told to use it, but relying on that
alone loses most of what matters: a model focused on answering a question does
not reliably stop to note that the user mentioned, in passing, that they have
moved cities. So a second cheap pass reads the finished turn and extracts
durable facts.

Three rules do most of the work:

**Attribution before content.** Every extracted memory names its subject. This
is the rule that earns its place: asked for its own favourite films, the
assistant listed five, and the extractor wrote all five down as *the user's*
favourites. Recall then served them back as facts about a person who had never
mentioned any of them. The reply half of a turn is the assistant talking — its
opinions belong to ``bob``, not to ``user``, and the prompt below says so in the
one place a model will actually read it.

**Everything written here starts short, without exception.** "I'm on the 14:05
train" is true for an hour; "I commute from Reading" is true for years — but
this is the worst possible moment to tell them apart, because it is the moment
both were just said and both look like they matter. So nothing decided here is
permanent: a row earns that later, by being recalled again when it turns out to
be useful. An explicit ``remember`` call is different — that is somebody asking
for something to be kept, and it may write straight to long-term.

**Deduplicate against what is already known.** Recall runs first and the
existing memories go into the extraction prompt, because the failure mode of
an automatic writer is fifty near-identical rows about the same preference,
each recalled with equal confidence.

This runs on the cheapest thing available — the local model when Ollama is up,
so an ordinary chat turn costs nothing extra at all. It is a background chore,
and paying premium prices for it would undo the point of the ladder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from ..llm.base import LLMRequest, system, user
from ..memory.base import Horizon, MemoryKind, MemoryRecord, Subject, short_ttl_for
from ..router.tiers import Tier

__all__ = ["MemoryWriter", "ExtractedMemory"]

_SYSTEM = (
    "You extract memories from one exchange between a user and an assistant "
    "called Bob, for Bob's memory.\n\n"
    "## Whose memory is it\n"
    "This is the rule you get wrong most often, so read it twice. The exchange "
    "has two speakers, and every memory belongs to exactly one of them:\n"
    '- subject "user" — something the USER said about themselves: what they '
    "like, decided, own, plan, or are dealing with.\n"
    '- subject "bob" — something BOB said about HIMSELF: his own opinions, '
    "tastes, picks, habits, or conclusions. If Bob named his favourite films, "
    'those are Bob\'s favourite films with subject "bob". They are NOT the '
    "user's. Asking someone what they like does not make their answer yours.\n"
    '- subject "world" — about neither person: the machine, a project, a file, '
    "a place, a service, a fact.\n"
    "Never turn something Bob said about himself into a fact about the user. "
    "That is the single worst mistake you can make here.\n\n"
    "## Do not write\n"
    "Restatements of the conversation, tool output, anything already listed as "
    "known below, or anything that would still be obvious without writing it.\n\n"
    "Each memory is one self-contained sentence, understandable with no other "
    "context. Write about the user in the third person ('the user prefers ...'); "
    "write about Bob in the first person ('I liked ...'), so the subject is "
    "unmistakable when it is read back.\n"
    'Reply as strict JSON: {"memories": [{"content": "...", "subject": '
    '"user|bob|world", "kind": "fact|preference|decision|observation", '
    '"importance": 0.0-1.0, "tags": ["short", "labels"]}]}\n'
    "An empty list is the correct answer for most conversations. Prefer writing "
    "nothing to writing something marginal."
)


@dataclass
class ExtractedMemory:
    content: str
    kind: MemoryKind = MemoryKind.FACT
    subject: Subject = Subject.USER
    horizon: Horizon = Horizon.SHORT
    importance: float = 0.6
    tags: tuple[str, ...] = ()

    def to_record(self, *, short_ttl: float | None = None) -> MemoryRecord:
        return MemoryRecord(
            content=self.content,
            kind=self.kind,
            subject=self.subject,
            horizon=self.horizon,
            expires_at=(
                # Scaled by importance, so something that reads as vital gets
                # long enough to be recalled once and earn permanence properly,
                # rather than being granted it on its own say-so.
                time.time() + (short_ttl_for(self.importance) if short_ttl is None else short_ttl)
                if self.horizon is Horizon.SHORT
                else None
            ),
            importance=self.importance,
            tags=self.tags,
            metadata={"source": "auto-extracted"},
        )


@dataclass
class MemoryWriter:
    """Post-turn extraction into memory."""

    brain: Any
    store: Any
    tier: Tier = Tier.C
    max_per_turn: int = 4
    #: Skip extraction entirely for turns this short — a greeting has nothing
    #: in it, and the call costs more than the miss.
    min_chars: int = 24
    #: Cosine/lexical near-duplicates above this are treated as already known.
    duplicate_threshold: float = 0.92
    #: How long an extracted short-horizon memory lives before the working set
    #: drops it. The store also caps how many may exist at once.
    short_ttl_seconds: float = 6 * 3600.0
    enabled: bool = True
    errors: int = 0
    last_error: str | None = None

    def extract(self, *, message: str, answer: str, known: Sequence[Any] = ()) -> list[ExtractedMemory]:
        if not self.enabled or len(message.strip()) < self.min_chars:
            return []

        known_lines = "\n".join(
            f"- {_known_line(k)}" for k in list(known)[:12]
        )
        prompt = (
            f"USER said:\n{message.strip()[:3000]}\n\n"
            f"BOB replied:\n{answer.strip()[:2000]}\n\n"
            "Remember: anything in BOB's reply that is an opinion, a pick or a "
            'preference of his own is subject "bob", not "user".\n\n'
            f"Already known (do not repeat these):\n{known_lines or '- (nothing)'}"
        )
        try:
            payload, _ = self.brain.complete_json(
                self.tier,
                LLMRequest(
                    messages=[system(_SYSTEM), user(prompt)],
                    temperature=0.0,
                    max_tokens=700,
                    metadata={"local_ok": True},
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
            subject = Subject.coerce(item.get("subject"))
            out.append(
                ExtractedMemory(
                    content=content,
                    kind=MemoryKind.coerce(item.get("kind")),
                    subject=_reattribute(content, subject),
                    # Never long, whatever the model says. Extraction happens
                    # at the moment of writing, and that is precisely the moment
                    # at which everything looks like it might matter — which is
                    # why permanence is earned by later recall instead. An
                    # explicit `remember` call, which is the user asking for
                    # something to be kept, may still write straight to long.
                    horizon=Horizon.SHORT,
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
            record = extracted.to_record(short_ttl=self.short_ttl_seconds)
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


#: Phrasings that give the subject away regardless of what the model labelled it.
#: The model is right most of the time; these catch the case it is wrong in, which
#: is precisely the one that corrupts the store.
_SELF_MARKERS = (
    "i like", "i love", "i prefer", "i enjoy", "i think", "i believe", "i found",
    "i would", "i'd ", "i chose", "i picked", "my favourite", "my favorite",
    "my top", "i decided", "i tend to", "i am ", "i'm ",
)
_USER_MARKERS = (
    "the user", "they said", "their favourite", "their favorite", "he said",
    "she said", "they prefer", "they like", "they want", "they asked",
)


def _reattribute(content: str, labelled: Subject) -> Subject:
    """Correct an obviously-wrong subject label from the sentence itself.

    A model that writes "I liked Blade Runner" and labels it ``user`` has
    contradicted itself in one line. Trusting the label there is how the store
    ends up believing the assistant's taste is the user's, so the sentence wins.
    """
    text = f" {content.strip().lower()} "
    if labelled is not Subject.SELF and any(text.startswith(f" {m}") or f". {m}" in text for m in _SELF_MARKERS):
        return Subject.SELF
    if labelled is Subject.SELF and any(m in text for m in _USER_MARKERS):
        return Subject.USER
    return labelled


def _known_line(item: Any) -> str:
    record = getattr(item, "record", item)
    content = getattr(record, "content", str(item))
    subject = getattr(record, "subject", None)
    if subject is not None and subject is not Subject.USER:
        return f"[about {getattr(subject, 'value', subject)}] {content}"
    return str(content)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.6
