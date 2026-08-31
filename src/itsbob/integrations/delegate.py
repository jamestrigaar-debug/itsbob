"""Sending the hard question somewhere cheaper, and getting structure back.

The premium tier is where the money goes: a Tier S step costs about twelve
times a Tier C one, and the work that needs it — real reasoning over a lot of
context — is exactly the work that is longest. If that reasoning can be done
somewhere free and the *result* handled locally, the expensive tier stops being
the default answer to "this is hard".

That is what this is: a question goes out of house, a structured answer comes
back, and a cheap local model turns it into the shape the rest of the system
expects. The transport is deliberately not specified here — it might be a
browser driving a chat site, a free API tier, anything — because the fragile
part is never the transport, it is the handoff.

Three things make the handoff survivable.

**Ask for structure explicitly.** A free chat interface will happily answer in
prose with a preamble, an apology and a follow-up question. The envelope asks
for one fenced JSON block with named fields, which turns "parse an essay" into
"find the fence".

**Never trust that structure arrived.** :func:`unwrap` tries the fence, then a
bare object, then gives up and says so. Giving up is not a failure — it hands
the raw text to the local model to shape, which is the cheap step that was
always going to happen anyway.

**Fail all the way back.** Every layer has somewhere to fall: no fence falls to
the local formatter, a dead transport falls to the normal tier ladder, and a
delegation that returns nothing usable is reported as unavailable rather than
being allowed to look like an answer. The one outcome that must never happen is
a confident-sounding empty response, because that is indistinguishable from a
real one until someone acts on it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = ["Delegation", "Envelope", "Delegate", "wrap", "unwrap", "DEFAULT_FIELDS"]

#: What a delegated answer is asked to contain. Kept small: every field is one
#: more thing that can be missing, and a field nobody reads is a field the
#: far end spends effort on for nothing.
DEFAULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("answer", "the full answer, in plain prose or markdown"),
    ("key_points", "a list of the load-bearing points, one string each"),
    ("caveats", "a list of anything uncertain, assumed, or out of date"),
    ("confidence", "high | medium | low"),
)


@dataclass(frozen=True)
class Envelope:
    """The prompt framework wrapped around a delegated question."""

    fields: Sequence[tuple[str, str]] = DEFAULT_FIELDS
    #: Prepended context. Short on purpose: a free chat interface has its own
    #: context limits and no prompt caching, so every word is paid twice.
    preamble: str = (
        "Answer the question below thoroughly and concretely. Do not ask me "
        "anything back — I cannot reply, this is a one-shot request."
    )

    def render(self, question: str, *, context: str = "") -> str:
        schema = ",\n".join(f'  "{name}": <{hint}>' for name, hint in self.fields)
        blocks = [self.preamble]
        if context.strip():
            blocks += ["", "Context you have been given:", context.strip()[:4000]]
        blocks += [
            "",
            "QUESTION:",
            question.strip(),
            "",
            "Reply with your reasoning first if you want, then end your message "
            "with exactly one fenced JSON block in this shape and nothing after it:",
            "```json",
            "{",
            schema,
            "}",
            "```",
        ]
        return "\n".join(blocks)


#: A fenced block, with or without the language tag. Non-greedy so a reply with
#: several fences yields each in turn and the last complete one wins.
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
#: A bare object at the end of a reply, for a model that ignored the fence.
_BARE = re.compile(r"(\{[^{}]*\"answer\"\s*:.*\})", re.S)


def wrap(question: str, *, context: str = "", envelope: Envelope | None = None) -> str:
    return (envelope or Envelope()).render(question, context=context)


def unwrap(reply: str) -> dict[str, Any] | None:
    """Pull the structured block out of a reply, or ``None`` if there isn't one.

    ``None`` is a normal outcome, not an error: it means the far end answered in
    prose, and prose is what the local formatter is for.
    """
    text = reply or ""
    candidates = [match.group(1) for match in _FENCE.finditer(text)]
    bare = _BARE.search(text)
    if bare:
        candidates.append(bare.group(1))
    # Last first: a reply that reasons aloud and then emits the block ends with
    # the one that counts.
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed:
            return parsed
    return None


@dataclass
class Delegation:
    """One question sent out of house, and everything that came of it."""

    question: str
    answer: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    ok: bool = False
    error: str = ""
    latency_ms: float = 0.0
    #: True when the far end returned the requested block; False when the local
    #: model had to shape prose into it.
    structured_at_source: bool = False
    raw_chars: int = 0

    def render(self) -> str:
        """What the agent sees. Complete, and cheaper than the raw reply."""
        if not self.ok:
            return f"delegation failed ({self.source}): {self.error}"
        lines = [self.answer.strip()]
        points = self.structured.get("key_points") or []
        if isinstance(points, list) and points:
            lines += ["", "Key points:"] + [f"- {p}" for p in points[:12]]
        caveats = self.structured.get("caveats") or []
        if isinstance(caveats, list) and caveats:
            lines += ["", "Caveats:"] + [f"- {c}" for c in caveats[:6]]
        confidence = str(self.structured.get("confidence") or "").strip()
        trail = f"(via {self.source}"
        if confidence:
            trail += f", confidence {confidence}"
        if not self.structured_at_source:
            trail += ", shaped locally"
        lines += ["", trail + ")"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question[:200],
            "ok": self.ok,
            "source": self.source,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "structured_at_source": self.structured_at_source,
            "raw_chars": self.raw_chars,
            "confidence": self.structured.get("confidence"),
        }


#: Asked of the local model when the far end answered in prose. Deliberately a
#: reshaping job and not a rewriting one: the cheap model is not being asked to
#: improve the answer, only to put it in the box.
_SHAPE_SYSTEM = (
    "You are reshaping an answer someone else wrote into a fixed structure. Do "
    "not add to it, correct it, or shorten it — the wording is theirs and it "
    "stays. Strip only the chat wrapper: greetings, offers to help further, "
    "questions back, and any 'as an AI' framing.\n"
    'Reply as strict JSON: {"answer": "<their answer, essentially verbatim>", '
    '"key_points": ["..."], "caveats": ["..."], "confidence": "high|medium|low"}'
)

#: A reply this short is a refusal, a login wall or a capture that missed —
#: never a real answer to a question worth delegating.
MIN_USEFUL_CHARS = 40

#: Phrases that mean the transport worked and the far end still said nothing.
_NON_ANSWERS = (
    "i can't help with that",
    "i cannot help with that",
    "please log in",
    "sign in to continue",
    "verify you are human",
    "rate limit",
    "too many requests",
)


@dataclass
class Delegate:
    """Runs a question through a transport and guarantees a shaped result."""

    #: ``fn(prompt) -> reply text``. Raises or returns empty if it cannot.
    transport: Callable[[str], str]
    #: ``fn(system, user) -> dict``, the cheap local model. Optional: without
    #: one, an unstructured reply is still returned, just unsplit.
    formatter: Callable[[str, str], dict[str, Any]] | None = None
    name: str = "delegate"
    envelope: Envelope = field(default_factory=Envelope)
    calls: int = 0
    failures: int = 0
    shaped_locally: int = 0
    last_error: str = ""

    def ask(self, question: str, *, context: str = "") -> Delegation:
        started = time.perf_counter()
        result = Delegation(question=question, source=self.name)
        self.calls += 1

        try:
            reply = self.transport(wrap(question, context=context, envelope=self.envelope))
        except Exception as exc:  # noqa: BLE001 - every transport failure is one outcome
            self.failures += 1
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            self.last_error = result.error
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        reply = (reply or "").strip()
        result.raw_chars = len(reply)
        problem = self._unusable(reply)
        if problem:
            self.failures += 1
            result.error = problem
            self.last_error = problem
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        structured = unwrap(reply)
        if structured is not None:
            result.structured_at_source = True
        else:
            structured = self._shape(reply)

        answer = str(structured.get("answer") or "").strip() or reply
        result.structured = structured
        result.answer = answer
        result.ok = True
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def _unusable(self, reply: str) -> str:
        """Why this reply is not an answer, or empty string if it is one."""
        if not reply:
            return "the transport returned nothing"
        if len(reply) < MIN_USEFUL_CHARS:
            return f"the reply was {len(reply)} characters — too short to be an answer"
        lowered = reply.lower()
        hit = next((phrase for phrase in _NON_ANSWERS if phrase in lowered), "")
        if hit and len(reply) < 400:
            return f"the far end declined or asked for a login ({hit!r})"
        return ""

    def _shape(self, reply: str) -> dict[str, Any]:
        """Prose into the structure, on the cheap model. Never raises."""
        if self.formatter is None:
            return {"answer": reply}
        self.shaped_locally += 1
        try:
            shaped = self.formatter(_SHAPE_SYSTEM, reply[:12000])
        except Exception as exc:  # noqa: BLE001 - the prose is still an answer
            self.last_error = f"formatter: {type(exc).__name__}: {exc}"[:200]
            return {"answer": reply}
        if not isinstance(shaped, dict) or not str(shaped.get("answer") or "").strip():
            return {"answer": reply}
        return shaped

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "failures": self.failures,
            "shaped_locally": self.shaped_locally,
            "last_error": self.last_error,
        }
