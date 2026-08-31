"""Sending the expensive questions somewhere free, without letting it run wild.

Tier S costs roughly twelve times Tier C, and the questions that reach it are
the long ones — the ones where both the prompt and the answer are large. A chat
site in a browser will answer many of those for nothing. That is a real saving
and it is worth taking.

It is also the single easiest thing in this system to overdo, so most of this
file is the restraint rather than the routing.

**What may go.** Only a question that can be answered by thinking. A chat site
cannot read the disk, call an API, take a screenshot or change anything, so
handing it "check whether the build passed" is not a saving, it is a wrong
answer arriving slowly. The screen is
:func:`~itsbob.router.gatekeeper.needs_tools`, shared with the classifier so
the two cannot disagree about what counts as tool work.

**How often.** A cap per hour, and a length floor. Delegation takes thirty
seconds to two minutes; spending that on a question the paid tier answers in
three is a bad trade even though it is free, because the person is waiting. So
short questions stay on the ladder no matter how they were classified.

**When it stops.** Consecutive failures open a circuit breaker for a while. The
far end is somebody else's website: it can be down, changed, rate-limited, or
asking for a login, and the failure mode without a breaker is every hard
question waiting two minutes to fail before falling back to the tier that was
going to answer it anyway. Two failures in a row is enough to conclude the
thing is not working right now.

**What it never does** is change the answer's standard. A delegated reply that
comes back empty, refusing, or too thin is not used; the tier ladder runs and
the turn proceeds as if none of this existed. The failure mode is "you paid for
the answer", never "you got a worse one".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..router.gatekeeper import needs_tools
from ..router.tiers import Tier

__all__ = ["DelegatePolicy", "Handoff"]


@dataclass
class Handoff:
    """Whether this turn may go out of house, and why not when it may not."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.allowed


@dataclass
class DelegatePolicy:
    """Decides which turns go to the free reasoner, and how often."""

    #: An :class:`~itsbob.integrations.delegate.Delegate`, or None to disable.
    delegate: Any = None
    #: Tiers whose work is expensive enough to be worth two minutes of waiting.
    tiers: frozenset[Tier] = frozenset({Tier.S})
    #: Ceiling per rolling hour. Deliberately low: this is a free service being
    #: used by an automated client, and hammering it is how it stops being one.
    per_hour: int = 8
    #: Below this, the round trip costs more in waiting than it saves in money.
    min_chars: int = 120
    #: Consecutive failures before standing down.
    trip_after: int = 2
    #: ...and for how long.
    cooldown_seconds: float = 900.0

    calls: int = 0
    answers: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    saved_tiers: int = 0
    last_error: str | None = None
    _recent: list[float] = field(default_factory=list, repr=False)
    _tripped_until: float = 0.0

    # -- deciding ----------------------------------------------------------

    def consider(self, *, message: str, tier: Tier, now: float | None = None) -> Handoff:
        """Whether this turn should be handed out, with the reason either way."""
        now = time.time() if now is None else now
        if self.delegate is None:
            return Handoff(False, "delegation is off")
        if tier not in self.tiers:
            return Handoff(False, f"tier {tier.value} is cheap enough already")

        text = (message or "").strip()
        if len(text) < self.min_chars:
            return Handoff(False, f"too short ({len(text)} chars) to be worth the wait")

        hit = needs_tools(text.lower())
        if hit:
            # Not a judgement call about quality — a chat site structurally
            # cannot do this. Sending it anyway produces a confident answer
            # about a machine it has never seen.
            return Handoff(False, f"needs tools ({hit!r}), which a chat site does not have")

        if now < self._tripped_until:
            left = int(self._tripped_until - now)
            return Handoff(False, f"standing down for another {left}s after repeated failures")

        self._recent = [stamp for stamp in self._recent if now - stamp < 3600]
        if len(self._recent) >= self.per_hour:
            return Handoff(False, f"already used {len(self._recent)} times this hour")

        return Handoff(True, "free, and this question is answerable by thinking")

    # -- doing -------------------------------------------------------------

    def ask(self, question: str, *, context: str = "", now: float | None = None) -> Any:
        """Put the question out of house. ``None`` when it did not come back usable.

        None is an ordinary outcome and the caller treats it as one: run the
        tier ladder, as it would have anyway.
        """
        now = time.time() if now is None else now
        self.calls += 1
        self._recent.append(now)
        try:
            result = self.delegate.ask(question, context=context)
        except Exception as exc:  # noqa: BLE001 - the ladder is the fallback for everything
            self._record_failure(f"{type(exc).__name__}: {exc}"[:200], now)
            return None

        if not getattr(result, "ok", False) or not str(getattr(result, "answer", "")).strip():
            self._record_failure(getattr(result, "error", "") or "empty reply", now)
            return None

        self.answers += 1
        self.saved_tiers += 1
        self.consecutive_failures = 0
        return result

    def _record_failure(self, error: str, now: float) -> None:
        self.failures += 1
        self.consecutive_failures += 1
        self.last_error = error
        if self.consecutive_failures >= self.trip_after:
            self._tripped_until = now + self.cooldown_seconds

    # -- reporting ---------------------------------------------------------

    def describe(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        return {
            "on": self.delegate is not None,
            "calls": self.calls,
            "answers": self.answers,
            "failures": self.failures,
            "used_this_hour": len([s for s in self._recent if now - s < 3600]),
            "per_hour": self.per_hour,
            "standing_down_for": max(0, int(self._tripped_until - now)),
            "last_error": self.last_error,
        }

    @classmethod
    def from_env(cls, env: Any = None, *, formatter: Any = None) -> "DelegatePolicy":
        """Built from the environment, or disabled when the switch is off."""
        import os

        from ..scripts.deepseek import build_delegate, enabled

        env = os.environ if env is None else env
        if not enabled(env):
            return cls(delegate=None)
        try:
            delegate = build_delegate(env=env, formatter=formatter)
        except Exception:  # noqa: BLE001 - no browser, no delegation, no crash
            return cls(delegate=None)
        return cls(
            delegate=delegate,
            per_hour=_int(env.get("ITSBOB_DEEPSEEK_PER_HOUR"), cls.per_hour),
            min_chars=_int(env.get("ITSBOB_DEEPSEEK_MIN_CHARS"), cls.min_chars),
        )


def _int(value: Any, fallback: int) -> int:
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return fallback


def context_from(memories: Sequence[Any], style: Sequence[str] = ()) -> str:
    """The little context worth paying for twice.

    A free chat interface has no prompt caching, so everything sent is billed
    in latency on every call. Only two things earn their place: what the person
    has said about how they want answers, and anything recalled that the
    question depends on.
    """
    lines: list[str] = []
    if style:
        lines.append("How the answer should be written: " + "; ".join(str(s) for s in style[:3]))
    for memory in list(memories)[:4]:
        content = getattr(getattr(memory, "record", memory), "content", "")
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines)[:800]
