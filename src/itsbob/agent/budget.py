"""Spending limits, and deciding whether a turn is worth starting at all.

Two failsafes, aimed at two different ways an agent burns money.

:class:`SpendGuard` is the blunt one: a ceiling on tokens per turn and per day.
It cannot tell a productive turn from a wasteful one, and does not try to — it
exists so that a loop nobody is watching has a bounded cost. A turn that hits
its ceiling is not killed; it is told to stop and answer with what it has, which
is the same ending the step budget produces and therefore already handled
everywhere downstream.

:class:`FeasibilityCheck` is the sharp one. Before an expensive turn starts, one
cheap call — the local model when Ollama is up, so usually free — reads the
request against the actual tool list and asks whether the tools present can do
this at all. The failure it prevents is specific and was costing real money:
asked for something the machine simply cannot do (a service with no API key, a
file that does not exist, a capability nobody wired up), the agent would spend
its whole step budget discovering that a step at a time on the premium tier, and
then report the discovery. Finding out first costs one small call.

The check is deliberately biased toward saying yes. A false "no" refuses work
the agent could have done, which is worse than a wasted turn; so it only stops
on an explicit, confident refusal, and any parse failure, timeout or missing
model means the turn proceeds as normal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..llm.base import LLMRequest, system, user
from ..router.tiers import Tier

__all__ = ["SpendGuard", "FeasibilityCheck", "Verdict"]


@dataclass
class SpendGuard:
    """Token ceilings for one turn and for one day.

    The daily count lives in memory, not on disk: it is a safety net for a
    runaway loop inside one process, and pretending it survives a restart would
    be a stronger claim than the implementation supports.
    """

    max_tokens_per_turn: int = 120_000
    max_tokens_per_day: int = 2_000_000
    #: Tokens spent in the turn currently running.
    turn_tokens: int = 0
    day_tokens: int = 0
    _day: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    #: Set when a ceiling stopped something, for the status panel.
    last_stop: str | None = None

    def start_turn(self) -> None:
        self.turn_tokens = 0
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self.day_tokens = 0

    def add(self, tokens: int) -> None:
        tokens = max(0, int(tokens))
        self.turn_tokens += tokens
        self.day_tokens += tokens

    def exceeded(self) -> str | None:
        """The reason to stop, or ``None`` to carry on."""
        if self.max_tokens_per_turn and self.turn_tokens >= self.max_tokens_per_turn:
            reason = (
                f"this turn has spent {self.turn_tokens:,} tokens, over its "
                f"{self.max_tokens_per_turn:,} limit"
            )
            self.last_stop = reason
            return reason
        if self.max_tokens_per_day and self.day_tokens >= self.max_tokens_per_day:
            reason = (
                f"today has spent {self.day_tokens:,} tokens, over the daily "
                f"{self.max_tokens_per_day:,} limit"
            )
            self.last_stop = reason
            return reason
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_tokens": self.turn_tokens,
            "day_tokens": self.day_tokens,
            "max_per_turn": self.max_tokens_per_turn,
            "max_per_day": self.max_tokens_per_day,
            "day": self._day,
            "last_stop": self.last_stop,
        }


@dataclass
class Verdict:
    """What the feasibility check concluded."""

    feasible: bool = True
    reason: str = ""
    missing: tuple[str, ...] = ()
    #: True only when a model actually answered; a skipped or failed check is
    #: feasible-by-default and must not be reported as a judgement.
    checked: bool = False
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "missing": list(self.missing),
            "checked": self.checked,
            "latency_ms": round(self.latency_ms, 1),
        }

    def explain(self) -> str:
        """The answer given to the user when a turn is refused before it starts."""
        lines = [f"I can't do this one: {self.reason.rstrip('.')}."]
        if self.missing:
            lines.append("What's missing: " + "; ".join(self.missing) + ".")
        lines.append(
            "Tell me how you'd like me to work around it, or set the missing piece up "
            "and ask again."
        )
        return " ".join(lines)


_SYSTEM = (
    "You are a feasibility check that runs before an assistant starts work, to "
    "stop it spending model calls on something it cannot finish.\n\n"
    "You are given the request and the exact tools the assistant has. Decide "
    "whether those tools could plausibly complete it.\n\n"
    "Say NOT feasible only when you are confident: the request needs a "
    "capability, service, credential or piece of hardware that is plainly not in "
    "the tool list, and no combination of the listed tools substitutes for it.\n"
    "Say feasible for everything else — including anything that merely looks "
    "hard, needs several steps, or needs information the assistant will have to "
    "go and find. Being wrong the other way refuses work it could have done, "
    "which is worse than one wasted turn. When unsure, say feasible.\n\n"
    'Reply as strict JSON: {"feasible": true|false, "reason": "<one short '
    'sentence>", "missing": ["<what is absent>", ...]}'
)


@dataclass
class FeasibilityCheck:
    """One cheap call that decides whether an expensive turn should start."""

    brain: Any
    tier: Tier = Tier.C
    enabled: bool = True
    #: Only guard turns the gatekeeper sent to these tiers. Cheap turns cost
    #: less than the check that would screen them.
    guard_tiers: frozenset[Tier] = frozenset({Tier.A, Tier.S})
    #: Below this many characters a request is not worth screening.
    min_chars: int = 40
    checks: int = 0
    refusals: int = 0
    errors: int = 0
    last_error: str | None = None

    def should_check(self, message: str, tier: Tier) -> bool:
        return (
            self.enabled
            and tier in self.guard_tiers
            and len(message.strip()) >= self.min_chars
        )

    def check(self, *, message: str, tools: Sequence[str], apis: Sequence[str] = ()) -> Verdict:
        started = time.perf_counter()
        prompt = (
            f"Request:\n{message.strip()[:1500]}\n\n"
            f"Tools available: {', '.join(tools) or '(none)'}\n"
            f"Configured APIs: {', '.join(apis) or '(none)'}"
        )
        self.checks += 1
        try:
            payload, _ = self.brain.complete_json(
                self.tier,
                LLMRequest(
                    messages=[system(_SYSTEM), user(prompt)],
                    temperature=0.0,
                    max_tokens=300,
                    metadata={"local_ok": True},
                ),
                purpose="agent.feasibility",
                default={"feasible": True},
            )
        except Exception as exc:  # noqa: BLE001 - a broken check must never block work
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return Verdict(latency_ms=(time.perf_counter() - started) * 1000)

        feasible = payload.get("feasible")
        # Anything but an explicit false is a yes. A model that answered with a
        # string, a null, or nothing at all has not made the confident refusal
        # this is allowed to act on.
        refused = feasible is False or str(feasible).strip().lower() == "false"
        reason = str(payload.get("reason") or "").strip()
        if refused and not reason:
            refused = False  # a refusal with no stated reason is not actionable
        if refused:
            self.refusals += 1
        return Verdict(
            feasible=not refused,
            reason=reason,
            missing=tuple(
                str(m).strip() for m in (payload.get("missing") or []) if str(m).strip()
            )[:5],
            checked=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "checks": self.checks,
            "refusals": self.refusals,
            "errors": self.errors,
            "last_error": self.last_error,
        }
