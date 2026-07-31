"""The router: pick a provider, survive its failures, account for what it cost.

Free tiers fail constantly and in boring ways — 429s, retired model ids, a
vendor having a bad afternoon. The router's whole job is to make that other
people's problem: try providers in order, walk each one's fallback models,
trip a breaker on repeat failures, and record every attempt so the energy
economy can bill for it.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

from .base import (
    AllProvidersFailed,
    BadRequest,
    LLMRequest,
    LLMResponse,
    Message,
    Provider,
    ProviderNotConfigured,
    ProviderUnavailable,
    RateLimited,
    Usage,
    system,
    user,
)

__all__ = [
    "UsageRecord",
    "UsageTracker",
    "RateLimiter",
    "CircuitBreaker",
    "LLMRouter",
    "Strategy",
    "extract_json",
]

Strategy = Literal["priority", "round_robin", "random", "least_used"]
Clock = Callable[[], float]


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    ok: bool
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    purpose: str = "unspecified"


class UsageTracker:
    """Append-only log of every provider attempt, successful or not."""

    def __init__(self, max_records: int = 2000) -> None:
        self.records: deque[UsageRecord] = deque(maxlen=max_records)

    def record(self, record: UsageRecord) -> UsageRecord:
        self.records.append(record)
        return record

    @property
    def calls(self) -> int:
        return len(self.records)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.records if r.ok)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.records if not r.ok)

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for record in self.records:
            total = total + record.usage
        return total

    def by_provider(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for record in self.records:
            entry = summary.setdefault(
                record.provider,
                {"calls": 0, "ok": 0, "failed": 0, "tokens": 0, "latency_ms": 0.0},
            )
            entry["calls"] += 1
            entry["ok" if record.ok else "failed"] += 1
            entry["tokens"] += record.usage.total_tokens
            entry["latency_ms"] += record.latency_ms
        for entry in summary.values():
            entry["avg_latency_ms"] = round(
                entry["latency_ms"] / entry["calls"] if entry["calls"] else 0.0, 1
            )
        return summary

    def summary(self) -> str:
        usage = self.total_usage
        return (
            f"{self.calls} attempts ({self.successes} ok / {self.failures} failed), "
            f"{usage.total_tokens} tokens"
        )


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window request limiter, one per provider.

    Enforced locally so we spend our free quota deliberately rather than
    discovering the ceiling via 429s.
    """

    def __init__(self, requests_per_minute: int, clock: Clock = time.monotonic) -> None:
        self.requests_per_minute = max(1, requests_per_minute)
        self._clock = clock
        self._hits: deque[float] = deque()

    def _trim(self, now: float) -> None:
        cutoff = now - 60.0
        while self._hits and self._hits[0] <= cutoff:
            self._hits.popleft()

    def allows(self) -> bool:
        now = self._clock()
        self._trim(now)
        return len(self._hits) < self.requests_per_minute

    def retry_after(self) -> float:
        now = self._clock()
        self._trim(now)
        if len(self._hits) < self.requests_per_minute:
            return 0.0
        return max(0.0, 60.0 - (now - self._hits[0]))

    def consume(self) -> None:
        self._hits.append(self._clock())


class CircuitBreaker:
    """Stop hammering a provider that just failed several times running."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown: float = 60.0,
        clock: Clock = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self.cooldown:
            # Half-open: let one request through to test the water.
            self._opened_at = None
            self._failures = self.failure_threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()

    def cooldown_remaining(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.cooldown - (self._clock() - self._opened_at))


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


@dataclass
class _Slot:
    provider: Provider
    limiter: RateLimiter
    breaker: CircuitBreaker
    uses: int = 0


class LLMRouter:
    """Failover front end over a list of providers."""

    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        strategy: Strategy = "priority",
        tracker: UsageTracker | None = None,
        max_attempts: int = 4,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Clock = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self._slots = [
            _Slot(
                provider=p,
                limiter=RateLimiter(p.config.requests_per_minute, clock=clock),
                breaker=CircuitBreaker(breaker_threshold, breaker_cooldown, clock=clock),
            )
            for p in providers
        ]
        self.strategy: Strategy = strategy
        self.tracker = tracker or UsageTracker()
        self.max_attempts = max(1, max_attempts)
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._cursor = 0

    # -- introspection -----------------------------------------------------

    @property
    def providers(self) -> tuple[Provider, ...]:
        return tuple(slot.provider for slot in self._slots)

    def provider_names(self) -> tuple[str, ...]:
        return tuple(slot.provider.name for slot in self._slots)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "provider": slot.provider.name,
                "models": list(slot.provider.models),
                "configured": slot.provider.is_configured(),
                "circuit_open": slot.breaker.is_open,
                "cooldown_s": round(slot.breaker.cooldown_remaining(), 1),
                "rpm": slot.limiter.requests_per_minute,
                "uses": slot.uses,
            }
            for slot in self._slots
        ]

    # -- selection ---------------------------------------------------------

    def _ordered_slots(self) -> list[_Slot]:
        slots = list(self._slots)
        if self.strategy == "round_robin":
            if slots:
                offset = self._cursor % len(slots)
                slots = slots[offset:] + slots[:offset]
                self._cursor += 1
        elif self.strategy == "random":
            self._rng.shuffle(slots)
        elif self.strategy == "least_used":
            slots.sort(key=lambda s: s.uses)
        return slots

    # -- the call ----------------------------------------------------------

    def complete(
        self,
        request: LLMRequest,
        *,
        purpose: str = "unspecified",
        providers: Iterable[str] | None = None,
    ) -> LLMResponse:
        """Run ``request`` against the first provider that answers.

        Raises :class:`AllProvidersFailed` once the attempt budget is spent.
        """
        allowed = set(providers) if providers is not None else None
        errors: dict[str, BaseException] = {}
        attempts = 0

        for slot in self._ordered_slots():
            if attempts >= self.max_attempts:
                break
            name = slot.provider.name
            if allowed is not None and name not in allowed:
                continue
            if not slot.provider.is_configured():
                errors.setdefault(
                    name, ProviderNotConfigured(f"{name}: no API key configured")
                )
                continue
            if slot.breaker.is_open:
                errors[name] = ProviderUnavailable(
                    f"{name}: circuit open for another "
                    f"{slot.breaker.cooldown_remaining():.0f}s"
                )
                continue
            if not slot.limiter.allows():
                errors[name] = RateLimited(
                    f"{name}: local rate budget spent",
                    retry_after=slot.limiter.retry_after(),
                )
                continue

            for model in slot.provider.candidate_models(request):
                if attempts >= self.max_attempts:
                    break
                attempts += 1
                slot.limiter.consume()
                slot.uses += 1
                started = self._clock()
                try:
                    response = slot.provider.complete(request, model=model)
                except Exception as exc:  # normalized below
                    latency = (self._clock() - started) * 1000
                    self.tracker.record(
                        UsageRecord(
                            provider=name,
                            model=model,
                            ok=False,
                            latency_ms=latency,
                            error=f"{type(exc).__name__}: {exc}"[:300],
                            purpose=purpose,
                        )
                    )
                    errors[name] = exc
                    if isinstance(exc, BadRequest):
                        # A bad model id says nothing about the provider's
                        # health — try its next model, don't blame the vendor.
                        continue
                    if (
                        isinstance(exc, RateLimited)
                        and slot.provider.config.rate_limit_scope == "model"
                    ):
                        # Per-model quota: the vendor is fine, this model is
                        # spent. Its siblings still have budget.
                        continue
                    # Everything else (account-wide 429, 5xx, timeout, auth) is
                    # provider-wide; another model on the same host fails alike.
                    slot.breaker.record_failure()
                    break

                slot.breaker.record_success()
                self.tracker.record(
                    UsageRecord(
                        provider=name,
                        model=response.model,
                        ok=True,
                        usage=response.usage,
                        latency_ms=response.latency_ms,
                        purpose=purpose,
                    )
                )
                return response

        raise AllProvidersFailed(errors)

    # -- conveniences ------------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        purpose: str = "chat",
        **kwargs: Any,
    ) -> LLMResponse:
        messages: list[Message] = []
        if system_prompt:
            messages.append(system(system_prompt))
        messages.append(user(prompt))
        return self.complete(LLMRequest(messages=messages, **kwargs), purpose=purpose)

    def complete_json(
        self,
        request: LLMRequest,
        *,
        purpose: str = "json",
        default: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Like :meth:`complete`, but parse the reply as a JSON object.

        Models ignore ``response_format`` more often than they admit, so the
        text is salvaged with :func:`extract_json` before giving up.
        """
        request.json_mode = True
        response = self.complete(request, purpose=purpose)
        parsed = extract_json(response.text)
        if parsed is None:
            if default is None:
                raise ValueError(
                    f"{response.provider}/{response.model} did not return JSON: "
                    f"{response.text[:200]!r}"
                )
            parsed = dict(default)
        return parsed, response


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from a chat reply."""
    if not text:
        return None

    candidates = [text.strip()]
    candidates.extend(match.strip() for match in _FENCE_RE.findall(text))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None
