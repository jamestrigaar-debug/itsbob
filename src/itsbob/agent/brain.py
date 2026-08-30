"""The tier ladder: which model answers, and what happens when it can't.

The original router picked a tier once per request. An agent picks one *per
step*, because the steps differ enormously in difficulty: deciding to call
``read_file`` is trivial, deciding whether a half-finished migration is safe to
resume is not, and paying premium-model prices for the first to afford the
second is the entire point of the ladder.

::

    Tier D   no model at all — a registered routine fires
    Tier C   cheapest — chat, formatting, an obvious single tool call
    Tier B   standard — real multi-step work, most tool use
    Tier A   premium — ambiguity, judgement, anything hard to undo
    Tier S   nothing could answer; stop and ask

Escalation is up first, then down. A Tier B call whose providers are all
failing tries A before it tries C: the expensive model is more likely to
actually answer, and one premium call beats a wrong cheap answer to a question
that already proved hard. Only when nothing above works does it slide
downward, and a total failure is Tier S — a halt, not a guess.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config import ProviderConfig, Settings
from ..llm.base import AllProvidersFailed, LLMRequest, LLMResponse, Provider
from ..llm.catalog import GOOGLE, default_provider_configs
from ..llm.local import OllamaProvider, is_ollama_running
from ..llm.providers import EchoProvider, GoogleProvider, build_provider
from ..llm.router import LLMRouter, UsageTracker, extract_json
from ..router.tiers import Tier

__all__ = ["TIER_MODELS", "TierResult", "TieredBrain", "build_brain"]


#: Gemini model per tier, cheapest first within each. Every id verified live
#: against the models listing — see :mod:`itsbob.llm.catalog` on why these are
#: defaults rather than constants.
TIER_MODELS: dict[Tier, tuple[str, ...]] = {
    Tier.C: ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite"),
    Tier.B: ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"),
    Tier.A: ("gemini-pro-latest", "gemini-3.6-flash", "gemini-3.5-flash"),
}

#: Where a tier goes when every provider on it fails. Up before down.
_ESCALATION: dict[Tier, tuple[Tier, ...]] = {
    Tier.C: (Tier.B, Tier.A),
    Tier.B: (Tier.A, Tier.C),
    Tier.A: (Tier.B, Tier.C),
}

_TIER_ENV = {Tier.C: "ITSBOB_TIER_C_MODEL", Tier.B: "ITSBOB_TIER_B_MODEL", Tier.A: "ITSBOB_TIER_A_MODEL"}


@dataclass
class TierResult:
    """One completion, plus what it actually cost to get."""

    response: LLMResponse
    tier: Tier
    #: The tier originally asked for, when escalation moved it.
    requested_tier: Tier | None = None
    attempts: int = 1
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def escalated(self) -> bool:
        return self.requested_tier is not None and self.requested_tier is not self.tier

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "requested_tier": self.requested_tier.value if self.requested_tier else None,
            "escalated": self.escalated,
            "provider": self.response.provider,
            "model": self.response.model,
            "latency_ms": round(self.response.latency_ms, 1),
            "tokens": self.response.usage.total_tokens,
            "errors": self.errors,
        }


class TieredBrain:
    """One :class:`~itsbob.llm.router.LLMRouter` per tier, with escalation between them."""

    def __init__(
        self,
        routers: Mapping[Tier, LLMRouter],
        *,
        local: Provider | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        if not routers:
            raise ValueError("TieredBrain needs at least one tier")
        self.routers = dict(routers)
        #: Ollama, when it is running. Free and private, so Tier C prefers it.
        self.local = local
        self.tracker = tracker or UsageTracker()

    def router_for(self, tier: Tier) -> LLMRouter | None:
        return self.routers.get(tier)

    def complete(
        self, tier: Tier, request: LLMRequest, *, purpose: str = "agent", escalate: bool = True
    ) -> TierResult:
        """Answer at ``tier``, escalating if every provider there fails."""
        chain: list[Tier] = [tier]
        if escalate:
            chain.extend(t for t in _ESCALATION.get(tier, ()) if t in self.routers)

        errors: dict[str, str] = {}
        attempts = 0

        # Tier C prefers the local model when one is up: it is free, private,
        # and fast enough for the work Tier C is given.
        if tier is Tier.C and self.local is not None:
            attempts += 1
            try:
                response = self.local.complete_with_fallback(request)
                if response.text.strip():
                    return TierResult(response=response, tier=Tier.C, attempts=attempts)
                errors["ollama"] = "empty response"
            except Exception as exc:  # noqa: BLE001 - cloud tiers still to try
                errors["ollama"] = f"{type(exc).__name__}: {exc}"[:200]

        for candidate in chain:
            router = self.routers.get(candidate)
            if router is None:
                continue
            attempts += 1
            try:
                response = router.complete(request, purpose=f"{purpose}.{candidate.value.lower()}")
            except AllProvidersFailed as exc:
                errors[candidate.value] = str(exc)[:300]
                continue
            return TierResult(
                response=response,
                tier=candidate,
                requested_tier=tier,
                attempts=attempts,
                errors=errors,
            )

        raise AllProvidersFailed({k: RuntimeError(v) for k, v in errors.items()})

    def complete_json(
        self,
        tier: Tier,
        request: LLMRequest,
        *,
        purpose: str = "agent",
        default: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], TierResult]:
        """Complete and parse a JSON object, salvaging a fenced or wrapped reply.

        A model that answers but not in JSON is a *retryable* failure, not a
        hard one: the next tier up frequently gets the format right where the
        cheap one rambled, so an unparseable reply escalates rather than
        raising.
        """
        request.json_mode = True
        chain: list[Tier] = [tier, *(t for t in _ESCALATION.get(tier, ()) if t in self.routers)]
        errors: dict[str, str] = {}

        for index, candidate in enumerate(chain):
            try:
                result = self.complete(candidate, request, purpose=purpose, escalate=False)
            except AllProvidersFailed as exc:
                errors[candidate.value] = str(exc)[:300]
                continue
            parsed = extract_json(result.text)
            if parsed is not None:
                result.requested_tier = tier
                result.errors.update(errors)
                return parsed, result
            errors[candidate.value] = (
                f"{result.response.provider}/{result.response.model} returned no JSON: "
                f"{result.text[:120]!r}"
            )
            if index == len(chain) - 1 and default is not None:
                result.requested_tier = tier
                result.errors.update(errors)
                return dict(default), result

        if default is not None:
            raise AllProvidersFailed({k: RuntimeError(v) for k, v in errors.items()})
        raise AllProvidersFailed({k: RuntimeError(v) for k, v in errors.items()})

    def describe(self) -> dict[str, Any]:
        return {
            "local": None if self.local is None else {
                "provider": self.local.name,
                "models": list(self.local.models),
            },
            "tiers": {
                tier.value: {
                    "label": tier.label,
                    "providers": router.describe(),
                }
                for tier, router in sorted(self.routers.items(), key=lambda kv: kv[0].value)
            },
            "usage": self.tracker.by_provider(),
        }


def _google_for(tier: Tier, models: Sequence[str], env: Mapping[str, str]) -> ProviderConfig | None:
    if not GOOGLE.api_key(env):
        return None
    from dataclasses import replace

    override = env.get(_TIER_ENV[tier], "").strip()
    ordered = [override, *models] if override else list(models)
    # Deduplicate while preserving order, so pinning a model that is already in
    # the ladder promotes it instead of listing it twice.
    seen = list(dict.fromkeys(m for m in ordered if m))
    return replace(GOOGLE, default_model=seen[0], fallback_models=tuple(seen[1:]))


def build_brain(
    settings: Settings | None = None,
    *,
    env: Mapping[str, str] | None = None,
    use_local: bool | None = None,
    tracker: UsageTracker | None = None,
) -> TieredBrain:
    """Assemble the ladder from whatever is configured, never failing for want of a key.

    Google supplies the tier separation, since one key gives access to models
    at genuinely different price points. Groq and OpenRouter sit *behind*
    Google on every tier as backups — reached only once every Gemini model has
    failed — because they have no comparable cheap/premium split to map onto.
    With no keys at all, every tier is the offline EchoProvider, so the loop
    still runs end to end.
    """
    env = os.environ if env is None else env
    settings = settings or Settings.from_env()
    tracker = tracker or UsageTracker()

    backups = [
        build_provider(config, env)
        for config in default_provider_configs(env)
        if config.name in ("groq", "openrouter") and config.is_configured(env)
    ]

    routers: dict[Tier, LLMRouter] = {}
    for tier, models in TIER_MODELS.items():
        providers: list[Provider] = []
        google = _google_for(tier, models, env)
        if google is not None:
            providers.append(GoogleProvider(google, env=env))
        providers.extend(backups)
        if not providers and settings.allow_offline:
            # Only when there is nothing real. Putting the offline provider
            # *behind* a configured one turns every transient failure — a
            # rate limit, a retired model — into a confident-looking wrong
            # answer instead of an error the caller can act on. A real
            # provider failing must fail.
            providers.append(EchoProvider())
        if not providers:
            continue
        routers[tier] = LLMRouter(
            providers, tracker=tracker, max_attempts=settings.max_attempts
        )

    if not routers:
        raise RuntimeError(
            "no LLM providers configured — set GOOGLE_API_KEY, GROQ_API_KEY or "
            "OPENROUTER_API_KEY (or allow the offline provider)"
        )

    local = None
    if use_local is not False and is_ollama_running():
        local = OllamaProvider(env=env)

    return TieredBrain(routers, local=local, tracker=tracker)
