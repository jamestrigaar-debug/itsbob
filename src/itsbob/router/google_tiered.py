"""An all-Google tier ladder: Gemini answers every cloud tier itself.

Instead of "Groq first, Google second, OpenRouter third" for every cloud
tier alike, this assembles two *separate* Google-only routers — a cheaper
Gemini model for Tier B, a stronger one for Tier A — and, only if you ask
for it, lets Groq/OpenRouter trail behind each as a low-tier backup rather
than being tried before Google.

    from itsbob.router import build_google_tiered_router
    router = build_google_tiered_router(Settings.from_env())

Requires ``GOOGLE_API_KEY``. Without it every route still degrades
gracefully the same way the rest of this package does — see
:func:`build_google_tiered_router`'s docstring.
"""

from __future__ import annotations

from dataclasses import replace

from ..config import ProviderConfig, Settings
from ..llm.catalog import GOOGLE
from ..llm.local import OllamaProvider, is_ollama_running
from ..llm.providers import EchoProvider, GoogleProvider, build_provider
from ..llm.router import LLMRouter
from .cache import SemanticCache
from .gatekeeper import Gatekeeper
from .pipeline import ComplexityRouter
from .scripts import ScriptRegistry, default_registry

__all__ = ["build_google_tiered_router", "google_config_for"]

#: itsbob's own tier ladder, in cheapest-to-strongest order. Override any of
#: these with the matching argument to :func:`build_google_tiered_router` if
#: your account has access to different Gemini model ids.
TIER_B_MODEL = "gemini-3.1-flash-lite"  # cheapest/fastest — Tier B's default pick
TIER_B_FALLBACK_MODEL = "gemini-3.5-flash-lite"  # tried next if Tier B's model is rate-limited/retired
TIER_A_MODEL = "gemini-3.6-flash"  # strongest available Flash-class model — Tier A's default pick


def google_config_for(default_model: str, *fallback_models: str) -> ProviderConfig:
    """A Google :class:`ProviderConfig` pinned to a specific model + fallbacks.

    Starts from the shared ``GOOGLE`` template (base URL, rate limits, env
    var) so this stays in sync with the rest of the catalog rather than
    duplicating it.
    """
    return replace(GOOGLE, default_model=default_model, fallback_models=tuple(fallback_models))


def build_google_tiered_router(
    settings: Settings | None = None,
    *,
    registry: ScriptRegistry | None = None,
    goal: str = "win the league",
    tier_b_model: str = TIER_B_MODEL,
    tier_a_model: str = TIER_A_MODEL,
    fallback_model: str = TIER_B_FALLBACK_MODEL,
    low_tier_backup: bool = True,
) -> ComplexityRouter:
    """A :class:`~itsbob.router.pipeline.ComplexityRouter` where Google alone
    answers Tier B and Tier A, at two different Gemini models.

    - **Tier B** tries ``tier_b_model`` (default ``gemini-3.1-flash-lite``),
      then ``fallback_model`` (default ``gemini-3.5-flash-lite``) if the first
      is rate-limited or retired.
    - **Tier A** tries ``tier_a_model`` (default ``gemini-3.6-flash``), then
      falls back to ``fallback_model``, then ``tier_b_model`` — a premium
      call degrades to a cheaper Gemini model before it degrades all the way
      to the local safe default.
    - If ``low_tier_backup=True`` (the default) and you have `GROQ_API_KEY`
      and/or `OPENROUTER_API_KEY` set, those providers are appended *behind*
      Google on both tiers — "low tier" in your sense: only reached if every
      Gemini model tried already failed, never tried first. Set it to
      `False` for a Google-only chain (no Groq/OpenRouter involved at all).

    No `GOOGLE_API_KEY`? Every Tier B/A call fails over exactly like the rest
    of this package: to the low-tier backups if configured, otherwise straight
    into Phase 2's timeout-escalation path (local model's safe pick →
    `MAINTAIN_FORMATION` → Tier S). Nothing raises for a missing key.
    """
    settings = settings or Settings.from_env()
    registry = registry or default_registry()

    def backups() -> list:
        if not low_tier_backup:
            return []
        providers = [
            build_provider(config)
            for config in settings.configured_providers()
            if config.name in ("groq", "openrouter")
        ]
        if settings.allow_offline:
            providers.append(EchoProvider())
        return providers

    tier_b_router = LLMRouter(
        [GoogleProvider(google_config_for(tier_b_model, fallback_model, tier_a_model)), *backups()],
        max_attempts=settings.max_attempts,
    )
    tier_a_router = LLMRouter(
        [GoogleProvider(google_config_for(tier_a_model, fallback_model, tier_b_model)), *backups()],
        max_attempts=settings.max_attempts,
    )

    local_provider = OllamaProvider() if is_ollama_running() else None
    gatekeeper = Gatekeeper(registry=registry, local_provider=local_provider)

    return ComplexityRouter(
        registry=registry,
        gatekeeper=gatekeeper,
        cloud_router=tier_b_router,
        premium_router=tier_a_router,
        cache=SemanticCache(),
        goal=goal,
    )
