"""The itsbob Complexity-Based Hierarchical Router.

Classify First, Execute Cheapest, Fallback Gracefully — Tiers S/A/B/C/D over
:mod:`itsbob.llm`.

    from itsbob.router import build_complexity_router
    from itsbob.config import Settings

    router = build_complexity_router(Settings.from_env())
    result = router.route({"stamina": 15, "minute": 78})
    print(result.tier, result.actions)
"""

from __future__ import annotations

import os
from typing import Mapping

from ..config import Settings
from ..llm.local import OLLAMA, OllamaProvider, is_ollama_running
from .cache import SemanticCache
from .gatekeeper import Gatekeeper
from .google_tiered import build_google_tiered_router, google_config_for
from .ingestion import GameState, compress
from .pipeline import ComplexityRouter, RouteResult
from .scripts import Script, ScriptRegistry, ScriptResult, default_registry
from .tiers import GateDecision, Tier

__all__ = [
    "ComplexityRouter",
    "GameState",
    "GateDecision",
    "OLLAMA",
    "OllamaProvider",
    "RouteResult",
    "Script",
    "ScriptRegistry",
    "ScriptResult",
    "SemanticCache",
    "Gatekeeper",
    "Tier",
    "build_complexity_router",
    "build_google_tiered_router",
    "compress",
    "default_registry",
    "google_config_for",
    "is_ollama_running",
]


def build_complexity_router(
    settings: Settings | None = None,
    *,
    registry: ScriptRegistry | None = None,
    goal: str = "win the league",
    mode: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ComplexityRouter:
    """One-line assembly: local Gatekeeper + cloud router(s) + cache + scripts.

    ``mode`` picks which cloud wiring Tier B/A use:

    - ``"priority"`` (default) — every configured cloud provider (Groq,
      Google, OpenRouter) in :data:`Settings.providers` order, same provider
      pool for both tiers. This is :func:`itsbob.factory.build_router`.
    - ``"google-tiered"`` — Google-only, at two different Gemini models (a
      cheap one for Tier B, a stronger one for Tier A); see
      :func:`itsbob.router.google_tiered.build_google_tiered_router`. Set
      ``ITSBOB_ROUTER_MODE=google-tiered`` to make this the default without
      changing call sites, or pass ``mode="google-tiered"`` directly.

    Mirrors :func:`itsbob.factory.build_router` in spirit either way — never
    fails for want of a local model or API keys. No Ollama running -> the
    Gatekeeper's heuristic classifier stands in. No API keys -> Tier B/C
    escalate straight through to the local/safe-default path, same as
    Phase 2's timeout monitor.
    """
    env = os.environ if env is None else env
    mode = mode or env.get("ITSBOB_ROUTER_MODE", "priority").strip() or "priority"

    if mode == "google-tiered":
        return build_google_tiered_router(settings, registry=registry, goal=goal)
    if mode != "priority":
        raise ValueError(f"unknown router mode {mode!r} (want: priority, google-tiered)")

    from ..factory import build_router

    settings = settings or Settings.from_env()
    registry = registry or default_registry()

    local_provider: OllamaProvider | None = None
    if is_ollama_running():
        local_provider = OllamaProvider()

    cloud_router = build_router(settings)
    gatekeeper = Gatekeeper(registry=registry, local_provider=local_provider)

    return ComplexityRouter(
        registry=registry,
        gatekeeper=gatekeeper,
        cloud_router=cloud_router,
        cache=SemanticCache(),
        goal=goal,
    )
