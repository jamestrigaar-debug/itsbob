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

from ..config import Settings
from ..llm.local import OLLAMA, OllamaProvider, is_ollama_running
from .cache import SemanticCache
from .gatekeeper import Gatekeeper
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
    "compress",
    "default_registry",
    "is_ollama_running",
]


def build_complexity_router(
    settings: Settings | None = None,
    *,
    registry: ScriptRegistry | None = None,
    goal: str = "win the league",
) -> ComplexityRouter:
    """One-line assembly: local Gatekeeper + cloud router(s) + cache + scripts.

    Mirrors :func:`itsbob.factory.build_router` — never fails for want of a
    local model or API keys. No Ollama running -> the Gatekeeper's heuristic
    classifier stands in. No API keys -> Tier B/C escalate straight through to
    the local/safe-default path, same as Phase 2's timeout monitor.
    """
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
