"""One-call assembly of the whole stack from :class:`~itsbob.config.Settings`.

Every piece is injectable on its own; this is just the ordinary wiring, so the
common case is one line and the unusual case is still open.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from .character.actions import ActionRegistry, default_registry
from .character.decisions import DecisionPolicy, build_policy
from .character.energy import TokenCostModel
from .character.state import Character, Traits
from .config import Settings
from .engine.events import EventBus
from .engine.simulation import Simulation
from .engine.world import World
from .llm.base import Provider
from .llm.providers import EchoProvider, build_provider
from .llm.router import LLMRouter, Strategy, UsageTracker

__all__ = ["build_router", "build_character", "build_simulation"]


def build_router(
    settings: Settings | None = None,
    *,
    env: Mapping[str, str] | None = None,
    strategy: Strategy = "priority",
    tracker: UsageTracker | None = None,
    extra_providers: Sequence[Provider] = (),
) -> LLMRouter:
    """Router over every provider that has a key, with the offline one last.

    A run never fails for want of an API key: if nothing is configured and
    ``allow_offline`` is set, :class:`~itsbob.llm.providers.EchoProvider` stands
    in and the simulation still exercises every real code path.
    """
    settings = settings or Settings.from_env()
    providers: list[Provider] = [
        build_provider(config, env) for config in settings.configured_providers(env)
    ]
    providers.extend(extra_providers)

    if not providers:
        if not settings.allow_offline:
            raise RuntimeError(
                "no LLM providers configured — set OPENROUTER_API_KEY, GROQ_API_KEY "
                "or GOOGLE_API_KEY (or allow the offline provider)"
            )
        # Only as a stand-in for *nothing*, never as a fallback behind a real
        # provider: a configured provider that fails must surface the failure,
        # not be papered over with a deterministic fake answer.
        providers.append(EchoProvider())

    return LLMRouter(
        providers,
        strategy=strategy,
        tracker=tracker,
        max_attempts=settings.max_attempts,
    )


def build_character(
    name: str = "Bob",
    settings: Settings | None = None,
    *,
    router: LLMRouter | None = None,
    traits: Traits | None = None,
    **kwargs: object,
) -> Character:
    settings = settings or Settings.from_env()
    return Character.create(
        name,
        traits=traits,
        energy_settings=settings.energy,
        memory_settings=settings.memory,
        router=router,
        **kwargs,
    )


def build_simulation(
    settings: Settings | None = None,
    *,
    name: str = "Bob",
    policy: DecisionPolicy | str = "hybrid",
    router: LLMRouter | None = None,
    world: World | None = None,
    registry: ActionRegistry | None = None,
    bus: EventBus | None = None,
    env: Mapping[str, str] | None = None,
    seed: int | None = None,
) -> Simulation:
    """Fully wired simulation, ready to :meth:`~itsbob.engine.Simulation.run`."""
    settings = settings or Settings.from_env()
    router = router or build_router(settings, env=env)
    cost_model = TokenCostModel.from_settings(settings.energy)

    if isinstance(policy, str):
        policy = build_policy(
            policy,
            cost_model=cost_model,
            deliberation_cost=settings.energy.deliberation_cost,
        )

    seed = settings.seed if seed is None else seed
    character = build_character(name, settings, router=router)
    return Simulation(
        character,
        world=world or World(),
        registry=registry or default_registry(),
        policy=policy,
        router=router,
        cost_model=cost_model,
        bus=bus,
        rng=random.Random(seed),
    )
