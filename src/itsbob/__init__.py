"""itsbob — a tick-based character simulation with a metered LLM habit.

A character with two memory banks, a finite pool of energy, and access to free
LLMs it must pay for out of that pool.

    from itsbob import build_simulation

    sim = build_simulation(seed=7)
    for report in sim.stream(20):
        print(report.line())
    sim.finish()

The pieces are independent: :mod:`itsbob.llm` is a usable failover router on its
own, and :mod:`itsbob.memory` a usable memory store on its own.
"""

from .character.actions import Action, ActionRegistry, ActionResult, default_registry
from .character.decisions import (
    Decision,
    HeuristicPolicy,
    HybridPolicy,
    LLMPolicy,
    build_policy,
)
from .character.energy import EnergyLedger, InsufficientEnergy, TokenCostModel
from .character.state import Character, Needs, Traits
from .config import EnergySettings, MemorySettings, ProviderConfig, Settings, load_dotenv
from .engine.events import Event, EventBus
from .engine.simulation import Simulation, TickReport
from .engine.world import World
from .factory import build_character, build_router, build_simulation
from .llm.base import LLMRequest, LLMResponse, Message, Usage
from .llm.router import LLMRouter, UsageTracker
from .memory.bank import MemoryBank
from .memory.base import MemoryKind, MemoryRecord
from .memory.long_term import LongTermMemory
from .memory.short_term import ShortTermMemory
from .router import ComplexityRouter, RouteResult, ScriptRegistry, Tier, build_complexity_router
from .router import default_registry as default_script_registry

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionRegistry",
    "ActionResult",
    "Character",
    "ComplexityRouter",
    "Decision",
    "EnergyLedger",
    "EnergySettings",
    "Event",
    "EventBus",
    "HeuristicPolicy",
    "HybridPolicy",
    "InsufficientEnergy",
    "LLMPolicy",
    "LLMRequest",
    "LLMResponse",
    "LLMRouter",
    "LongTermMemory",
    "MemoryBank",
    "MemoryKind",
    "MemoryRecord",
    "MemorySettings",
    "Message",
    "Needs",
    "ProviderConfig",
    "RouteResult",
    "ScriptRegistry",
    "Settings",
    "ShortTermMemory",
    "Simulation",
    "Tier",
    "TickReport",
    "TokenCostModel",
    "Traits",
    "Usage",
    "UsageTracker",
    "World",
    "__version__",
    "build_character",
    "build_complexity_router",
    "build_policy",
    "build_router",
    "build_simulation",
    "default_registry",
    "default_script_registry",
    "load_dotenv",
]
