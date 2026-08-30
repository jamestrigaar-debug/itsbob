"""itsbob — a memory-backed assistant with a tiered LLM router.

    from itsbob import build_agent

    bob = build_agent()
    print(bob.chat("what did I say about the deploy?").final)

Four subsystems, each usable on its own:

* :mod:`itsbob.memory` — hybrid lexical + semantic recall over SQLite.
* :mod:`itsbob.tools` — an allow-list registry and the policy that gates it.
* :mod:`itsbob.agent` — the loop, and the tier ladder it routes each step to.
* :mod:`itsbob.daemon` — scheduled tasks, and the gate on interrupting you.

:mod:`itsbob.character` and :mod:`itsbob.engine` are the original tick-based
simulation this project grew out of. They still run (``itsbob run``) and share
the LLM layer, but nothing in the assistant depends on them.
"""

from .agent import Agent, AgentEvent, Persona, TieredBrain, Turn, build_agent, build_brain
from .config import EnergySettings, MemorySettings, ProviderConfig, Settings, load_dotenv
from .daemon import Daemon, Task, TaskStore, build_daemon, parse_schedule
from .llm.base import LLMRequest, LLMResponse, Message, Usage
from .llm.embeddings import EmbeddingRouter, default_embedder
from .llm.router import LLMRouter, UsageTracker
from .memory.bank import MemoryBank
from .memory.base import MemoryKind, MemoryRecord
from .memory.long_term import LongTermMemory, RecallHit
from .memory.short_term import ShortTermMemory
from .router.gatekeeper import Gatekeeper
from .router.ingestion import Snapshot, compress
from .router.tiers import GateDecision, Tier
from .tools import Mode, Policy, Tool, ToolRegistry, ToolResult, Toolbox, build_toolbox

__version__ = "0.3.0"

__all__ = [
    "Agent",
    "AgentEvent",
    "Daemon",
    "EmbeddingRouter",
    "EnergySettings",
    "GateDecision",
    "Gatekeeper",
    "LLMRequest",
    "LLMResponse",
    "LLMRouter",
    "LongTermMemory",
    "MemoryBank",
    "MemoryKind",
    "MemoryRecord",
    "MemorySettings",
    "Message",
    "Mode",
    "Persona",
    "Policy",
    "ProviderConfig",
    "RecallHit",
    "Settings",
    "ShortTermMemory",
    "Snapshot",
    "Task",
    "TaskStore",
    "Tier",
    "TieredBrain",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "Toolbox",
    "Turn",
    "Usage",
    "UsageTracker",
    "__version__",
    "build_agent",
    "build_brain",
    "build_daemon",
    "build_toolbox",
    "compress",
    "default_embedder",
    "load_dotenv",
    "parse_schedule",
]


def __getattr__(name: str):
    """Lazily expose the character simulation.

    Importing it eagerly pulled the whole tick engine into every ``import
    itsbob``, including the daemon's, for a subsystem most callers never touch.
    """
    _simulation = {
        "Action": ("character.actions", "Action"),
        "ActionRegistry": ("character.actions", "ActionRegistry"),
        "Character": ("character.state", "Character"),
        "Decision": ("character.decisions", "Decision"),
        "EnergyLedger": ("character.energy", "EnergyLedger"),
        "HeuristicPolicy": ("character.decisions", "HeuristicPolicy"),
        "HybridPolicy": ("character.decisions", "HybridPolicy"),
        "LLMPolicy": ("character.decisions", "LLMPolicy"),
        "Needs": ("character.state", "Needs"),
        "Simulation": ("engine.simulation", "Simulation"),
        "TickReport": ("engine.simulation", "TickReport"),
        "Traits": ("character.state", "Traits"),
        "World": ("engine.world", "World"),
        "build_character": ("factory", "build_character"),
        "build_router": ("factory", "build_router"),
        "build_simulation": ("factory", "build_simulation"),
    }
    if name in _simulation:
        import importlib

        module, attribute = _simulation[name]
        return getattr(importlib.import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
