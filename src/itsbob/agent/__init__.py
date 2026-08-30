"""The agent: memory, tools, and a tier ladder, in a loop.

    from itsbob.agent import build_agent

    bob = build_agent(workspace="~/bob")
    print(bob.chat("what do you remember about me?").final)

:func:`build_agent` is the one-line assembly; every part is injectable if you
want to replace it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import Settings
from ..llm.embeddings import default_embedder
from ..memory.long_term import LongTermMemory
from ..tools import Mode, Policy, Tool, Toolbox, build_toolbox
from .brain import TIER_MODELS, TierResult, TieredBrain, build_brain
from .context import Conversation, Step, Turn
from .loop import Agent, AgentEvent
from .persona import Persona
from .writer import ExtractedMemory, MemoryWriter

__all__ = [
    "Agent",
    "AgentEvent",
    "Conversation",
    "ExtractedMemory",
    "MemoryWriter",
    "Persona",
    "Policy",
    "Toolbox",
    "build_toolbox",
    "Step",
    "TIER_MODELS",
    "TierResult",
    "TieredBrain",
    "Turn",
    "build_agent",
    "build_brain",
    "default_home",
]


def default_home(env: Mapping[str, str] | None = None) -> Path:
    """Where memory, the workspace and the audit log live by default.

    One directory, so "back up Bob" and "start over" are both one command.
    """
    env = os.environ if env is None else env
    return Path(env.get("ITSBOB_HOME", "").strip() or Path.home() / ".itsbob").expanduser()


def build_agent(
    *,
    home: str | Path | None = None,
    workspace: str | Path | None = None,
    mode: Mode | str | None = None,
    confirm: Any = None,
    persona: Persona | None = None,
    settings: Settings | None = None,
    memory: Any = None,
    toolbox: Toolbox | None = None,
    brain: TieredBrain | None = None,
    extra_tools: Sequence[Tool] = (),
    max_steps: int = 8,
    embeddings: bool = True,
    env: Mapping[str, str] | None = None,
) -> Agent:
    """Assemble a ready agent, creating its home directory if needed.

    Nothing here raises for a missing key or a stopped Ollama: the tier ladder
    falls back to the offline provider, and memory falls back to lexical-only
    recall. A first run with no configuration at all still answers.
    """
    env = os.environ if env is None else env
    root = Path(home).expanduser() if home else default_home(env)
    root.mkdir(parents=True, exist_ok=True)

    if memory is None:
        memory = LongTermMemory(
            root / "memory.sqlite",
            embedder=default_embedder(env) if embeddings else None,
        )
    if brain is None:
        brain = build_brain(settings, env=env)
    if toolbox is None:
        toolbox = build_toolbox(
            memory=memory,
            workspace=workspace or env.get("ITSBOB_WORKSPACE", "").strip() or root / "workspace",
            mode=mode,
            confirm=confirm,
            audit_path=root / "audit.jsonl",
            extra_tools=extra_tools,
            env=env,
        )

    return Agent(
        brain=brain,
        toolbox=toolbox,
        memory=memory,
        persona=persona or Persona(),
        max_steps=max_steps,
    )
