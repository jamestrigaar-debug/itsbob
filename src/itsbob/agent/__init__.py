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
from ..router.tiers import Tier
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


def _condenser(brain: TieredBrain):
    """Turns a list of headlines into prose, on the cheapest thing available.

    Given to the briefing tool rather than imported by it, so the news module
    never reaches back into the model ladder — and so the report degrades to a
    plain list rather than failing when nothing can condense it.
    """

    def condense(headlines) -> str:
        from ..llm.base import LLMRequest, system, user

        lines = "\n".join(
            f"- {h.title} ({h.source}): {h.summary}" for h in list(headlines)[:20]
        )
        try:
            result = brain.complete(
                Tier.C,
                LLMRequest(
                    messages=[
                        system(
                            "Condense these headlines into a short briefing for one "
                            "person. Group by what is actually happening, not by "
                            "outlet. Lead with anything geopolitically significant or "
                            "large in scale. Three to six sentences, plain prose, no "
                            "preamble, no bullet points, no speculation beyond what "
                            "the headlines say. If nothing is significant, say so in "
                            "one sentence."
                        ),
                        user(lines),
                    ],
                    temperature=0.2,
                    max_tokens=500,
                    # Free when Ollama is up. A daily chore is exactly the work
                    # that should never reach a paid model if it does not have to.
                    metadata={"local_ok": True},
                ),
                purpose="briefing.condense",
            )
        except Exception:  # noqa: BLE001 - the raw list is a fine fallback
            return ""
        return result.text.strip()

    return condense


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
    max_steps: int = 10,
    hard_max_steps: int = 60,
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
            summarize=_condenser(brain),
            env=env,
        )

    return Agent(
        brain=brain,
        toolbox=toolbox,
        memory=memory,
        persona=persona or Persona(),
        # A checkpoint rather than a wall: a turn that is getting somewhere
        # extends itself up to `hard_max_steps`. See `Agent._extend`.
        max_steps=max_steps,
        hard_max_steps=hard_max_steps,
    )
