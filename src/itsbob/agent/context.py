"""Assembling what the model sees: persona, memory, history, scratchpad.

Context is a budget, not a bucket. Everything here is bounded, and the bounds
are chosen so the *shape* of the prompt stays constant as a conversation grows
— a model behaves differently when the transcript is 90% of its window, and
"it got worse after a while" is the hardest class of bug to notice.

Order matters and is deliberate: stable content first (persona, pinned facts),
then recalled memory, then recent turns, then this turn's scratchpad last.
Models weight the end of a prompt most heavily, and the scratchpad is what the
next decision is actually about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..llm.base import Message, assistant, system, user
from ..memory.base import MemoryRecord
from .persona import Persona

__all__ = ["Turn", "Step", "Conversation", "build_messages"]


@dataclass
class Step:
    """One iteration inside a turn: a thought, then a tool call or an answer."""

    index: int
    thought: str = ""
    tool: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    ok: bool = True
    tier: str = ""
    model: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "tool": self.tool,
            "params": self.params,
            "observation": self.observation[:2000],
            "ok": self.ok,
            "tier": self.tier,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class Turn:
    """One user message and everything that happened because of it."""

    message: str
    final: str = ""
    steps: list[Step] = field(default_factory=list)
    tier: str = ""
    started_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    tokens: int = 0
    error: str | None = None
    remembered: list[str] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "final": self.final,
            "steps": [s.as_dict() for s in self.steps],
            "tier": self.tier,
            "tools_used": self.tools_used,
            "duration_ms": round(self.duration_ms, 1),
            "tokens": self.tokens,
            "error": self.error,
            "remembered": self.remembered,
        }


@dataclass
class Conversation:
    """Bounded rolling history. Older turns are summarized, not silently dropped."""

    turns: list[Turn] = field(default_factory=list)
    #: Full turns kept verbatim. Beyond this, only the one-line summary survives.
    window: int = 8
    max_chars_per_turn: int = 1200

    def add(self, turn: Turn) -> Turn:
        self.turns.append(turn)
        return turn

    def recent(self) -> list[Turn]:
        return self.turns[-self.window :]

    def older(self) -> list[Turn]:
        return self.turns[: -self.window] if len(self.turns) > self.window else []

    def summary_of_older(self) -> str:
        older = self.older()
        if not older:
            return ""
        lines = []
        for turn in older[-20:]:
            tools = f" (used {', '.join(turn.tools_used)})" if turn.tools_used else ""
            lines.append(f"- asked: {_clip(turn.message, 100)}{tools}")
        return "Earlier in this conversation:\n" + "\n".join(lines)

    def as_messages(self) -> list[Message]:
        messages: list[Message] = []
        for turn in self.recent():
            messages.append(user(_clip(turn.message, self.max_chars_per_turn)))
            if turn.final:
                messages.append(assistant(_clip(turn.final, self.max_chars_per_turn)))
        return messages

    def __len__(self) -> int:
        return len(self.turns)


def render_memories(records: Sequence[Any], *, limit: int = 6) -> str:
    """Recalled memory, as compact lines the model can cite back.

    Ids are included so ``forget`` and ``update_memory`` are usable without a
    second lookup — the agent can only correct a memory it can name.
    """
    if not records:
        return ""
    lines = []
    for item in records[:limit]:
        record: MemoryRecord = getattr(item, "record", item)
        age = _ago(record.created_at)
        lines.append(f"- [{record.kind.value}] {record.content}  (id={record.id[:8]}, {age})")
    return "What you remember that may be relevant:\n" + "\n".join(lines)


def build_messages(
    *,
    persona: Persona,
    tools: str,
    snapshot_text: str,
    conversation: Conversation,
    memories: Sequence[Any] = (),
    steps: Sequence[Step] = (),
    apis: str = "",
    workspace: Any = None,
    policy_note: str = "",
    memory_limit: int = 6,
    tool_names: Sequence[str] = (),
) -> list[Message]:
    """The full message list for one step of one turn.

    Two shapes here are load-bearing.

    **Everything static is one system message.** Providers disagree about
    system messages that are not the first — Gemini's OpenAI-compatible shim
    folds them into a single preamble — so splitting instructions across
    several of them makes behaviour depend on the vendor. One message, built
    once, removes the question.

    **This turn's steps are assistant/user turns, not a system block.** An
    earlier version rendered the scratchpad as trailing system text, and the
    model did not read it as its own history: it re-issued the same tool call
    every step, each time thinking it was starting fresh. Encoding an action as
    an assistant message and its result as the user's reply is the shape models
    are actually trained on, and it fixed the loop outright.
    """
    background = "\n\n".join(
        block
        for block in (
            conversation.summary_of_older(),
            render_memories(memories, limit=memory_limit),
        )
        if block
    )
    messages: list[Message] = [
        system(
            persona.render(
                tools=tools,
                apis=apis,
                workspace=workspace,
                policy_note=policy_note,
                tool_names=tuple(tool_names),
                background=background,
            )
        )
    ]

    messages.extend(conversation.as_messages())
    messages.append(user(snapshot_text))

    for step in steps:
        messages.append(assistant(_render_action(step)))
        messages.append(user(_render_observation(step)))
    return messages


def _render_action(step: Step) -> str:
    """One past step, in exactly the format the model is asked to produce."""
    import json as _json

    return _json.dumps(
        {
            "thought": step.thought,
            "tool": step.tool,
            "params": step.params,
            "final": None,
        },
        default=str,
    )


def _render_observation(step: Step) -> str:
    return (
        f"Result of {step.tool}:\n{_clip(step.observation, 3000)}\n\n"
        "Continue: next JSON object, or `final` if the request is now satisfied."
    )


def _args(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={_clip(str(v), 60)}" for k, v in params.items())


def _clip(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [+{len(text) - limit} chars]"


def _ago(created_at: float) -> str:
    seconds = max(0.0, time.time() - created_at)
    for limit, unit, size in ((60, "s", 1), (3600, "m", 60), (86400, "h", 3600), (86400 * 30, "d", 86400)):
        if seconds < limit:
            return f"{int(seconds // size)}{unit} ago"
    return f"{int(seconds // (86400 * 30))}mo ago"
