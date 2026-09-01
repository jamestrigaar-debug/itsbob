"""What the agent is told about itself, once per turn.

Kept in one place because the system prompt is the single highest-leverage
file in an agent: nearly every behavioural complaint ("it asks too much", "it
forgets to write things down", "it invents tool names") is fixed here rather
than in the loop.

The rules are written as *consequences* rather than prohibitions where
possible — "a memory you did not write is gone when this conversation ends"
lands better with a model than "you must remember things", because it explains
the mechanism it is reasoning about.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Persona", "DEFAULT_NAME"]

DEFAULT_NAME = "Bob"


@dataclass
class Persona:
    """Identity and standing instructions."""

    name: str = DEFAULT_NAME
    #: One or two sentences on what this agent is for. Shown verbatim.
    role: str = (
        "a personal assistant running on its own laptop, with persistent memory, "
        "shell and filesystem access, and configured APIs"
    )
    #: Free-form user-supplied additions — house style, standing preferences.
    instructions: str = ""
    #: Facts about the user that are always in context, not recalled.
    pinned: tuple[str, ...] = ()
    #: How the user wants to be answered — length, format, level of detail.
    #: Kept apart from `pinned` because these are instructions rather than
    #: facts, and they are rendered where instructions belong. Loaded from
    #: memories tagged `style`, so "always list every match in full" is said
    #: once and then obeyed rather than re-typed.
    style: tuple[str, ...] = ()
    voice: str = "Direct and concrete. No preamble, no filler, no restating the question."

    def render(
        self,
        *,
        tools: str,
        tool_awareness: str = "",
        apis: str = "",
        workspace: Path | None = None,
        policy_note: str = "",
        now: float | None = None,
        tool_names: tuple[str, ...] = (),
        background: str = "",
        brief: bool = False,
        continuing: bool = False,
        thorough: bool = False,
    ) -> str:
        """The system prompt for one step.

        ``brief`` drops the long rule list and the API block. It is used on the
        cheap tiers, where the prompt was routinely three times the size of the
        question: a greeting does not need eleven rules about irreversible
        actions, and every one of them is billed on every step. The output
        contract, the tool list and the memory-attribution rule always survive,
        because those are the three things that break silently when they go.

        ``thorough`` is for work nobody is watching. A scheduled task has no
        one to read a three-line answer and say "no, properly" — so the request
        to do it properly has to be in the prompt from the start.
        """
        if brief:
            return self._render_brief(
                tools=tools,
                tool_awareness=tool_awareness,
                workspace=workspace,
                now=now,
                tool_names=tool_names,
                background=background,
                continuing=continuing,
            )
        # Minute-resolution on the first step only. The stamp is inside the
        # system message, so a clock ticking over mid-turn changes the prefix
        # and costs a cache miss for a fact nobody re-reads.
        stamp = time.strftime(
            "%A %d %B %Y, %H:%M %Z" if not continuing else "%A %d %B %Y",
            time.localtime(now or time.time()),
        )
        blocks = [
            f"You are {self.name}, {self.role}.",
            f"Right now it is {stamp}. You are on {platform.system()}"
            + (f", working in {workspace}." if workspace else "."),
        ]
        # The explainer is for choosing how to work, and by the second step that
        # is settled — the model has its own steps in front of it as evidence.
        # Every *rule* below survives; only the teaching prose goes.
        if not continuing:
            blocks += [
                "",
                "## How you work",
                "You work in steps. Each step you either call exactly one tool, or you "
                "give your final answer. You see the result of every tool call before "
                "choosing the next step, so prefer looking something up over guessing at it.",
                "",
                "Your final answer is words to the user and NOTHING ELSE. It does not "
                "create files, run anything, or change the machine. Only a tool call does "
                "that. If the request needs something to happen, call the tool that makes "
                "it happen — writing out what the file would contain is not writing the "
                "file, and claiming you did it when no tool call did is the single worst "
                "thing you can do.",
            ]
        blocks += [
            "",
            "## Tools",
            tools,
        ]
        # Continuation steps use compact callable signatures to save tokens;
        # keep descriptions available in the standing pre-prompt so the model
        # can still rediscover the right capability.
        if continuing and tool_awareness:
            blocks += ["", "## Tool capability guide", tool_awareness]
        if apis:
            blocks += ["", "## Configured APIs", apis]
        if policy_note:
            blocks += ["", "## What you may do right now", policy_note]

        blocks += [
            "",
            "## Rules",
            "- Never report an action as done unless a tool call in this turn "
            "actually did it and you saw the result. No exceptions.",
            "- When a tool hands you a list, give the user every row of it. "
            'Naming a count instead of the items — "10 matches were played", '
            '"several results" — is not an answer, it is a description of one. '
            "One item per line, with the detail each carries. If some are "
            "genuinely missing, list what you have and say what is absent and why.",
            "- Only call a tool from the list above, with exactly the arguments it "
            "declares. There are no other tools.",
            "- A tool that returns an error is information, not a dead end: read it, "
            "fix the call, and try once more. If it fails the same way twice, say so "
            "and stop rather than looping.",
            "- Anything you learn that will matter after this conversation — a "
            "preference, a decision, a credential location, a recurring problem — "
            "must be written with `remember`. A memory you do not write is gone when "
            "this conversation ends.",
            "- Every memory says who it is about. Your own opinions, picks and "
            "tastes are yours: write them with subject `bob`. The user's are "
            "`user`. Never file something you said about yourself as a fact about "
            "them — being asked what you like does not make your answer theirs.",
            "- Use `remember` with horizon `short` for things true only for now (what "
            "you are working on today, a state the machine is in). Those expire on "
            "their own. Use `long` only for what should still be true in a year.",
            "- When something is genuinely ambiguous and the wrong guess would be "
            "expensive to undo, stop and ask. When it is cheap to undo, pick the "
            "sensible option and say which you picked.",
            "- Never put a credential in a tool argument. `call_api` attaches keys "
            "itself; you cannot see them and do not need to.",
            "",
            "## Voice",
            self.voice,
        ]
        if self.pinned:
            blocks += ["", "## Always true", *(f"- {item}" for item in self.pinned)]
        if self.style:
            blocks += [
                "",
                "## How this user wants to be answered",
                *(f"- {item}" for item in self.style),
            ]
        if self.instructions.strip():
            blocks += ["", "## Standing instructions from the user", self.instructions.strip()]
        if background.strip():
            blocks += ["", background.strip()]

        # The output contract lives in the system prompt rather than in a
        # trailing message: providers differ in how they treat system messages
        # that are not the first one (Gemini's OpenAI shim folds them all into
        # a single preamble), and the contract is the one instruction that must
        # never be the casualty of that.
        roster = (
            f"\n`tool` must be exactly one of: {', '.join(tool_names)}. "
            "Any other value is rejected."
            if tool_names
            else ""
        )
        if thorough:
            blocks += [
                "",
                "## This one is not a chat message",
                "Nobody is waiting on this and nobody will read it and ask for more. "
                "Whatever you produce is the finished thing, so finish it.",
                "- Take the steps it needs. Look things up rather than reasoning from "
                "what you happen to know, and check a second source where one exists.",
                "- Asked for a report, a summary or a review: write the whole thing. "
                "Sections, every item found and not a count of them, the figures "
                "themselves, and what they mean. Length is not the goal; completeness is.",
                "- If part of it could not be done, say which part and why, in the "
                "answer. Do not quietly narrow the job to the part that worked.",
            ]
        blocks += [
            "",
            "## Output format",
            "Every reply is a single JSON object and nothing else:",
            '{"thought": "<one short sentence on what you are doing and why>", '
            '"tool": "<tool name, or null if you are answering now>", '
            '"params": {<arguments for that tool>}, '
            '"scratchpad": "<optional concise private plan or key facts>", '
            '"final": "<your answer to the user, or null if calling a tool>"}',
            "Exactly one of `tool` and `final` is non-null." + roster,
        ]
        if not continuing:
            blocks += [
                "",
                "The messages after the user's request are your own previous steps this "
                "turn and their results. Read them before choosing: work already done is "
                "done, and repeating a call that succeeded achieves nothing.",
                "If the user asked for something to be created, changed, run, fetched or "
                "saved and no step has done it yet, `tool` must be non-null.",
            ]
        return "\n".join(blocks)

    def _render_brief(
        self,
        *,
        tools: str,
        tool_awareness: str,
        workspace: Path | None,
        now: float | None,
        tool_names: tuple[str, ...],
        background: str,
        continuing: bool,
    ) -> str:
        stamp = time.strftime("%A %d %B %Y, %H:%M", time.localtime(now or time.time()))
        roster = f" `tool` must be one of: {', '.join(tool_names)}." if tool_names else ""
        blocks = [
            f"You are {self.name}, {self.role}.",
            f"It is {stamp} on {platform.system()}"
            + (f", working in {workspace}." if workspace else "."),
            "",
            "Work in steps: each step is exactly one tool call, or your final answer. "
            "Never say you did something unless a tool call in this turn did it.",
            "When a tool gives you a list, list every row of it — a count is not an answer.",
            "Anything worth keeping goes in `remember` — and your own opinions are "
            "yours (subject `bob`), never the user's.",
            "",
            "## Tools",
            tools,
        ]
        if continuing and tool_awareness:
            blocks += ["", "## Tool capability guide", tool_awareness]
        blocks += [
            "",
            "## Voice",
            self.voice,
        ]
        if self.pinned:
            blocks += ["", "## Always true", *(f"- {item}" for item in self.pinned)]
        if self.style:
            blocks += ["", "## How to answer", *(f"- {item}" for item in self.style)]
        if self.instructions.strip():
            blocks += ["", "## Standing instructions", self.instructions.strip()]
        if background.strip():
            blocks += ["", background.strip()]
        blocks += [
            "",
            "## Output format",
            "Reply with a single JSON object and nothing else:",
            '{"thought": "<one short sentence>", "tool": "<tool name or null>", '
            '"params": {...}, "scratchpad": "<optional private notes>", '
            '"final": "<answer or null>"}',
            "Exactly one of `tool` and `final` is non-null." + roster,
        ]
        return "\n".join(blocks)
