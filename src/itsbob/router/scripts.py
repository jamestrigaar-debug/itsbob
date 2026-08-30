"""Tier D and the Golden Rule: pre-defined, hardened macros only.

"Never let the local LLM generate the script code itself... the Cloud API
must only output pre-defined script names... The local deterministic engine
maps these names to hardened, pre-tested Python macros."

This registry is that map. Nothing upstream of it — not the Gatekeeper, not
a cloud response — ever executes free-form code; they can only *name* an
action already registered here, and an unknown name is a routing error, not
an execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .ingestion import GameState

__all__ = ["ScriptResult", "Script", "ScriptRegistry", "default_registry"]


@dataclass
class ScriptResult:
    ok: bool
    action: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


ScriptFn = Callable[[GameState, Mapping[str, Any]], ScriptResult]


@dataclass
class Script:
    name: str
    description: str
    #: Deterministic if/else trigger — no LLM involved in deciding whether
    #: this script *should* fire. Optional: the direct-script path can also
    #: be entered explicitly by name (from a cloud/local decision).
    trigger: Callable[[GameState], bool] | None
    run: ScriptFn


class ScriptRegistry:
    """The "hardened, pre-tested Python macros" a name maps to."""

    def __init__(self) -> None:
        self._scripts: dict[str, Script] = {}

    def register(
        self,
        name: str,
        run: ScriptFn,
        *,
        description: str = "",
        trigger: Callable[[GameState], bool] | None = None,
    ) -> None:
        self._scripts[name] = Script(name=name, description=description, trigger=trigger, run=run)

    def names(self) -> list[str]:
        return sorted(self._scripts)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": s.name, "description": s.description}
            for s in sorted(self._scripts.values(), key=lambda s: s.name)
        ]

    def has(self, name: str) -> bool:
        return name in self._scripts

    def first_triggered(self, state: GameState) -> str | None:
        """Deterministic if/else pass — Tier D's actual trigger check."""
        for name, script in self._scripts.items():
            if script.trigger is not None and script.trigger(state):
                return name
        return None

    def execute(self, name: str, state: GameState, params: Mapping[str, Any] | None = None) -> ScriptResult:
        """Run a pre-registered macro by name.

        Raises :class:`KeyError` on an unknown name rather than guessing —
        that is the Golden Rule enforced in code: a hallucinated action name
        is a hard stop, never a best-effort execution.
        """
        script = self._scripts[name]
        return script.run(state, params or {})


def default_registry() -> ScriptRegistry:
    """A small set of example macros so the pipeline is runnable out of the box.

    These stand in for real keyboard/mouse automation — swap ``run`` bodies
    for actual actuation without touching anything upstream (the Gatekeeper
    and cloud prompts only ever deal in these names).
    """
    registry = ScriptRegistry()

    registry.register(
        "SUB_LOW_STAMINA",
        lambda state, params: ScriptResult(
            ok=True,
            action="SUB_LOW_STAMINA",
            detail="substituted the player flagged as low-stamina",
        ),
        description="Trivial threshold check: stamina < 20% -> substitute.",
        trigger=lambda state: float(state.facts.get("stamina", 100)) < 20,
    )
    registry.register(
        "HIGH_PRESS",
        lambda state, params: ScriptResult(
            ok=True, action="HIGH_PRESS", detail="switched tactic to high press"
        ),
        description="Apply an aggressive high-press tactical macro.",
    )
    registry.register(
        "WING_PLAY",
        lambda state, params: ScriptResult(
            ok=True, action="WING_PLAY", detail="switched tactic to wide wing play"
        ),
        description="Widen play down the flanks.",
    )
    registry.register(
        "MAINTAIN_FORMATION",
        lambda state, params: ScriptResult(
            ok=True,
            action="MAINTAIN_FORMATION",
            detail="no change — held current formation",
        ),
        description="The safe no-op used by Tier-B timeout downgrades (Phase 2).",
    )
    return registry
