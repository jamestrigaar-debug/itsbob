"""The tick loop.

One tick is: the world moves, the character perceives it, memory is queried,
a policy decides, the action runs, the consequences are paid for and
remembered. Everything else in the framework is in service of that sequence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..character.actions import ActionContext, ActionRegistry, ActionResult, default_registry
from ..character.decisions import Decision, DecisionContext, DecisionPolicy, HeuristicPolicy
from ..character.energy import InsufficientEnergy, TokenCostModel
from ..character.state import Character
from ..llm.router import LLMRouter
from ..memory.base import MemoryKind
from .events import EventBus
from .world import World

__all__ = ["Simulation", "TickReport"]


@dataclass
class TickReport:
    """What happened in one tick. The unit of output for any front end."""

    tick: int
    decision: Decision
    result: ActionResult
    energy_before: float
    energy_after: float
    llm_calls: int = 0
    tokens: int = 0
    recalled: int = 0
    consolidated: int = 0
    events: list[str] = field(default_factory=list)

    @property
    def action_name(self) -> str:
        return self.decision.action.name

    @property
    def energy_spent(self) -> float:
        return self.energy_before - self.energy_after

    def line(self, width: int = 96) -> str:
        """One-line narration, trimmed to fit a terminal."""
        flag = "!" if not self.result.ok else " "
        text = " ".join((self.result.narrative or self.decision.rationale).split())
        if width and len(text) > width:
            text = text[: width - 1] + "…"
        return (
            f"t{self.tick:>3}{flag} [{self.decision.source:^13}] "
            f"{self.action_name:<15} "
            f"energy {self.energy_before:5.1f}→{self.energy_after:5.1f} "
            f"| {text}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "action": self.action_name,
            "source": self.decision.source,
            "rationale": self.decision.rationale,
            "confidence": self.decision.confidence,
            "ok": self.result.ok,
            "narrative": self.result.narrative,
            "energy_before": round(self.energy_before, 2),
            "energy_after": round(self.energy_after, 2),
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "data": self.result.data,
        }


class Simulation:
    """Drives a :class:`Character` through a :class:`World`."""

    def __init__(
        self,
        character: Character,
        *,
        world: World | None = None,
        registry: ActionRegistry | None = None,
        policy: DecisionPolicy | None = None,
        router: LLMRouter | None = None,
        cost_model: TokenCostModel | None = None,
        bus: EventBus | None = None,
        rng: random.Random | None = None,
        seed: int | None = None,
    ) -> None:
        self.character = character
        self.world = world or World()
        self.registry = registry or default_registry()
        self.policy = policy or HeuristicPolicy()
        self.router = router
        self.cost_model = cost_model or TokenCostModel()
        self.bus = bus or EventBus()
        self.rng = rng or random.Random(seed)
        self.reports: list[TickReport] = []

    # -- one tick ----------------------------------------------------------

    def step(self) -> TickReport:
        self.world.advance(self.rng)
        tick = self.world.tick
        character = self.character
        energy_before = character.energy.current

        action_ctx = ActionContext(
            character=character,
            world=self.world,
            tick=tick,
            rng=self.rng,
            router=self.router,
            cost_model=self.cost_model,
        )

        self._perceive(tick)
        recalled = self._recall(tick)
        options = self.registry.available(action_ctx)
        if not options:
            # Nothing is affordable — resting is always possible in principle,
            # so the character collapses into it rather than deadlocking.
            options = [self.registry.get("rest") or self.registry.all()[0]]

        calls_before = self._llm_calls()
        decision = self.policy.decide(
            DecisionContext(action_ctx=action_ctx, options=options, recalled=recalled)
        )
        self.bus.emit("decision", tick=tick, decision=decision)

        result = self._execute(decision, action_ctx)
        self._apply(decision, result, action_ctx)

        consolidation = character.memory.tick(tick)
        character.needs.tick()
        # Passive recovery is a trickle on purpose — if idling refilled the bar,
        # nothing would ever be a real trade-off. Resting is the way back up.
        rest_bonus = 2.5 if "recovery" in decision.action.tags else 1.0
        character.energy.regenerate(tick=tick, multiplier=rest_bonus * 0.25)

        report = TickReport(
            tick=tick,
            decision=decision,
            result=result,
            energy_before=energy_before,
            energy_after=character.energy.current,
            llm_calls=self._llm_calls() - calls_before,
            tokens=self._tokens_since(calls_before),
            recalled=len(recalled),
            consolidated=consolidation.touched,
        )
        self.reports.append(report)
        self.bus.emit("tick", tick=tick, report=report)
        return report

    # -- phases ------------------------------------------------------------

    def _perceive(self, tick: int) -> None:
        self.character.memory.remember(
            f"{self.world.describe()} at {self.character.location}.",
            kind=MemoryKind.OBSERVATION,
            importance=0.25,
            tick=tick,
            tags=("world", self.world.phase),
        )

    def _recall(self, tick: int) -> list:
        pressing, _ = self.character.needs.most_pressing()
        query = f"{pressing} {self.character.goal} {self.world.phase}"
        return self.character.memory.recall(query, tick=tick)

    def _execute(self, decision: Decision, ctx: ActionContext) -> ActionResult:
        action = decision.action
        try:
            self.character.energy.spend(action.energy_cost, action.name, tick=ctx.tick)
        except InsufficientEnergy as exc:
            return ActionResult(
                ok=False,
                narrative=f"{self.character.name} tries to {action.name} and cannot: {exc}",
                needs_delta={"rest": 0.05},
                mood_delta=-0.05,
            )
        try:
            return action.run(ctx)
        except Exception as exc:  # an action must never kill the run
            self.bus.emit("action_error", tick=ctx.tick, action=action.name, error=str(exc))
            return ActionResult(
                ok=False,
                narrative=f"{action.name} went wrong: {type(exc).__name__}: {exc}",
                mood_delta=-0.05,
            )

    def _apply(self, decision: Decision, result: ActionResult, ctx: ActionContext) -> None:
        character = self.character
        tick = ctx.tick

        if result.extra_energy:
            # try_spend: the cost was already incurred upstream (tokens burned),
            # so a shortfall drains the ledger rather than raising.
            if not character.energy.try_spend(
                result.extra_energy, f"{decision.action.name}_extra", tick=tick
            ):
                character.energy.spend(
                    character.energy.current, f"{decision.action.name}_extra", tick=tick
                )
        if result.energy_gained:
            character.energy.gain(result.energy_gained, decision.action.name, tick=tick)

        for need, delta in result.needs_delta.items():
            if delta < 0:
                character.needs.satisfy(need, -delta)
            else:
                character.needs.aggravate(need, delta)

        if result.mood_delta:
            character.adjust_mood(result.mood_delta)

        character.memory.remember(
            f"Chose to {decision.action.name}. {decision.rationale}",
            kind=MemoryKind.DECISION,
            importance=0.4 + 0.2 * decision.confidence,
            tick=tick,
            tags=("decision", decision.source),
            confidence=decision.confidence,
        )
        for content, kind, importance in result.memories:
            character.memory.remember(
                content, kind=kind, importance=importance, tick=tick, tags=(decision.action.name,)
            )

        self.bus.emit("action", tick=tick, action=decision.action.name, result=result)

    # -- accounting --------------------------------------------------------

    def _llm_calls(self) -> int:
        return self.router.tracker.calls if self.router else 0

    def _tokens_since(self, calls_before: int) -> int:
        if not self.router:
            return 0
        records = list(self.router.tracker.records)[calls_before:]
        return sum(r.usage.total_tokens for r in records)

    # -- running -----------------------------------------------------------

    def run(self, ticks: int = 10) -> list[TickReport]:
        return [self.step() for _ in range(ticks)]

    def stream(self, ticks: int = 10) -> Iterator[TickReport]:
        """Same as :meth:`run` but yields as it goes, for live output."""
        for _ in range(ticks):
            yield self.step()

    def finish(self) -> None:
        """Flush working memory into long-term. Call when a run ends."""
        self.character.memory.flush(self.world.tick)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        actions: dict[str, int] = {}
        sources: dict[str, int] = {}
        for report in self.reports:
            actions[report.action_name] = actions.get(report.action_name, 0) + 1
            sources[report.decision.source] = sources.get(report.decision.source, 0) + 1
        summary: dict[str, Any] = {
            "ticks": len(self.reports),
            "world": self.world.snapshot(),
            "character": self.character.snapshot(),
            "actions": dict(sorted(actions.items(), key=lambda kv: -kv[1])),
            "decision_sources": sources,
            "energy_spent_by_reason": {
                k: round(v, 1) for k, v in self.character.energy.spent_by_reason().items()
            },
        }
        if self.router:
            summary["llm"] = {
                "summary": self.router.tracker.summary(),
                "by_provider": self.router.tracker.by_provider(),
            }
        return summary
