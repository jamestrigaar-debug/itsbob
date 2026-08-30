"""Decision policies: how the character picks its next action.

Three of them, and the difference between them is what deliberation costs.
:class:`HeuristicPolicy` is free and always available. :class:`LLMPolicy` spends
energy to think with a model. :class:`HybridPolicy` decides which of those it
can currently afford — that choice, made every tick, is the game.

Every policy returns a :class:`Decision`, and a decision is itself remembered,
so the reasoning behind a choice becomes material for later recall.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..memory.base import MemoryRecord, render_records
from .actions import Action, ActionContext
from .energy import TokenCostModel

__all__ = [
    "DecisionContext",
    "Decision",
    "DecisionPolicy",
    "HeuristicPolicy",
    "LLMPolicy",
    "HybridPolicy",
    "build_policy",
]


@dataclass
class DecisionContext:
    """Everything a policy may look at."""

    action_ctx: ActionContext
    options: list[Action]
    recalled: list[MemoryRecord] = field(default_factory=list)

    @property
    def character(self):  # noqa: ANN201 - forwarding accessor
        return self.action_ctx.character

    @property
    def tick(self) -> int:
        return self.action_ctx.tick

    @property
    def rng(self) -> random.Random:
        return self.action_ctx.rng

    def option_names(self) -> list[str]:
        return [a.name for a in self.options]


@dataclass
class Decision:
    action: Action
    rationale: str = ""
    confidence: float = 0.5
    #: "instinct" | "deliberation" | "fallback"
    source: str = "instinct"
    #: Energy spent *deciding*, before the action itself runs.
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return (
            f"{self.action.name} via {self.source} "
            f"(confidence {self.confidence:.2f}): {self.rationale}"
        )


@runtime_checkable
class DecisionPolicy(Protocol):
    name: str

    def decide(self, ctx: DecisionContext) -> Decision: ...


class HeuristicPolicy:
    """Free, deterministic-ish scoring. The character's instincts.

    Scores each option by the need pressure it relieves, discounted by energy
    price, then nudged by traits. No LLM, no cost — this is always the fallback
    when deliberation is unaffordable or fails.
    """

    name = "heuristic"

    def __init__(self, *, exploration: float = 0.08) -> None:
        #: Small random jitter so a tie doesn't lock the character into a rut.
        self.exploration = exploration

    def score(self, action: Action, ctx: DecisionContext) -> float:
        character = ctx.character
        needs = character.needs
        relief = sum(
            needs[name] * amount for name, amount in action.satisfies.items()
        )

        # Energy price, softened so a cheap-but-useless action can't dominate.
        price = action.energy_cost / max(1.0, character.energy.capacity * 0.25)
        scarcity = 1.0 - character.energy.fraction
        score = relief - price * (0.4 + scarcity)

        # Traits amplify an existing need rather than inventing appetite: a
        # curious character reaches for the oracle *when curious*, not always.
        if "llm" in action.tags:
            score += 0.35 * character.traits.curiosity * needs["curiosity"]
        if "social" in action.tags:
            score += 0.25 * character.traits.sociability * needs["social"]
        if "labour" in action.tags:
            score += 0.25 * character.traits.diligence * needs["purpose"]
        if "recovery" in action.tags and character.energy.is_exhausted:
            score += 1.2  # exhaustion overrides preference

        return score + ctx.rng.uniform(0.0, self.exploration)

    def decide(self, ctx: DecisionContext) -> Decision:
        if not ctx.options:
            raise ValueError("no actions available")
        scored = sorted(
            ((self.score(a, ctx), a) for a in ctx.options), key=lambda p: -p[0]
        )
        best_score, action = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else best_score - 1.0
        margin = max(0.0, best_score - runner_up)
        pressing, level = ctx.character.needs.most_pressing()
        return Decision(
            action=action,
            rationale=f"instinct: {pressing} at {level:.2f}, {action.name} is the cheapest relief",
            confidence=min(0.95, 0.45 + margin),
            source="instinct",
            cost=0.0,
            metadata={"scores": {a.name: round(s, 3) for s, a in scored}},
        )


class LLMPolicy:
    """Deliberation: ask a model what to do, in character.

    Charges ``deliberation_cost`` up front plus the token cost of the call, and
    validates the reply against the actual option list — a hallucinated action
    name falls back to instinct rather than crashing the tick.
    """

    name = "llm"

    def __init__(
        self,
        *,
        deliberation_cost: float = 4.0,
        cost_model: TokenCostModel | None = None,
        fallback: DecisionPolicy | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> None:
        self.deliberation_cost = deliberation_cost
        self.cost_model = cost_model or TokenCostModel()
        self.fallback = fallback or HeuristicPolicy()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_prompt(self, ctx: DecisionContext) -> str:
        registry_block = "\n".join(a.describe() for a in ctx.options)
        world = getattr(ctx.action_ctx.world, "describe", lambda: "")()
        return (
            f"{ctx.character.sheet()}\n\n"
            f"Situation (tick {ctx.tick}): {world}\n\n"
            f"Memories that came to mind:\n{render_records(ctx.recalled)}\n\n"
            f"Available actions:\n{registry_block}\n\n"
            "Choose exactly one action. Prefer actions that relieve the most "
            "pressing need without spending energy you cannot spare.\n"
            'Reply as JSON: {"action": "<name>", "rationale": "<one sentence>", '
            '"confidence": 0.0-1.0}'
        )

    def decide(self, ctx: DecisionContext) -> Decision:
        router = ctx.action_ctx.router
        character = ctx.character
        if router is None or not ctx.options:
            return self._fall_back(ctx, "no router")

        ledger = character.energy
        if not character.can_deliberate(self.deliberation_cost):
            return self._fall_back(ctx, "too tired to deliberate")

        from ..llm.base import AllProvidersFailed, LLMRequest, system, user

        request = LLMRequest(
            messages=[
                system(
                    f"You are {character.name}. Decide what to do next, in character, "
                    "and answer only with JSON."
                ),
                user(self.build_prompt(ctx)),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        affordable = self.cost_model.affordable_max_tokens(
            ledger, request.approx_prompt_tokens()
        )
        if affordable < 64:
            return self._fall_back(ctx, "not enough energy for a useful answer")
        request.max_tokens = min(request.max_tokens, affordable)

        spent = self.deliberation_cost
        ledger.spend(spent, "deliberation", tick=ctx.tick)

        try:
            payload, response = router.complete_json(request, purpose="decision")
        except (AllProvidersFailed, ValueError) as exc:
            decision = self._fall_back(ctx, f"deliberation failed ({type(exc).__name__})")
            decision.cost = spent
            return decision

        token_cost = self.cost_model.energy_for(response.usage)
        # try_spend, not spend: the call already happened, so an over-budget
        # answer drains what is left rather than raising after the fact.
        if ledger.try_spend(token_cost, "deliberation_tokens", tick=ctx.tick):
            spent += token_cost
        else:
            spent += ledger.current
            ledger.spend(ledger.current, "deliberation_tokens", tick=ctx.tick)

        name = str(payload.get("action", "")).strip()
        action = next((a for a in ctx.options if a.name == name), None)
        if action is None:
            decision = self._fall_back(ctx, f"model chose unavailable action {name!r}")
            decision.cost = spent
            decision.metadata["llm_raw"] = payload
            return decision

        question = payload.get("question")
        if question:
            # Let the model supply the oracle prompt it intends to ask.
            ctx.action_ctx.params["question"] = str(question)

        return Decision(
            action=action,
            rationale=str(payload.get("rationale", "")).strip() or "no rationale given",
            confidence=_confidence(payload.get("confidence"), default=0.6),
            source="deliberation",
            cost=spent,
            metadata={
                "provider": response.provider,
                "model": response.model,
                "tokens": response.usage.total_tokens,
            },
        )

    def _fall_back(self, ctx: DecisionContext, why: str) -> Decision:
        decision = self.fallback.decide(ctx)
        decision.source = "fallback"
        decision.rationale = f"{why}; {decision.rationale}"
        return decision


class HybridPolicy:
    """Deliberate when it is worth it and affordable; otherwise act on instinct.

    "Worth it" is deliberately crude: high need pressure, or curiosity, or a
    close call between the top instinctive options. The point is that thinking
    is rationed, not that the rationing is clever.
    """

    name = "hybrid"

    def __init__(
        self,
        *,
        llm_policy: LLMPolicy | None = None,
        heuristic: HeuristicPolicy | None = None,
        pressure_threshold: float = 0.45,
        min_energy_fraction: float = 0.35,
        deliberation_chance: float = 0.5,
    ) -> None:
        self.heuristic = heuristic or HeuristicPolicy()
        self.llm_policy = llm_policy or LLMPolicy(fallback=self.heuristic)
        self.pressure_threshold = pressure_threshold
        self.min_energy_fraction = min_energy_fraction
        self.deliberation_chance = deliberation_chance

    def should_deliberate(self, ctx: DecisionContext) -> bool:
        character = ctx.character
        if ctx.action_ctx.router is None:
            return False
        if character.energy.fraction < self.min_energy_fraction:
            return False
        if not character.can_deliberate(self.llm_policy.deliberation_cost):
            return False
        if character.needs.pressure >= self.pressure_threshold:
            return True
        chance = self.deliberation_chance * character.traits.curiosity
        return ctx.rng.random() < chance

    def decide(self, ctx: DecisionContext) -> Decision:
        if self.should_deliberate(ctx):
            return self.llm_policy.decide(ctx)
        return self.heuristic.decide(ctx)


def _confidence(value: Any, *, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    if confidence > 1.0:  # models like answering 85 instead of 0.85
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def build_policy(
    name: str,
    *,
    cost_model: TokenCostModel | None = None,
    deliberation_cost: float = 4.0,
) -> DecisionPolicy:
    """Policy by name, for the CLI and config files."""
    heuristic = HeuristicPolicy()
    if name == "heuristic":
        return heuristic
    llm_policy = LLMPolicy(
        deliberation_cost=deliberation_cost,
        cost_model=cost_model,
        fallback=heuristic,
    )
    if name == "llm":
        return llm_policy
    if name == "hybrid":
        return HybridPolicy(llm_policy=llm_policy, heuristic=heuristic)
    raise ValueError(f"unknown policy {name!r} (want: heuristic, llm, hybrid)")
