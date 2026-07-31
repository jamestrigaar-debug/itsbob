"""Actions: the verbs available to the character each tick.

An action declares what it costs, what it relieves, and when it is even
possible; its ``effect`` does the work and reports back. The registry is the
extension point — a new verb is a :class:`Action` and a function, and every
policy picks it up for free.

``consult_oracle`` is the one that matters: it is how the character reaches an
LLM, and it is priced in the same energy everything else spends.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from ..memory.base import MemoryKind
from .energy import EnergyLedger, TokenCostModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..llm.router import LLMRouter
    from .state import Character

__all__ = [
    "ActionContext",
    "ActionResult",
    "Action",
    "ActionRegistry",
    "default_registry",
]


@dataclass
class ActionContext:
    """Everything an effect is allowed to touch."""

    character: "Character"
    world: Any
    tick: int
    rng: random.Random = field(default_factory=random.Random)
    router: "LLMRouter | None" = None
    cost_model: TokenCostModel = field(default_factory=TokenCostModel)
    #: Free-form hints from the decision layer (e.g. the oracle question).
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def energy(self) -> EnergyLedger:
        return self.character.energy


@dataclass
class ActionResult:
    """What an action did."""

    ok: bool = True
    narrative: str = ""
    #: Energy spent *beyond* the action's declared base cost (LLM tokens, etc).
    extra_energy: float = 0.0
    energy_gained: float = 0.0
    needs_delta: dict[str, float] = field(default_factory=dict)
    mood_delta: float = 0.0
    #: (content, kind, importance) triples to write into memory.
    memories: list[tuple[str, MemoryKind, float]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def remember(
        self,
        content: str,
        kind: MemoryKind = MemoryKind.ACTION,
        importance: float = 0.5,
    ) -> "ActionResult":
        self.memories.append((content, kind, importance))
        return self


Effect = Callable[[ActionContext], ActionResult]
Precondition = Callable[[ActionContext], bool]


@dataclass
class Action:
    name: str
    description: str
    effect: Effect
    energy_cost: float = 0.0
    #: need name -> how much this action relieves it (used by the heuristic policy)
    satisfies: dict[str, float] = field(default_factory=dict)
    precondition: Precondition | None = None
    #: True if the action may make an LLM call, so cost is only an estimate.
    uses_llm: bool = False
    tags: tuple[str, ...] = ()

    def is_available(self, ctx: ActionContext) -> bool:
        if not ctx.energy.can_afford(self.energy_cost):
            return False
        if self.uses_llm and ctx.router is None:
            return False
        return self.precondition(ctx) if self.precondition else True

    def run(self, ctx: ActionContext) -> ActionResult:
        return self.effect(ctx)

    def describe(self) -> str:
        cost = f"~{self.energy_cost:.0f}+" if self.uses_llm else f"{self.energy_cost:.0f}"
        return f"- {self.name} (costs {cost} energy): {self.description}"


class ActionRegistry:
    """Ordered, name-addressable set of actions."""

    def __init__(self, actions: Iterable[Action] = ()) -> None:
        self._actions: dict[str, Action] = {}
        for action in actions:
            self.register(action)

    def register(self, action: Action) -> Action:
        self._actions[action.name] = action
        return action

    def add(
        self,
        name: str,
        description: str,
        *,
        energy_cost: float = 0.0,
        satisfies: dict[str, float] | None = None,
        precondition: Precondition | None = None,
        uses_llm: bool = False,
        tags: Sequence[str] = (),
    ) -> Callable[[Effect], Effect]:
        """Decorator form: ``@registry.add("nap", "sleep it off", energy_cost=0)``."""

        def decorator(effect: Effect) -> Effect:
            self.register(
                Action(
                    name=name,
                    description=description,
                    effect=effect,
                    energy_cost=energy_cost,
                    satisfies=dict(satisfies or {}),
                    precondition=precondition,
                    uses_llm=uses_llm,
                    tags=tuple(tags),
                )
            )
            return effect

        return decorator

    def get(self, name: str) -> Action | None:
        return self._actions.get(name)

    def all(self) -> list[Action]:
        return list(self._actions.values())

    def available(self, ctx: ActionContext) -> list[Action]:
        return [a for a in self._actions.values() if a.is_available(ctx)]

    def names(self) -> list[str]:
        return list(self._actions)

    def render(self, actions: Sequence[Action] | None = None) -> str:
        return "\n".join(a.describe() for a in (actions or self.all()))

    def __len__(self) -> int:
        return len(self._actions)

    def __contains__(self, name: object) -> bool:
        return name in self._actions


# --------------------------------------------------------------------------
# Built-in actions
# --------------------------------------------------------------------------


def _rest(ctx: ActionContext) -> ActionResult:
    recovered = ctx.character.energy.regen_per_tick * 2.0
    result = ActionResult(
        narrative=f"{ctx.character.name} sits down and does nothing on purpose.",
        energy_gained=recovered,
        needs_delta={"rest": -0.45, "purpose": 0.05},
        mood_delta=0.02,
    )
    return result.remember(
        f"Rested at {ctx.character.location}; recovered {recovered:.0f} energy.",
        MemoryKind.ACTION,
        0.2,
    )


def _eat(ctx: ActionContext) -> ActionResult:
    recovered = ctx.character.energy.regen_per_tick * 1.5
    return ActionResult(
        narrative=f"{ctx.character.name} eats, without ceremony.",
        energy_gained=recovered,
        needs_delta={"sustenance": -0.6},
        mood_delta=0.05,
    ).remember("Ate a meal.", MemoryKind.ACTION, 0.15)


def _observe(ctx: ActionContext) -> ActionResult:
    detail = ctx.world.observe(ctx.rng) if hasattr(ctx.world, "observe") else "not much"
    return ActionResult(
        narrative=f"{ctx.character.name} looks around: {detail}",
        needs_delta={"curiosity": -0.15},
    ).remember(f"Observed: {detail}", MemoryKind.OBSERVATION, 0.35)


def _work(ctx: ActionContext) -> ActionResult:
    progress = ctx.rng.uniform(0.05, 0.2) * (0.5 + ctx.character.traits.diligence)
    if hasattr(ctx.world, "advance_project"):
        ctx.world.advance_project(progress)
    return ActionResult(
        narrative=f"{ctx.character.name} works the problem; it moves a little.",
        needs_delta={"purpose": -0.4, "rest": 0.1},
        mood_delta=0.06,
        data={"progress": progress},
    ).remember(f"Made {progress:.2f} progress on the work.", MemoryKind.ACTION, 0.5)


def _socialize(ctx: ActionContext) -> ActionResult:
    other = ctx.world.random_neighbour(ctx.rng) if hasattr(ctx.world, "random_neighbour") else None
    who = other or "nobody in particular"
    return ActionResult(
        narrative=f"{ctx.character.name} talks with {who}.",
        needs_delta={"social": -0.5},
        mood_delta=0.08 * (0.5 + ctx.character.traits.sociability),
    ).remember(f"Talked with {who}.", MemoryKind.DIALOGUE, 0.4)


def _journal(ctx: ActionContext) -> ActionResult:
    pressing, level = ctx.character.needs.most_pressing()
    note = (
        f"Feeling {ctx.character.mood_word}; {pressing} is the loudest thing "
        f"at {level:.2f}, energy {ctx.character.energy.current:.0f}."
    )
    return ActionResult(
        narrative=f"{ctx.character.name} writes it down.",
        needs_delta={"purpose": -0.15},
    ).remember(note, MemoryKind.REFLECTION, 0.55)


def _reflect(ctx: ActionContext) -> ActionResult:
    """Deliberate recall — cheap if the LLM is unavailable, richer if not."""
    insights = ctx.character.memory.reflect(ctx.tick)
    if insights:
        joined = "; ".join(i.content for i in insights)
        return ActionResult(
            narrative=f"{ctx.character.name} thinks it over and concludes: {joined}",
            needs_delta={"purpose": -0.3, "curiosity": -0.1, "rest": 0.08},
            mood_delta=0.04,
            data={"insights": [i.content for i in insights]},
        )
    recalled = ctx.character.memory.recall("what matters", limit=3, tick=ctx.tick)
    summary = "; ".join(r.content for r in recalled) or "nothing in particular"
    return ActionResult(
        narrative=f"{ctx.character.name} turns things over: {summary}",
        needs_delta={"purpose": -0.15, "rest": 0.05},
    ).remember(f"Dwelt on: {summary}", MemoryKind.REFLECTION, 0.5)


def _consult_oracle(ctx: ActionContext) -> ActionResult:
    """Ask an LLM a question. The one action whose price isn't known up front.

    Cost is billed from real token usage after the call, so a rambling answer is
    genuinely more expensive than a terse one.
    """
    from ..llm.base import AllProvidersFailed, LLMRequest, system, user

    assert ctx.router is not None  # guaranteed by Action.is_available

    question = ctx.params.get("question") or _default_question(ctx)
    context_block = ctx.character.memory.render_context(question, limit=5, tick=ctx.tick)
    request = LLMRequest(
        messages=[
            system(
                "You are an oracle consulted by a character in a simulation. "
                "Answer in at most three sentences, concrete and actionable. "
                "If the question can't be answered, say so plainly."
            ),
            user(
                f"{ctx.character.sheet()}\n\n"
                f"Relevant memories:\n{context_block}\n\n"
                f"Question: {question}"
            ),
        ],
        temperature=0.6,
        max_tokens=400,
    )

    affordable = ctx.cost_model.affordable_max_tokens(
        ctx.energy, request.approx_prompt_tokens()
    )
    if affordable < 64:
        return ActionResult(
            ok=False,
            narrative=f"{ctx.character.name} reaches for the oracle and finds no strength for it.",
            needs_delta={"curiosity": 0.05},
            mood_delta=-0.05,
        )
    request.max_tokens = min(request.max_tokens, affordable)

    try:
        response = ctx.router.complete(request, purpose="oracle")
    except AllProvidersFailed as exc:
        return ActionResult(
            ok=False,
            narrative=f"{ctx.character.name} asks, and the oracle is silent.",
            extra_energy=ctx.cost_model.call_overhead,  # the attempt still costs
            needs_delta={"curiosity": 0.08},
            mood_delta=-0.08,
            data={"error": str(exc)[:200]},
        ).remember(
            f"Asked the oracle {question!r} and got nothing back.",
            MemoryKind.OBSERVATION,
            0.4,
        )

    answer = response.text.strip()
    cost = ctx.cost_model.energy_for(response.usage)
    return ActionResult(
        narrative=f"{ctx.character.name} consults the oracle — {answer}",
        extra_energy=cost,
        needs_delta={"curiosity": -0.55, "rest": 0.1},
        mood_delta=0.05,
        data={
            "question": question,
            "answer": answer,
            "provider": response.provider,
            "model": response.model,
            "tokens": response.usage.total_tokens,
            "energy_cost": cost,
        },
    ).remember(
        f"Asked the oracle {question!r}. It said: {answer}",
        MemoryKind.FACT,
        0.75,
    )


def _default_question(ctx: ActionContext) -> str:
    pressing, level = ctx.character.needs.most_pressing()
    return (
        f"I am {ctx.character.name}, at {ctx.character.location}. "
        f"My {pressing} need is at {level:.2f} and my goal is: {ctx.character.goal}. "
        "What is the single most useful thing to do next, and why?"
    )


def default_registry() -> ActionRegistry:
    """The starter verb set. Extend or replace it wholesale."""
    return ActionRegistry(
        [
            Action(
                name="rest",
                description="Stop and recover energy.",
                effect=_rest,
                energy_cost=0.0,
                satisfies={"rest": 0.45},
                tags=("recovery",),
            ),
            Action(
                name="eat",
                description="Find food and eat it.",
                effect=_eat,
                energy_cost=1.0,
                satisfies={"sustenance": 0.6},
                tags=("recovery",),
            ),
            Action(
                name="observe",
                description="Study the surroundings and note what has changed.",
                effect=_observe,
                energy_cost=3.0,
                satisfies={"curiosity": 0.15},
                tags=("perception",),
            ),
            Action(
                name="work",
                description="Make progress on the current project.",
                effect=_work,
                energy_cost=12.0,
                satisfies={"purpose": 0.4},
                tags=("labour",),
            ),
            Action(
                name="socialize",
                description="Seek out someone and talk.",
                effect=_socialize,
                energy_cost=7.0,
                satisfies={"social": 0.5},
                tags=("social",),
            ),
            Action(
                name="journal",
                description="Write down the current state of things.",
                effect=_journal,
                energy_cost=2.0,
                satisfies={"purpose": 0.15},
                tags=("memory",),
            ),
            Action(
                name="reflect",
                description="Draw conclusions from recent memories.",
                effect=_reflect,
                energy_cost=5.0,
                satisfies={"purpose": 0.3, "curiosity": 0.1},
                tags=("memory",),
            ),
            Action(
                name="consult_oracle",
                description="Spend energy to ask an LLM a question and remember the answer.",
                effect=_consult_oracle,
                energy_cost=2.0,
                satisfies={"curiosity": 0.55},
                uses_llm=True,
                tags=("llm", "information"),
            ),
        ]
    )
