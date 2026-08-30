"""The agent loop: classify, recall, think, act, observe, repeat.

One turn is:

1. **Classify** the message into a tier (:mod:`itsbob.router.gatekeeper`).
2. **Recall** relevant memory, before any model call — retrieval is cheaper
   than reasoning, and a model that already has the fact does not have to
   decide to go looking for it.
3. **Step**, up to a budget: the tier's model returns one JSON object naming
   either a tool call or a final answer. A tool call runs through the gated
   registry and its result becomes the next step's observation.
4. **Extract** durable facts from the finished turn on the cheapest tier.

Design decisions worth naming:

**The tier is chosen once per turn, then raised, never lowered.** A turn that
needed the premium model for step 1 does not get cheaper at step 4 — it has
already demonstrated it is hard. Escalation is one-way within a turn.

**A refused tool call is an observation, not an abort.** "You may not do that
because X" goes back into the scratchpad, so the agent can explain the block
to the user or find a permitted route. The one exception is a user saying no
at a confirm prompt: that is a decision, not an obstacle, and the loop stops
rather than looking for a way around it.

**The step budget extends itself while there is progress.** A fixed budget was
ending real work mid-task: the agent would be four files into a six-file job,
hit step eight, and stop to explain where it got to. So the budget is now a
*checkpoint*, not a wall. When it runs out, the loop asks whether the last
stretch actually achieved anything — successful tool calls, no repeats, no
invented names — and if so extends itself, up to a hard ceiling and inside the
time and token limits. A turn that is going nowhere still stops at the first
checkpoint, which is the case the budget was there for.

**The budget always produces an answer.** Whatever ends a turn — the ceiling,
the clock, the token guard — forces one final call with the tools removed, so a
hard turn ends with "here is what I found and where I got stuck" instead of
silence.

**A turn it cannot finish should not start.** Before expensive work, one cheap
call checks the request against the tools that actually exist (see
:mod:`itsbob.agent.budget`). Discovering "there is no API key for that" in one
small call beats discovering it eight premium steps later.

**An identical call never runs twice in a turn.** :class:`TurnGuard` caches
each (tool, arguments) pair and replays the first result instead of executing
again. This started as loop control — a model that had just written a file
would write it another seven times — but the safety property is the more
important one: re-running a mutating call has real side effects, and "the
model looped" should never mean "the email went out twice".

**Invented tool names are counted, not just refused.** The registry already
rejects them, but a model that answers a rejection by inventing a *different*
wrong name will spend the entire budget doing it. After two, the tier is
raised once; after three, the turn ends with an explanation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..llm.base import AllProvidersFailed, LLMRequest, system
from ..router.gatekeeper import Gatekeeper
from ..router.ingestion import Snapshot, compress
from ..router.tiers import Tier
from ..tools import ToolCall, Toolbox
from .brain import TieredBrain
from .budget import FeasibilityCheck, SpendGuard, Verdict
from .context import Conversation, Step, Turn, build_messages
from .persona import Persona
from .writer import MemoryWriter

__all__ = ["Agent", "AgentEvent"]


@dataclass
class AgentEvent:
    """Progress notification, so a UI can show thinking rather than a spinner."""

    kind: str  #: classified | step | tool | observation | final | error | memory
    turn: int
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "turn": self.turn, "data": self.data}


EventFn = Callable[[AgentEvent], None]

#: Raising within a turn only. A turn never gets cheaper than it started.
_RAISE = {Tier.C: Tier.B, Tier.B: Tier.A, Tier.A: Tier.S, Tier.S: Tier.S,
          Tier.D: Tier.B, Tier.H: Tier.A}

#: How many invented tool names to tolerate before giving up on the turn.
MAX_UNKNOWN_TOOLS = 3
#: How many consecutive failing tool calls before giving up on the turn.
MAX_CONSECUTIVE_FAILURES = 3

#: Tiers cheap enough that the full system prompt costs more than it earns.
_BRIEF_TIERS = frozenset({Tier.C, Tier.B})


class TurnGuard:
    """Stops the three ways a step loop wastes a budget without progressing."""

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}
        self.unknown_tools = 0
        self.consecutive_failures = 0
        self.repeats = 0

    @staticmethod
    def key(name: str, params: dict[str, Any]) -> str:
        try:
            return f"{name}:{json.dumps(params, sort_keys=True, default=str)}"
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return f"{name}:{params!r}"

    def cached(self, name: str, params: dict[str, Any]) -> Any | None:
        """The result of an identical earlier call this turn, if there was one.

        Replaying beats re-executing: it is faster, and it means a looping
        model cannot apply a mutating call twice.
        """
        hit = self.seen.get(self.key(name, params))
        if hit is not None:
            self.repeats += 1
        return hit

    def record(self, name: str, params: dict[str, Any], result: Any) -> None:
        self.seen.setdefault(self.key(name, params), result)
        if result.ok:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if "no tool named" in (result.error or ""):
                self.unknown_tools += 1

    @property
    def give_up_reason(self) -> str | None:
        if self.unknown_tools >= MAX_UNKNOWN_TOOLS:
            return f"the model kept naming tools that do not exist ({self.unknown_tools} times)"
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return f"{self.consecutive_failures} tool calls failed in a row"
        if self.repeats >= 3:
            return "the model kept repeating calls it had already made"
        return None


class Agent:
    """A memory-backed, tool-using assistant over the tier ladder."""

    def __init__(
        self,
        *,
        brain: TieredBrain,
        toolbox: Toolbox,
        memory: Any = None,
        persona: Persona | None = None,
        gatekeeper: Gatekeeper | None = None,
        writer: MemoryWriter | None = None,
        conversation: Conversation | None = None,
        max_steps: int = 10,
        max_seconds: float = 600.0,
        recall_limit: int = 6,
        hard_max_steps: int = 60,
        extend_by: int = 6,
        guard: SpendGuard | None = None,
        feasibility: FeasibilityCheck | None = None,
    ) -> None:
        self.brain = brain
        self.toolbox = toolbox
        self.memory = memory if memory is not None else toolbox.memory
        self.persona = persona or Persona()
        self.conversation = conversation or Conversation()
        self.max_steps = max(1, max_steps)
        #: The wall the budget may never extend past, however well it is going.
        #: Something has to be finite, and this is it.
        self.hard_max_steps = max(self.max_steps, hard_max_steps)
        #: How many steps a productive turn earns at each checkpoint.
        self.extend_by = max(1, extend_by)
        self.max_seconds = max_seconds
        self.recall_limit = recall_limit
        self.guard = guard if guard is not None else SpendGuard()
        self.feasibility = (
            feasibility if feasibility is not None else FeasibilityCheck(brain=brain)
        )
        self.gatekeeper = gatekeeper or Gatekeeper(
            local_provider=brain.local, cloud_classifier=self._cheap_classify
        )
        self.writer = writer
        if self.writer is None and self.memory is not None:
            self.writer = MemoryWriter(brain=brain, store=self.memory)

    # -- the turn ----------------------------------------------------------

    def chat(self, message: str, *, on_event: EventFn | None = None, context: Any = None) -> Turn:
        """Run one full turn and return everything that happened in it."""
        started = time.perf_counter()
        turn_index = len(self.conversation) + 1
        turn = Turn(message=message)

        def emit(kind: str, **data: Any) -> None:
            if on_event is not None:
                try:
                    on_event(AgentEvent(kind=kind, turn=turn_index, data=data))
                except Exception:  # noqa: BLE001 - a broken listener must not fail the turn
                    pass

        self.guard.start_turn()
        snapshot = self._snapshot(message, context)
        decision = self.gatekeeper.classify(snapshot)
        tier = decision.tier if decision.tier.is_model else Tier.B
        turn.tier = tier.value
        emit("classified", tier=tier.value, decision=decision.as_dict())

        verdict = self._feasible(message, tier, emit)
        if not verdict.feasible:
            turn.final = verdict.explain()
            turn.tier = tier.value
            turn.refused = verdict.reason
            turn.duration_ms = (time.perf_counter() - started) * 1000
            self.conversation.add(turn)
            emit("final", text=turn.final, tier=tier.value, steps=0, refused=True)
            return turn

        memories = self._recall(message)
        if memories:
            emit("memory", recalled=[h.as_dict() for h in memories])

        deadline = started + self.max_seconds
        answer, tier = self._run_steps(
            snapshot=snapshot,
            turn=turn,
            tier=tier,
            memories=memories,
            deadline=deadline,
            emit=emit,
        )

        turn.final = answer
        turn.tier = tier.value
        turn.duration_ms = (time.perf_counter() - started) * 1000
        self.conversation.add(turn)
        emit("final", text=answer, tier=tier.value, steps=len(turn.steps))

        if self.writer is not None and answer:
            # Anything the agent already wrote with `remember` this turn counts
            # as known. Without this the writer re-extracts what was just
            # stored, in slightly different words, and near-duplicates are the
            # one thing that degrades recall fastest: three rows saying the
            # same thing all surface with equal confidence, and none of them is
            # the one you would have written.
            known = [*memories, *_written_this_turn(turn)]
            for record in self.writer.write(message=message, answer=answer, known=known):
                turn.remembered.append(record.content)
                emit(
                    "memory",
                    wrote=record.content,
                    id=record.id,
                    subject=record.subject.value,
                    horizon=record.horizon.value,
                )

        self._tidy_memory(emit)
        return turn

    def _feasible(self, message: str, tier: Tier, emit: Callable[..., None]) -> Verdict:
        """Screen an expensive turn before paying for it."""
        if self.feasibility is None or not self.feasibility.should_check(message, tier):
            return Verdict()
        verdict = self.feasibility.check(
            message=message,
            tools=self.toolbox.registry.names(),
            apis=self.toolbox.catalog.names() if self.toolbox.catalog is not None else (),
        )
        if verdict.checked:
            emit("feasibility", **verdict.as_dict())
        return verdict

    def _tidy_memory(self, emit: Callable[..., None]) -> None:
        """Expire the short-term working set once per turn.

        Here rather than on a timer because "a few replies" is measured in
        replies: pruning when a turn ends is the only moment that reliably
        happens once per reply, whether the agent is being chatted to or is
        working through its own schedule.
        """
        store = self.memory
        if store is None:
            return
        try:
            expired = store.expire() if hasattr(store, "expire") else 0
            dropped = store.prune_short_term() if hasattr(store, "prune_short_term") else 0
        except Exception:  # noqa: BLE001 - housekeeping must never fail a turn
            return
        if expired or dropped:
            emit("memory", expired=expired, pruned=dropped)

    def _run_steps(
        self,
        *,
        snapshot: Snapshot,
        turn: Turn,
        tier: Tier,
        memories: Sequence[Any],
        deadline: float,
        emit: Callable[..., None],
    ) -> tuple[str, Tier]:
        guard = TurnGuard()
        budget = self.max_steps
        index = 0
        while index < budget:
            index += 1
            if time.perf_counter() > deadline:
                return self._forced_answer(snapshot, turn, tier, "the time budget ran out"), tier
            overspent = self.guard.exceeded()
            if overspent:
                return self._forced_answer(snapshot, turn, tier, overspent), tier

            brief = tier in _BRIEF_TIERS
            messages = build_messages(
                persona=self.persona,
                tools=self.toolbox.render_for_prompt(),
                snapshot_text=snapshot.render(),
                conversation=self.conversation,
                memories=memories,
                steps=turn.steps,
                # The API block is a fixed cost paid on every step; a cheap tier
                # doing one obvious thing does not need the catalogue.
                apis=""
                if brief
                else (
                    self.toolbox.catalog.render_for_prompt(self.toolbox.env)
                    if self.toolbox.catalog is not None and len(self.toolbox.catalog)
                    else ""
                ),
                workspace=self.toolbox.policy.workspace,
                policy_note="" if brief else _policy_note(self.toolbox),
                tool_names=self.toolbox.registry.names(),
                brief=brief,
                observation_chars=_observation_budget(len(turn.steps)),
            )

            step = Step(index=index, tier=tier.value)
            try:
                payload, result = self.brain.complete_json(
                    tier,
                    LLMRequest(messages=messages, temperature=0.3, max_tokens=2000),
                    purpose="agent.step",
                )
            except AllProvidersFailed as exc:
                turn.error = str(exc)
                emit("error", message=str(exc))
                return (
                    "Every model tier failed, so I could not answer. "
                    f"Last errors: {str(exc)[:300]}",
                    tier,
                )

            tier = result.tier
            step.tier = tier.value
            step.model = f"{result.response.provider}/{result.response.model}"
            step.latency_ms = result.response.latency_ms
            turn.tokens += result.response.usage.total_tokens
            step.tokens = result.response.usage.total_tokens
            if result.response.provider != "ollama":
                # Local calls are free; counting them against a spend ceiling
                # would penalise the thing the ceiling exists to encourage.
                self.guard.add(result.response.usage.total_tokens)
            step.thought = str(payload.get("thought") or "").strip()

            final = payload.get("final")
            tool_name = payload.get("tool")
            if isinstance(tool_name, str) and tool_name.strip().lower() in ("null", "none", ""):
                tool_name = None

            if not tool_name:
                text = str(final or step.thought or "").strip()
                if text:
                    turn.steps.append(step)
                    emit("step", **step.as_dict())
                    return text, tier
                # Neither a tool nor an answer: a malformed reply. Raising the
                # tier is the cheapest correction available and usually works.
                raised = _RAISE[tier]
                step.observation = "reply named neither a tool nor a final answer"
                step.ok = False
                turn.steps.append(step)
                emit("step", **step.as_dict())
                if raised is tier:
                    return self._forced_answer(snapshot, turn, tier, "the model would not answer"), tier
                tier = raised
                continue

            params = payload.get("params")
            if not isinstance(params, dict):
                params = {}
            step.tool = str(tool_name)
            step.params = params
            emit("tool", name=step.tool, params=params, thought=step.thought)

            replay = guard.cached(step.tool, params)
            if replay is not None:
                # Identical call, already made this turn. Replay it rather than
                # re-running: a mutating tool must not fire twice because the
                # model lost track of what it had done.
                tool_result = replay
                step.observation = (
                    "This exact call already ran earlier in this turn. Its result was:\n"
                    f"{replay.render()}\n"
                    "Do not call it again — use this result, or do something different."
                )
                step.ok = replay.ok
            else:
                call = ToolCall(name=step.tool, params=params, reason=step.thought)
                tool_result = self.toolbox.invoke(call)
                step.observation = tool_result.render()
                step.ok = tool_result.ok

            guard.record(step.tool, params, tool_result)
            turn.steps.append(step)
            emit("observation", tool=step.tool, ok=tool_result.ok, output=step.observation[:2000])

            if _user_refused(tool_result):
                # A person said no. That is an answer, not an obstacle.
                return (
                    f"Stopped: you declined the `{step.tool}` step, so I have not gone further.",
                    tier,
                )

            reason = guard.give_up_reason
            if reason:
                return self._forced_answer(snapshot, turn, tier, reason), tier

            if guard.unknown_tools == 2 and _RAISE[tier].rank > tier.rank:
                # A stronger model usually reads the tool list correctly where a
                # cheaper one pattern-matched to a plausible-sounding name.
                tier = _RAISE[tier]

            if index >= budget:
                extended = self._extend(budget, turn, guard, deadline)
                if extended is None:
                    break
                emit(
                    "budget_extended",
                    steps=extended,
                    was=budget,
                    ceiling=self.hard_max_steps,
                )
                budget = extended

        return self._forced_answer(snapshot, turn, tier, _stop_reason(budget, self.hard_max_steps)), tier

    def _extend(
        self, budget: int, turn: Turn, guard: TurnGuard, deadline: float
    ) -> int | None:
        """More steps, if the last stretch earned them. ``None`` to stop.

        "Earned" is deliberately mechanical rather than a judgement call — no
        extra model call decides this, since paying to ask whether to keep
        paying is its own kind of waste. A turn continues when it is doing
        things that work and is not repeating itself, and stops otherwise.
        """
        if budget >= self.hard_max_steps:
            return None
        if time.perf_counter() > deadline or self.guard.exceeded():
            return None
        if guard.give_up_reason or guard.repeats:
            return None
        # Progress means work that landed: a successful tool call somewhere in
        # the stretch just finished. A run of pure thinking with nothing to show
        # for it is exactly the turn the checkpoint is meant to catch.
        recent = turn.steps[-self.max_steps :]
        if not any(step.tool and step.ok for step in recent):
            return None
        turn.extensions += 1
        return min(self.hard_max_steps, budget + self.extend_by)

    # -- helpers -----------------------------------------------------------

    def _snapshot(self, message: str, context: Any) -> Snapshot:
        snapshot = compress(message)
        if context:
            extra = compress(context)
            # The message wins on overlap: background context is the older,
            # weaker claim about the world.
            merged = dict(extra.facts)
            merged.update(snapshot.facts)
            snapshot.facts = merged
            snapshot.events = [*extra.events, *snapshot.events]
            if extra.text and extra.text != snapshot.text:
                snapshot.text = f"{extra.text}\n\n{snapshot.text}".strip()
        return snapshot

    def _recall(self, query: str) -> list[Any]:
        if self.memory is None or not query.strip():
            return []
        try:
            return self.memory.search(query, limit=self.recall_limit)
        except Exception:  # noqa: BLE001 - memory is an assist, never a dependency
            return []

    def _cheap_classify(self, request: LLMRequest) -> str:
        """Gatekeeper's cloud fallback: classify on the cheapest tier available."""
        return self.brain.complete(Tier.C, request, purpose="gatekeeper", escalate=False).text

    def _forced_answer(self, snapshot: Snapshot, turn: Turn, tier: Tier, why: str) -> str:
        """Last call of a turn, with tools removed, so it always ends in words."""
        turn.stopped_because = why
        messages = build_messages(
            persona=self.persona,
            tools="(no tools available for this final step)",
            snapshot_text=snapshot.render(),
            conversation=self.conversation,
            steps=turn.steps,
            # The final call summarizes what happened, so it needs more of the
            # scratchpad than a working step does.
            full_observations=5,
            observation_chars=1500,
        )
        messages[-1] = system(
            f"Stop working: {why}. Answer the user now with what you have. "
            "Say plainly what you did, what you found, and what is still unresolved. "
            'Reply as JSON: {"final": "<your answer>"}'
        )
        try:
            payload, result = self.brain.complete_json(
                tier,
                LLMRequest(messages=messages, temperature=0.3, max_tokens=1200),
                purpose="agent.forced",
                default={},
            )
            turn.tokens += result.response.usage.total_tokens
            text = str(payload.get("final") or "").strip()
            if text:
                return text
        except AllProvidersFailed:
            pass

        done = ", ".join(turn.tools_used) or "nothing"
        return (
            f"I stopped because {why}. Steps completed: {len(turn.steps)} "
            f"(tools used: {done}). Ask again with a narrower request and I'll pick it up."
        )


class _Known:
    """Adapter so a plain string looks like a recall hit to the writer."""

    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content


def _written_this_turn(turn: Turn) -> list[_Known]:
    return [
        _Known(str(step.params.get("content", "")).strip())
        for step in turn.steps
        if step.tool == "remember" and step.ok and step.params.get("content")
    ]


def _user_refused(result: Any) -> bool:
    return bool(result.error and "declined by user" in result.error)


#: Observation clipping, tightened as a turn goes on. Early steps get room to
#: show a full file or a long listing; by step twelve the scratchpad is the
#: dominant cost and the model is working from its own notes anyway.
def _observation_budget(steps_so_far: int) -> int:
    if steps_so_far < 4:
        return 3000
    if steps_so_far < 10:
        return 1800
    return 900


def _stop_reason(budget: int, ceiling: int) -> str:
    if budget >= ceiling:
        return f"the hard step ceiling ({ceiling}) was reached"
    return "the step budget ran out with no progress to show for the last stretch"


def _policy_note(toolbox: Toolbox) -> str:
    policy = toolbox.policy
    lines = [f"Tool mode is `{policy.mode.value}`, working inside {policy.workspace}."]
    if policy.confirm is None:
        lines.append(
            "Nobody is available to approve anything right now, so tools needing "
            "confirmation will be refused. Prefer routes that do not need them, and "
            "say clearly when something needs a person."
        )
    else:
        lines.append("Steps that change things outside the workspace will be shown to the user first.")
    return " ".join(lines)
