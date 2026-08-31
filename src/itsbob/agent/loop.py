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

from ..llm.base import AllProvidersFailed, LLMRequest, assistant, system
from ..router.gatekeeper import Gatekeeper
from ..router.ingestion import Snapshot, compress
from ..router.tiers import Tier
from ..tools import ToolCall, Toolbox
from .brain import TieredBrain
from .budget import FeasibilityCheck, SpendGuard, Verdict
from .completeness import REWRITE_INSTRUCTION, inspect as inspect_completeness
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

#: Keep the descriptions for these on every step. They are the tools a model
#: reaches for *late* in a turn — after the work is done and it is deciding
#: what to write down or how to finish — so they are the ones whose prose is
#: still doing something at step nine.
_ALWAYS_DESCRIBED = frozenset({"remember", "recall", "keep_memory", "forget"})


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

    def chat(
        self,
        message: str,
        *,
        on_event: EventFn | None = None,
        context: Any = None,
        min_tier: Tier | None = None,
        thorough: bool = False,
    ) -> Turn:
        """Run one full turn and return everything that happened in it.

        ``min_tier`` is a floor, not an override: the classifier still runs and
        may pick something higher. It exists for work nobody is watching — a
        scheduled task has no one to say "no, do it properly", so a classifier
        that reads "write a report on X" as one quick lookup has nothing to
        correct it.

        ``thorough`` says finish the job rather than sketch it: more steps
        before the budget has to justify itself, and the persona told as much.
        """
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
        if min_tier is not None and min_tier.rank > tier.rank:
            tier = min_tier
        turn.tier = tier.value
        emit(
            "classified",
            tier=tier.value,
            floor=min_tier.value if min_tier else None,
            decision=decision.as_dict(),
        )

        verdict = self._feasible(message, tier, emit)
        if not verdict.feasible:
            turn.final = verdict.explain()
            turn.tier = tier.value
            turn.refused = verdict.reason
            turn.duration_ms = (time.perf_counter() - started) * 1000
            self.conversation.add(turn)
            emit("final", text=turn.final, tier=tier.value, steps=0, refused=True)
            return turn

        self._load_style()
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
            thorough=thorough,
        )

        answer = self._pay_out_the_list(answer, turn, tier, emit)
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

    def _pay_out_the_list(
        self, answer: str, turn: Turn, tier: Tier, emit: Callable[..., None]
    ) -> str:
        """Rewrite an answer that announced a list and then did not give one.

        Checked for free on the finished text, and only rewritten when a tool
        this turn actually returned several rows *and* the answer stood in a
        count or a hedge for them. Both together are rare, so the second call
        is rare — which is the point: a verification pass on every turn would
        double the bill to catch a minority of turns.
        """
        shortfall = inspect_completeness(answer, [s.observation for s in turn.steps])
        if not shortfall.short:
            return answer

        emit("incomplete", **shortfall.as_dict())
        messages = build_messages(
            persona=self.persona,
            tools="(answer from what you already have; no tools for this step)",
            snapshot_text=turn.message,
            conversation=self.conversation,
            steps=turn.steps,
            full_observations=len(turn.steps) or 1,
            observation_chars=6000,
        )
        messages.append(assistant(answer))
        messages.append(system(f"{REWRITE_INSTRUCTION}\n\nHere, {shortfall.note()}."))
        try:
            payload, result = self.brain.complete_json(
                tier,
                LLMRequest(messages=messages, temperature=0.2, max_tokens=2500),
                purpose="agent.enumerate",
                default={},
            )
            turn.tokens += result.response.usage.total_tokens
            self.guard.add(result.response.usage.total_tokens)
        except AllProvidersFailed:
            return answer
        rewritten = str(payload.get("final") or "").strip()
        if not rewritten:
            return answer
        turn.rewritten_for_completeness = True
        emit("rewritten", listed=shortfall.listed, available=shortfall.available)
        return rewritten

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
            # Promote before pruning, so a row that has earned permanence is
            # not dropped by the working set on the same pass.
            kept = store.consolidate() if hasattr(store, "consolidate") else []
            expired = store.expire() if hasattr(store, "expire") else 0
            dropped = store.prune_short_term() if hasattr(store, "prune_short_term") else 0
        except Exception:  # noqa: BLE001 - housekeeping must never fail a turn
            return
        if kept or expired or dropped:
            emit("memory", promoted=len(kept), expired=expired, pruned=dropped)

    def _run_steps(
        self,
        *,
        snapshot: Snapshot,
        turn: Turn,
        tier: Tier,
        memories: Sequence[Any],
        deadline: float,
        emit: Callable[..., None],
        thorough: bool = False,
    ) -> tuple[str, Tier]:
        guard = TurnGuard()
        # Thorough work starts with the room to be thorough. The budget still
        # extends itself when a turn is being productive, but making a report
        # earn its sixth step one at a time is how a report comes out as a
        # paragraph: the model can see the budget, and it writes to fit.
        budget = self.max_steps * 2 if thorough else self.max_steps
        budget = min(budget, self.hard_max_steps)
        index = 0
        while index < budget:
            index += 1
            if time.perf_counter() > deadline:
                return self._forced_answer(snapshot, turn, tier, "the time budget ran out"), tier
            overspent = self.guard.exceeded()
            if overspent:
                return self._forced_answer(snapshot, turn, tier, overspent), tier

            brief = tier in _BRIEF_TIERS and not thorough
            # After the first step, tool *descriptions* stop earning their
            # keep: choosing is done, and what remains is calling. Everything
            # stays listed and callable — the roster below is complete — but
            # only the tools already in play keep their prose. Measured at 37
            # tools this is ~1,900 tokens a step down to ~650.
            in_play = {step.tool for step in turn.steps if step.tool}
            first_step = not turn.steps
            messages = build_messages(
                persona=self.persona,
                tools=self.toolbox.render_for_prompt(
                    describe_only=None if first_step else (in_play | _ALWAYS_DESCRIBED)
                ),
                snapshot_text=snapshot.render(),
                conversation=self.conversation,
                memories=memories,
                steps=turn.steps,
                # The API block is a fixed cost paid on every step. A cheap tier
                # doing one obvious thing does not need the catalogue, and
                # neither does a step that is not about to call an API.
                apis=self._api_block(brief, first_step, in_play),
                workspace=self.toolbox.policy.workspace,
                policy_note="" if brief else _policy_note(self.toolbox),
                tool_names=self.toolbox.registry.names(),
                brief=brief,
                thorough=thorough,
                continuing=not first_step,
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

    def _api_block(self, brief: bool, first_step: bool, in_play: set[str]) -> str:
        """The configured-API list, when it can still change a decision.

        Shown on the first step (where the API might be chosen) and on any
        later step that has already reached for one (where the worked examples
        are what fix a bad call). Otherwise it is ~340 tokens of catalogue paid
        for nothing.
        """
        catalog = self.toolbox.catalog
        if catalog is None or not len(catalog):
            return ""
        if "call_api" in in_play:
            # Already reaching for an API: the worked examples are exactly what
            # turns a failed call into a right one, so they are shown even on a
            # cheap tier that otherwise skips this block.
            return catalog.render_for_prompt(self.toolbox.env)
        return "" if brief or not first_step else catalog.render_for_prompt(self.toolbox.env)

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

    #: How often standing style preferences are re-read from memory. They
    #: change rarely, and a query per turn to find that out is a query wasted.
    STYLE_TTL = 120.0

    def _load_style(self) -> None:
        """Put the user's standing answer-style preferences into the prompt.

        A preference like "always list every match, never summarise" is exactly
        the sort of thing that has to be said once and then hold. Storing it as
        an ordinary memory means it is only recalled when the *query* happens
        to look similar — which for a rule about formatting it never does. So
        memories tagged `style` are loaded directly into the persona instead.
        """
        if self.memory is None:
            return
        now = time.time()
        if now - getattr(self, "_style_read_at", 0.0) < self.STYLE_TTL:
            return
        self._style_read_at = now
        try:
            found = self.memory.by_tag("style", limit=8)
        except Exception:  # noqa: BLE001 - a preference is an assist, never a dependency
            return
        self.persona.style = tuple(dict.fromkeys(r.content.strip() for r in found if r.content.strip()))

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
