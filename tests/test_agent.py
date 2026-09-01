"""The agent loop, its guards, and prompt assembly. Fully offline."""

from __future__ import annotations

import json


from itsbob.agent.brain import TierResult
from itsbob.agent.context import Conversation, Step, Turn, build_messages
from itsbob.agent.loop import Agent, TurnGuard
from itsbob.agent.persona import Persona
from itsbob.agent.writer import MemoryWriter
from itsbob.llm.base import AllProvidersFailed, LLMResponse, Usage
from itsbob.memory.base import MemoryRecord
from itsbob.memory.long_term import LongTermMemory
from itsbob.router.gatekeeper import Gatekeeper
from itsbob.router.tiers import Tier
from itsbob.tools import Mode, build_toolbox


class FakeBrain:
    """Replays a script of JSON replies, recording what it was asked."""

    def __init__(self, script, *, local=None):
        self.script = list(script)
        self.local = local
        self.requests = []
        self.tiers = []

    def _next(self, tier):
        self.tiers.append(tier)
        if not self.script:
            return {"final": "(script exhausted)"}
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, tier, request, *, purpose="", escalate=True):
        self.requests.append(request)
        payload = self._next(tier)
        return TierResult(response=_response(json.dumps(payload)), tier=tier)

    def complete_json(self, tier, request, *, purpose="", default=None):
        self.requests.append(request)
        payload = self._next(tier)
        return payload, TierResult(response=_response(json.dumps(payload)), tier=tier)


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text, model="fake-1", provider="fake", usage=Usage(10, 5), latency_ms=1.0
    )


def _agent(tmp_path, script, *, memory=None, mode=Mode.TRUSTED, **kwargs):
    toolbox = build_toolbox(memory=memory, workspace=tmp_path / "ws", mode=mode, env={})
    return Agent(
        brain=FakeBrain(script),
        toolbox=toolbox,
        memory=memory,
        persona=Persona(name="Test"),
        writer=None,
        # No classifier model: the rule-based path is deterministic, so tier
        # assertions test the heuristic rather than a scripted reply, and the
        # script is consumed only by the loop itself.
        gatekeeper=Gatekeeper(),
        **kwargs,
    )


# -- the happy path --------------------------------------------------------


def test_a_tool_call_then_an_answer(tmp_path):
    agent = _agent(
        tmp_path,
        [
            {"thought": "writing it", "tool": "write_file", "params": {"path": "a.txt", "content": "hi"}},
            {"thought": "done", "final": "Created a.txt."},
        ],
    )
    turn = agent.chat("make a.txt saying hi")
    assert turn.final == "Created a.txt."
    assert turn.tools_used == ["write_file"]
    assert (tmp_path / "ws" / "a.txt").read_text() == "hi"


def test_answering_with_no_tool_is_one_step(tmp_path):
    agent = _agent(tmp_path, [{"final": "Hello."}])
    turn = agent.chat("hi")
    assert turn.final == "Hello." and len(turn.steps) == 1 and turn.tools_used == []


def test_the_turn_is_added_to_the_conversation(tmp_path):
    agent = _agent(tmp_path, [{"final": "one"}, {"final": "two"}])
    agent.chat("a")
    agent.chat("b")
    assert len(agent.conversation) == 2
    assert [t.final for t in agent.conversation.turns] == ["one", "two"]


def test_tokens_are_accumulated(tmp_path):
    agent = _agent(tmp_path, [{"final": "x"}])
    assert agent.chat("hi").tokens == 15


# -- the guard -------------------------------------------------------------


def test_an_identical_call_is_replayed_not_re_executed(tmp_path):
    """The safety property: a looping model must not apply a change twice."""
    call = {"tool": "write_file", "params": {"path": "n.txt", "content": "x", "append": True}}
    agent = _agent(tmp_path, [call, call, {"final": "done"}])
    agent.chat("append x")
    # Appended once despite being requested twice.
    assert (tmp_path / "ws" / "n.txt").read_text() == "x"


def test_a_replayed_call_is_told_it_already_ran(tmp_path):
    call = {"tool": "list_dir", "params": {}}
    agent = _agent(tmp_path, [call, call, {"final": "done"}])
    turn = agent.chat("list")
    assert "already ran earlier in this turn" in turn.steps[1].observation


def test_repeated_calls_end_the_turn(tmp_path):
    call = {"tool": "list_dir", "params": {}}
    agent = _agent(tmp_path, [call] * 6 + [{"final": "never reached"}], max_steps=8)
    turn = agent.chat("list")
    assert len(turn.steps) < 6


def test_invented_tool_names_end_the_turn(tmp_path):
    agent = _agent(
        tmp_path,
        [{"tool": f"not_a_tool_{i}", "params": {}} for i in range(6)] + [{"final": "x"}],
        max_steps=8,
    )
    turn = agent.chat("do something")
    assert len(turn.steps) == 3
    assert all(not s.ok for s in turn.steps)


def test_consecutive_failures_end_the_turn(tmp_path):
    agent = _agent(
        tmp_path,
        [{"tool": "read_file", "params": {"path": f"missing{i}.txt"}} for i in range(5)]
        + [{"final": "x"}],
        max_steps=8,
    )
    assert len(agent.chat("read things").steps) == 3


def test_a_successful_call_resets_the_failure_count():
    guard = TurnGuard()

    class R:
        def __init__(self, ok):
            self.ok = ok
            self.error = None if ok else "boom"

    guard.record("t", {"i": 1}, R(False))
    guard.record("t", {"i": 2}, R(False))
    guard.record("t", {"i": 3}, R(True))
    assert guard.consecutive_failures == 0
    assert guard.give_up_reason is None


def test_guard_key_is_order_independent():
    assert TurnGuard.key("t", {"a": 1, "b": 2}) == TurnGuard.key("t", {"b": 2, "a": 1})


# -- budgets and endings ---------------------------------------------------


def test_running_out_of_steps_still_answers(tmp_path):
    varied = [{"tool": "read_file", "params": {"path": f"f{i}.txt"}} for i in range(20)]
    agent = _agent(tmp_path, varied + [{"final": "forced"}], max_steps=3)
    turn = agent.chat("read everything")
    assert turn.final  # never silence


def test_a_reply_with_neither_tool_nor_answer_raises_the_tier(tmp_path):
    agent = _agent(tmp_path, [{"thought": ""}, {"final": "recovered"}])
    turn = agent.chat("hi")
    assert turn.final == "recovered"
    assert agent.brain.tiers[1].rank > agent.brain.tiers[0].rank


def test_total_model_failure_is_reported_not_raised(tmp_path):
    agent = _agent(tmp_path, [AllProvidersFailed({"google": RuntimeError("down")})])
    turn = agent.chat("hi")
    assert "failed" in turn.final.lower()
    assert turn.error


def test_a_declined_confirmation_stops_the_turn(tmp_path):
    toolbox = build_toolbox(workspace=tmp_path / "ws", mode=Mode.GUARDED, confirm=lambda *a: False, env={})
    agent = Agent(
        brain=FakeBrain(
            [{"tool": "run_shell", "params": {"command": "echo x"}}, {"final": "should not reach"}]
        ),
        toolbox=toolbox,
        memory=None,
        writer=None,
        gatekeeper=Gatekeeper(),
    )
    turn = agent.chat("run it")
    assert "declined" in turn.final
    assert len(turn.steps) == 1  # stopped, did not look for another route


def test_a_policy_denial_is_an_observation_not_an_abort(tmp_path):
    """Refusal is information the agent gets to respond to."""
    agent = _agent(
        tmp_path,
        [
            {"tool": "run_shell", "params": {"command": "echo x"}},
            {"final": "I can't run commands unattended."},
        ],
        mode=Mode.GUARDED,
    )
    turn = agent.chat("run it")
    assert turn.final == "I can't run commands unattended."
    assert not turn.steps[0].ok


# -- tier selection --------------------------------------------------------


def test_the_tier_never_drops_within_a_turn(tmp_path):
    agent = _agent(tmp_path, [{"thought": ""}, {"thought": ""}, {"final": "x"}])
    agent.chat("delete everything and tell me if that was wise")
    ranks = [t.rank for t in agent.brain.tiers]
    assert ranks == sorted(ranks)


def test_a_destructive_request_starts_at_the_strongest_tier(tmp_path):
    agent = _agent(tmp_path, [{"final": "no"}])
    agent.chat("delete all the backups in /var")
    assert agent.brain.tiers[0] is Tier.S


def test_small_talk_starts_cheap(tmp_path):
    agent = _agent(tmp_path, [{"final": "hi"}])
    agent.chat("hello there")
    assert agent.brain.tiers[0] is Tier.C


# -- memory ----------------------------------------------------------------


def test_relevant_memory_reaches_the_prompt(tmp_path):
    store = LongTermMemory(":memory:", embedder=None)
    store.add(MemoryRecord(content="James takes his coffee black"))
    agent = _agent(tmp_path, [{"final": "black"}], memory=store)
    agent.chat("how do I take my coffee?")
    assert "coffee black" in agent.brain.requests[0].messages[0].content


def test_a_broken_memory_store_does_not_break_the_turn(tmp_path):
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("disk gone")

    agent = _agent(tmp_path, [{"final": "still works"}], memory=Broken())
    assert agent.chat("hello").final == "still works"


def test_the_writer_stores_extracted_facts(tmp_path):
    store = LongTermMemory(":memory:", embedder=None)
    brain = FakeBrain([{"memories": [{"content": "James lives in Reading", "importance": 0.7}]}])
    writer = MemoryWriter(brain=brain, store=store)
    written = writer.write(message="I moved to Reading last year, it's fine", answer="Noted.")
    assert [r.content for r in written] == ["James lives in Reading"]


def test_what_the_agent_remembered_is_hidden_from_the_writer(tmp_path):
    """Otherwise one statement becomes three near-identical rows."""
    store = LongTermMemory(":memory:", embedder=None)
    toolbox = build_toolbox(memory=store, workspace=tmp_path / "ws", mode=Mode.TRUSTED, env={})
    brain = FakeBrain(
        [
            {"tool": "remember", "params": {"content": "James keeps SSH keys in ~/.ssh/work"}},
            {"final": "Noted."},
            {"memories": []},  # what the writer is asked
        ]
    )
    agent = Agent(brain=brain, toolbox=toolbox, memory=store, gatekeeper=Gatekeeper())
    agent.writer = MemoryWriter(brain=brain, store=store)
    agent.chat("I keep my SSH keys in ~/.ssh/work, remember that please")
    # The writer's prompt must list what was just stored as already known.
    assert "~/.ssh/work" in brain.requests[-1].messages[-1].content
    assert len(store) == 1


def test_the_writer_skips_a_duplicate(tmp_path):
    store = LongTermMemory(":memory:", embedder=None)
    store.add(MemoryRecord(content="James lives in Reading"))
    brain = FakeBrain([{"memories": [{"content": "James lives in Reading"}]}])
    writer = MemoryWriter(brain=brain, store=store)
    assert writer.write(message="a message long enough to be extracted from", answer="ok") == []
    assert len(store) == 1


def test_the_writer_ignores_short_turns(tmp_path):
    store = LongTermMemory(":memory:", embedder=None)
    brain = FakeBrain([{"memories": [{"content": "should not be written"}]}])
    assert MemoryWriter(brain=brain, store=store).write(message="hi", answer="hello") == []


def test_a_failing_writer_never_breaks_the_turn(tmp_path):
    store = LongTermMemory(":memory:", embedder=None)
    brain = FakeBrain([AllProvidersFailed({"x": RuntimeError("down")})])
    writer = MemoryWriter(brain=brain, store=store)
    assert writer.write(message="a long enough message to extract from", answer="ok") == []
    assert writer.errors == 1


# -- prompt shape ----------------------------------------------------------


def _messages(steps=(), **kwargs):
    return build_messages(
        persona=Persona(),
        tools="- read_file(path: string) — read",
        snapshot_text="do the thing",
        conversation=Conversation(),
        steps=steps,
        tool_names=["read_file"],
        **kwargs,
    )


def test_all_static_instruction_is_in_one_system_message():
    """Providers differ on non-leading system messages; there is only one."""
    messages = _messages()
    assert sum(1 for m in messages if m.role == "system") == 1
    assert messages[0].role == "system"


def test_the_output_contract_and_roster_are_in_the_system_message():
    body = _messages()[0].content
    assert '"thought"' in body and "read_file" in body


def test_steps_are_encoded_as_assistant_and_user_turns():
    """The fix for the loop: history the model reads as its own."""
    step = Step(index=1, thought="looking", tool="read_file", params={"path": "a"}, observation="1 line")
    messages = _messages(steps=[step])
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert json.loads(messages[2].content)["tool"] == "read_file"
    assert "Result of read_file" in messages[3].content


def test_memory_ids_are_shown_so_they_can_be_corrected():
    record = MemoryRecord(content="a stale fact")
    body = _messages(memories=[record])[0].content
    assert record.id[:8] in body


def test_older_turns_are_summarized_not_dropped():
    conversation = Conversation(window=2)
    for i in range(5):
        conversation.add(Turn(message=f"question {i}", final=f"answer {i}"))
    summary = conversation.summary_of_older()
    assert "question 0" in summary and "question 2" in summary
    assert len(conversation.as_messages()) == 4  # only the window, verbatim


def test_context_merges_behind_the_message(tmp_path):
    agent = _agent(tmp_path, [{"final": "ok"}])
    snapshot = agent._snapshot("current thing", {"facts": {"project": "itsbob"}})
    assert snapshot.facts["project"] == "itsbob"
    assert "current thing" in snapshot.text


def test_the_message_wins_over_context_on_overlap(tmp_path):
    agent = _agent(tmp_path, [{"final": "ok"}])
    snapshot = agent._snapshot(
        json.dumps({"facts": {"env": "prod"}}), {"facts": {"env": "dev", "region": "eu"}}
    )
    assert snapshot.facts == {"env": "prod", "region": "eu"}


# -- events ----------------------------------------------------------------


def test_events_are_emitted_for_each_phase(tmp_path):
    agent = _agent(
        tmp_path,
        [{"tool": "list_dir", "params": {}}, {"final": "done"}],
    )
    seen = []
    agent.chat("list it", on_event=lambda e: seen.append(e.kind))
    assert seen[0] == "classified"
    assert "tool" in seen and "observation" in seen and seen[-1] == "final"


def test_a_broken_event_listener_does_not_break_the_turn(tmp_path):
    def explode(event):
        raise RuntimeError("bad listener")

    agent = _agent(tmp_path, [{"final": "fine"}])
    assert agent.chat("hi", on_event=explode).final == "fine"


# -- token discipline: budget, feasibility, local-first ---------------------


def test_the_step_budget_extends_itself_while_work_is_landing(tmp_path):
    """A fixed budget was stopping real work mid-task. Progress buys more steps."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    # Reads a different file each step, so every step succeeds and none repeats.
    for i in range(12):
        (workspace / f"f{i}.txt").write_text(f"file {i}", encoding="utf-8")

    script = [
        {"thought": "reading", "tool": "read_file", "params": {"path": f"f{i}.txt"}}
        for i in range(12)
    ] + [{"final": "done"}]
    agent = _agent(tmp_path, script, max_steps=3, extend_by=3, hard_max_steps=15)
    turn = agent.chat("read every file in the workspace and tell me what they say")
    assert turn.extensions >= 2
    assert len(turn.steps) > 3  # the original budget did not end it


def test_a_turn_going_nowhere_still_stops_at_the_first_checkpoint(tmp_path):
    """Extension is earned by successful tool calls, not granted by default."""
    script = [
        {"thought": "hmm", "tool": "read_file", "params": {"path": "missing.txt"}}
    ] * 20 + [{"final": "gave up"}]
    agent = _agent(tmp_path, script, max_steps=6, extend_by=6, hard_max_steps=30)
    turn = agent.chat("read a file that is not there")
    assert turn.extensions == 0
    assert len(turn.steps) <= 6


def test_the_feasibility_check_refuses_before_spending_the_budget():
    """One cheap call beats eight premium steps discovering the same thing."""
    from itsbob.agent.budget import FeasibilityCheck

    class Refusing:
        def complete_json(self, tier, request, **kwargs):
            return {"feasible": False, "reason": "there is no printer tool",
                    "missing": ["a printer"]}, None

    check = FeasibilityCheck(brain=Refusing())
    verdict = check.check(message="print this on paper", tools=["read_file"])
    assert not verdict.feasible and verdict.checked
    assert "printer" in verdict.explain()


def test_an_unusable_feasibility_answer_lets_the_turn_proceed():
    """A false 'no' refuses work it could have done — worse than a wasted turn."""
    from itsbob.agent.budget import FeasibilityCheck

    class Broken:
        def complete_json(self, tier, request, **kwargs):
            raise RuntimeError("no model")

    assert FeasibilityCheck(brain=Broken()).check(message="x", tools=[]).feasible

    class Vague:
        def complete_json(self, tier, request, **kwargs):
            return {"feasible": False}, None  # refused, but said nothing useful

    assert FeasibilityCheck(brain=Vague()).check(message="x", tools=[]).feasible


def test_the_spend_guard_stops_a_turn_that_has_cost_too_much():
    from itsbob.agent.budget import SpendGuard

    guard = SpendGuard(max_tokens_per_turn=100)
    guard.start_turn()
    guard.add(60)
    assert guard.exceeded() is None
    guard.add(60)
    assert "over its" in (guard.exceeded() or "")
    guard.start_turn()  # a new turn starts clean
    assert guard.exceeded() is None


def test_cheap_work_goes_to_the_local_model_and_is_counted():
    """'Configured' and 'answered' are different claims; only one saves money."""
    from itsbob.agent.brain import TieredBrain
    from itsbob.llm.base import LLMRequest, LLMResponse, Usage, user
    from itsbob.router.tiers import Tier

    class FakeLocal:
        name = "ollama"
        models = ("qwen2.5:1.5b",)
        calls = 0

        def complete_with_fallback(self, request, preferred_model=None):
            type(self).calls += 1
            return LLMResponse(text="hi", model="qwen2.5:1.5b", provider="ollama",
                               usage=Usage(prompt_tokens=5, completion_tokens=2))

    class NeverAsked:
        def complete(self, request, *, purpose=""):
            raise AssertionError("the cloud tier should not have been reached")

        def describe(self):
            return []

    brain = TieredBrain({Tier.A: NeverAsked()}, local=FakeLocal())
    request = LLMRequest(messages=[user("hello")])
    assert brain.complete(Tier.C, request).response.provider == "ollama"
    # A bookkeeping chore takes the local path even at a higher tier.
    chore = LLMRequest(messages=[user("classify")], metadata={"local_ok": True})
    assert brain.complete(Tier.A, chore).response.provider == "ollama"
    assert brain.local_answers == 2 and brain.local_share == 1.0
    assert brain.describe()["local"]["answers"] == 2


# -- memory attribution ----------------------------------------------------


def test_bobs_own_opinions_are_stored_as_bobs(tmp_path):
    """Asked for its favourite films, it wrote all five down as the user's."""
    from itsbob.memory.base import Subject

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    brain = FakeBrain(
        [
            {
                "memories": [
                    {"content": "I liked Blade Runner most", "subject": "bob",
                     "kind": "preference", "horizon": "long"},
                    {"content": "The user is looking for something to watch tonight",
                     "subject": "user", "horizon": "short"},
                ]
            }
        ]
    )
    written = MemoryWriter(brain=brain, store=store).write(
        message="what are your five favourite films?",
        answer="Blade Runner, Alien, Heat, Chungking Express, Stalker.",
    )
    by_subject = {r.subject: r for r in written}
    assert by_subject[Subject.SELF].content.startswith("I liked")
    assert by_subject[Subject.USER].expires_at is not None  # short-term expires


def test_a_self_labelled_sentence_wins_over_a_wrong_subject_label():
    """A model that writes 'I liked X' and labels it `user` contradicted itself."""
    from itsbob.agent.writer import MemoryWriter
    from itsbob.memory.base import Subject

    brain = FakeBrain(
        [{"memories": [{"content": "I liked Blade Runner", "subject": "user"}]}]
    )

    class Store:
        def search(self, *a, **k):
            return []

    extracted = MemoryWriter(brain=brain, store=Store()).extract(
        message="what did you think of it?", answer="Blade Runner was the best."
    )
    assert extracted[0].subject is Subject.SELF


def test_recalled_memories_are_grouped_by_who_they_are_about():
    from itsbob.memory.base import MemoryRecord, Subject

    rendered = build_messages(
        persona=Persona(name="Bob"),
        tools="",
        snapshot_text="hi",
        conversation=Conversation(),
        memories=[
            MemoryRecord(content="prefers dark roast", subject=Subject.USER),
            MemoryRecord(content="I liked Blade Runner", subject=Subject.SELF),
        ],
    )[0].content
    assert "About the user:" in rendered
    assert "do not attribute these to the user" in rendered


def test_short_term_memory_is_pruned_by_count_and_by_clock(tmp_path):
    import time

    from itsbob.memory.base import Horizon, MemoryRecord

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    store.short_term_capacity = 3
    for i in range(6):
        store.add(MemoryRecord(content=f"working on step {i}", horizon=Horizon.SHORT))
    # Long-horizon, and so untouchable by the working set's limits.
    store.add(MemoryRecord(content="lives in Reading", horizon=Horizon.LONG))
    assert store.prune_short_term() == 3
    assert len(store) == 4
    assert store.counts_by("horizon") == {"short": 3, "long": 1}

    # Old enough to expire regardless of how few there are.
    store.short_term_ttl_seconds = 0.0
    assert store.prune_short_term() == 3
    assert store.counts_by("horizon") == {"long": 1}

    # And a promoted memory survives both limits.
    stale = store.add(MemoryRecord(content="this thread matters", horizon=Horizon.SHORT))
    assert store.promote(stale.id)
    assert store.prune_short_term(keep=0) == 0
    assert store.get(stale.id).expires_at is None
    assert time.time() > 0  # (the clock is only used for the TTL above)


# -- what a turn actually costs --------------------------------------------


def test_later_steps_send_a_smaller_prompt_than_the_first(tmp_path):
    """Measured, not asserted by construction: the descriptions stop being
    re-sent once choosing is done, which was ~1,900 tokens a step at 37 tools."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        (workspace / f"f{i}.txt").write_text("x" * 200, encoding="utf-8")

    script = [
        {"thought": "reading", "tool": "read_file", "params": {"path": f"f{i}.txt"}}
        for i in range(6)
    ] + [{"final": "done"}]
    agent = _agent(tmp_path, script, max_steps=10)
    agent.chat("read all six files in the workspace and summarise them")

    # Only the step requests: the feasibility screen and the gatekeeper make
    # their own calls, and neither carries a tool list.
    systems = [
        r.messages[0].content
        for r in agent.brain.requests
        if "- read_file(" in r.messages[0].content
    ]
    assert len(systems) >= 3
    # The prompt grows with the scratchpad, so compare the fixed part instead:
    # the system message, which is where the tool list lives.
    assert len(systems[1]) < len(systems[0]) * 0.65
    # And nothing became uncallable.
    for name in agent.toolbox.registry.names():
        assert name in systems[1], f"{name} disappeared from the shortened prompt"


def test_private_scratchpad_is_carried_between_steps_and_bounded(tmp_path):
    agent = _agent(
        tmp_path,
        [
            {"thought": "plan", "scratchpad": "keep this plan", "tool": "read_file", "params": {"path": "a.txt"}},
            {"final": "done"},
        ],
    )
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "ws" / "a.txt").write_text("hello", encoding="utf-8")
    turn = agent.chat("read a.txt")
    assert turn.scratchpad == "keep this plan"
    assert any("Private scratchpad" in m.content for r in agent.brain.requests[1:] for m in r.messages)


def test_a_tool_already_in_play_keeps_its_description(tmp_path):
    """What it is still reasoning about stays legible; the rest is signatures."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "a.txt").write_text("hello", encoding="utf-8")

    agent = _agent(
        tmp_path,
        [
            {"thought": "read it", "tool": "read_file", "params": {"path": "a.txt"}},
            {"thought": "again", "tool": "read_file", "params": {"path": "a.txt"}},
            {"final": "done"},
        ],
    )
    agent.chat("read a.txt and tell me what is in it please")
    steps = [
        r.messages[0].content
        for r in agent.brain.requests
        if "- read_file(" in r.messages[0].content
    ]
    second = steps[1]
    assert "- read_file(" in second
    read_line = next(x for x in second.splitlines() if x.startswith("- read_file("))
    assert "—" in read_line  # kept its prose: it is the tool in play
    other = next(x for x in second.splitlines() if x.startswith("- check_network("))
    assert "—" not in other  # signature only
    # Memory tools keep their prose throughout: they are reached for late.
    remember = next(x for x in second.splitlines() if x.startswith("- remember("))
    assert "—" in remember


# -- paying out a list that was announced ----------------------------------


def test_an_answer_that_names_a_count_is_asked_to_pay_it_out(tmp_path):
    """The exact reported failure: '10 matches were played', then two named."""
    from itsbob.integrations.shaping import shape

    shaped = shape("football", "matches", {
        "competition": {"name": "Premier League"},
        "matches": [
            {"homeTeam": {"name": f"Home {i} FC"}, "awayTeam": {"name": f"Away {i} FC"},
             "score": {"fullTime": {"home": i % 3, "away": 1}}, "status": "FINISHED"}
            for i in range(10)
        ],
    })

    class Fixed:
        """A tool that hands back a full list, whatever it is asked."""

        name = "list_things"
        description = "Return ten things."
        risk = __import__("itsbob.tools.base", fromlist=["Risk"]).Risk.READ
        mutates = False
        parameters = {"type": "object", "properties": {}}

    from itsbob.tools.base import Risk, Tool, ToolResult

    tool = Tool(name="list_things", description="Return ten things.",
                run=lambda p, c: ToolResult(ok=True, output=shaped), risk=Risk.READ)
    toolbox = build_toolbox(workspace=tmp_path / "ws", mode=Mode.TRUSTED, env={},
                            extra_tools=[tool])
    brain = FakeBrain([
        {"thought": "fetch", "tool": "list_things", "params": {}},
        {"final": "10 matches were played, including Home 0 and Home 1, among others."},
        {"final": "All ten:\n" + "\n".join(f"- Home {i} beat Away {i}" for i in range(10))},
    ])
    agent = Agent(brain=brain, toolbox=toolbox, memory=None, persona=Persona(name="Test"),
                  writer=None, gatekeeper=Gatekeeper())
    turn = agent.chat("give me every match result from the weekend please")

    assert turn.rewritten_for_completeness
    assert turn.final.startswith("All ten:")
    for i in range(10):
        assert f"Home {i}" in turn.final


def test_a_complete_answer_costs_no_extra_call(tmp_path):
    """The check is free; only the rewrite costs, and it must stay rare."""
    from itsbob.tools.base import Risk, Tool, ToolResult

    shaped = "4 results, all listed:\n" + "\n".join(f"- row {i}" for i in range(4))
    tool = Tool(name="list_things", description="d",
                run=lambda p, c: ToolResult(ok=True, output=shaped), risk=Risk.READ)
    toolbox = build_toolbox(workspace=tmp_path / "ws", mode=Mode.TRUSTED, env={},
                            extra_tools=[tool])
    complete = "Here they are:\n" + "\n".join(f"- row {i}" for i in range(4))
    brain = FakeBrain([
        {"thought": "fetch", "tool": "list_things", "params": {}},
        {"final": complete},
    ])
    agent = Agent(brain=brain, toolbox=toolbox, memory=None, persona=Persona(name="Test"),
                  writer=None, gatekeeper=Gatekeeper())
    turn = agent.chat("list everything you can find for me right now")
    assert not turn.rewritten_for_completeness
    assert turn.final == complete
    assert brain.script == []  # the third scripted reply was never needed


def test_standing_style_preferences_reach_every_prompt(tmp_path):
    """A rule about formatting is never similar to the query, so recall alone
    would never surface it — it goes straight into the persona instead."""
    from itsbob.memory.base import MemoryRecord

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    store.add(MemoryRecord(content="Always list every match in full — never summarise.",
                           tags=("style",)))
    agent = _agent(tmp_path, [{"final": "ok"}], memory=store)
    agent.chat("what is the weather like")
    # The step prompt, not the memory writer's — that has its own system message.
    system = next(
        r.messages[0].content for r in agent.brain.requests
        if "You are Test" in r.messages[0].content
    )
    assert "Always list every match in full" in system
    assert "How this user wants to be answered" in system or "How to answer" in system


def test_automatic_extraction_never_writes_straight_to_long_term():
    """Whatever the model says, and it says "long" often.

    Extraction happens at the one moment least suited to judging permanence:
    everything just said looks like it matters. So the horizon it proposes is
    ignored outright, and permanence is earned afterwards by being recalled.
    """
    from itsbob.memory.base import Horizon

    store = LongTermMemory(":memory:", embedder=None)
    brain = FakeBrain(
        [
            {
                "memories": [
                    {"content": "the user commutes from Reading", "subject": "user",
                     "horizon": "long", "importance": 0.9},
                    {"content": "the user prefers tea", "subject": "user",
                     "horizon": "permanent", "importance": 0.99},
                ]
            }
        ]
    )
    written = MemoryWriter(brain=brain, store=store).write(
        message="I commute from Reading and I only drink tea", answer="Noted."
    )

    assert len(written) == 2
    for record in written:
        assert record.horizon is Horizon.SHORT
        assert record.expires_at is not None, "a short memory with no clock never expires"
    assert store.counts_by("horizon") == {"short": 2}


def test_an_explicit_remember_may_still_write_long_term(tmp_path):
    """The user asking for something to be kept is a different act entirely."""
    from itsbob.memory.base import Horizon

    store = LongTermMemory(":memory:", embedder=None)
    toolbox = build_toolbox(memory=store, workspace=tmp_path / "ws", mode=Mode.TRUSTED, env={})
    toolbox.call(
        "remember", content="the wifi password is on the router", horizon="long"
    )
    kept = store.all()[0]
    assert kept.horizon is Horizon.LONG
    assert kept.expires_at is None


# -- the expensive turn that did not cost anything --------------------------


def _fixed_tier(tier):
    """A gatekeeper that always lands on one tier, so the test is about routing."""
    from itsbob.router.tiers import GateDecision

    class Fixed:
        def classify(self, snapshot):
            return GateDecision(tier=tier, fingerprint="test", source="test")

    return Fixed()


def test_a_hard_thinking_turn_is_answered_free_and_never_reaches_the_tier(tmp_path):
    from itsbob.agent.delegation import DelegatePolicy
    from itsbob.integrations.delegate import Delegation

    class Free:
        def ask(self, question, context=""):
            return Delegation(
                question=question,
                answer="Renting keeps you liquid; buying builds equity. Over ten years...",
                ok=True,
                source="deepseek",
            )

    hard = (
        "Talk me through the trade-offs between renting and buying somewhere to "
        "live over a ten year horizon, assuming rates stay roughly where they "
        "are and I might move cities once in that time."
    )
    agent = _agent(tmp_path, [{"final": "the paid tier answered"}])
    agent.delegation = DelegatePolicy(delegate=Free())
    agent.gatekeeper = _fixed_tier(Tier.S)

    turn = agent.chat(hard)
    assert turn.final.startswith("Renting keeps you liquid")
    assert turn.delegated is True
    # The point of the exercise: no step ran, so no premium tokens were spent.
    assert turn.steps == [] and turn.tokens == 0


def test_when_the_free_path_fails_the_turn_runs_exactly_as_it_would_have(tmp_path):
    """The failure mode is "you paid for the answer", never a worse one."""
    from itsbob.agent.delegation import DelegatePolicy

    class Broken:
        def ask(self, question, context=""):
            raise RuntimeError("the site is asking for a login")

    hard = (
        "Talk me through the trade-offs between renting and buying somewhere to "
        "live over a ten year horizon, assuming rates stay roughly where they "
        "are and I might move cities once in that time."
    )
    agent = _agent(tmp_path, [{"final": "the paid tier answered"}])
    agent.delegation = DelegatePolicy(delegate=Broken())
    agent.gatekeeper = _fixed_tier(Tier.S)

    turn = agent.chat(hard)
    assert turn.final == "the paid tier answered"
    assert turn.delegated is False
    assert len(turn.steps) == 1


def test_a_turn_that_needs_a_tool_is_never_handed_out(tmp_path):
    from itsbob.agent.delegation import DelegatePolicy

    class NeverCalled:
        def ask(self, question, context=""):
            raise AssertionError("sent a tool question to something with no tools")

    agent = _agent(tmp_path, [{"final": "read it myself"}])
    agent.delegation = DelegatePolicy(delegate=NeverCalled())
    agent.gatekeeper = _fixed_tier(Tier.S)

    turn = agent.chat(
        "Read through the build log in the workspace and work out which step "
        "first failed, then tell me what the underlying cause actually was."
    )
    assert turn.final == "read it myself"
    assert turn.delegated is False
