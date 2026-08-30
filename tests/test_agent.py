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
