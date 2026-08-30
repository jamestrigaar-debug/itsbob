"""The browser interface: streaming, approvals, and the endpoints. Offline."""

from __future__ import annotations

import json
import threading
import time

import pytest

from itsbob.agent.context import Turn

flask = pytest.importorskip("flask")

from itsbob.gui.session import PendingApproval, Session  # noqa: E402


class _Agent:
    """Stands in for the agent; optionally blocks so timing can be tested."""

    def __init__(self, *, delay: float = 0.0, boom: bool = False):
        self.delay = delay
        self.boom = boom
        self.seen: list[str] = []
        self.conversation = None

    def chat(self, message, *, on_event=None, context=None):
        self.seen.append(message)
        if self.delay:
            time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("model exploded")
        if on_event is not None:
            on_event(type("E", (), {"kind": "classified", "data": {"tier": "B"}})())
        return Turn(message=message, final="done")


class _Tool:
    name = "run_shell"
    risk = "execute"


class _Call:
    reason = "counting the lines"


def _drain(session, *, limit=20, timeout=3.0):
    """Collect events from one listener until it goes quiet."""
    collected: list[dict] = []
    stream = session.listen()
    deadline = time.time() + timeout
    for chunk in stream:
        if chunk.startswith(":"):
            continue
        collected.append(json.loads(chunk[6:]))
        if len(collected) >= limit or time.time() > deadline:
            break
    return collected


# -- approvals -------------------------------------------------------------


def test_an_unanswered_approval_is_denied(monkeypatch):
    """Silence must never read as consent — a closed tab is not a yes."""
    import itsbob.gui.session as module

    monkeypatch.setattr(module, "APPROVAL_TIMEOUT", 0.3)
    session = Session(lambda confirm: _Agent())
    assert session.confirm(_Tool(), {"command": "rm -rf x"}, _Call()) is False


def test_an_approved_request_returns_true():
    session = Session(lambda confirm: _Agent())

    def answer():
        for _ in range(50):
            if session.pending:
                session.decide(next(iter(session.pending)), approved=True)
                return
            time.sleep(0.02)

    threading.Thread(target=answer, daemon=True).start()
    assert session.confirm(_Tool(), {"command": "ls"}, _Call()) is True


def test_a_denied_request_returns_false():
    session = Session(lambda confirm: _Agent())

    def answer():
        for _ in range(50):
            if session.pending:
                session.decide(next(iter(session.pending)), approved=False)
                return
            time.sleep(0.02)

    threading.Thread(target=answer, daemon=True).start()
    assert session.confirm(_Tool(), {"command": "ls"}, _Call()) is False


def test_always_allow_skips_the_prompt_next_time():
    session = Session(lambda confirm: _Agent())

    def answer():
        for _ in range(50):
            if session.pending:
                session.decide(next(iter(session.pending)), approved=True, remember=True)
                return
            time.sleep(0.02)

    threading.Thread(target=answer, daemon=True).start()
    assert session.confirm(_Tool(), {"command": "ls"}, _Call()) is True
    assert "run_shell" in session.auto_allow
    # No thread answering this time: it must not block.
    assert session.confirm(_Tool(), {"command": "pwd"}, _Call()) is True


def test_deciding_an_unknown_approval_is_reported():
    assert Session(lambda confirm: _Agent()).decide("nope", approved=True) is False


def test_pending_approvals_are_denied_when_a_turn_ends():
    """A turn that errors must not leave a tool parked forever."""
    session = Session(lambda confirm: _Agent())
    session.pending["x"] = PendingApproval(id="x", tool="t", params={}, risk="execute", reason="")
    assert session.cancel_pending() == 1


# -- turns -----------------------------------------------------------------


def test_turns_run_one_at_a_time_in_order():
    """Two at once would interleave into each other's conversation history."""
    agent = _Agent(delay=0.25)
    session = Session(lambda confirm: agent)
    for message in ("first", "second", "third"):
        assert session.submit(message)["accepted"] is True
    time.sleep(1.6)
    assert agent.seen == ["first", "second", "third"]


def test_a_message_sent_while_busy_is_queued_not_refused():
    session = Session(lambda confirm: _Agent(delay=0.4))
    assert session.submit("first")["started_now"] is True
    time.sleep(0.1)
    second = session.submit("second")
    assert second["accepted"] is True and second["started_now"] is False
    assert [q["text"] for q in session.queued_messages()] == ["second"]
    time.sleep(1.2)
    assert session.queued_messages() == []


def test_the_queue_has_a_ceiling():
    session = Session(lambda confirm: _Agent(delay=5.0))
    results = [session.submit(f"m{i}") for i in range(25)]
    assert results[-1]["accepted"] is False
    assert "already waiting" in results[-1]["error"]


def test_a_typed_message_goes_ahead_of_queued_scheduled_work():
    """Autonomous work is not urgent; making a person wait behind a nightly
    backup summary to ask a question is the wrong way round."""
    agent = _Agent(delay=0.4)
    session = Session(lambda confirm: agent)
    session.submit("running", source="task", label="nightly")
    time.sleep(0.05)
    session.submit("task A", source="task", label="backup")
    session.submit("task B", source="task", label="tidy")
    session.submit("my question", source="user")
    assert [q["text"] for q in session.queued_messages()] == ["my question", "task A", "task B"]


def test_a_running_turn_is_never_preempted():
    """Interrupting mid-tool-call would leave the work half done."""
    agent = _Agent(delay=0.4)
    session = Session(lambda confirm: agent)
    session.submit("already running", source="task")
    time.sleep(0.1)
    session.submit("urgent", source="user")
    time.sleep(1.2)
    assert agent.seen[0] == "already running"


def test_clearing_the_queue_leaves_the_running_turn_alone():
    agent = _Agent(delay=0.4)
    session = Session(lambda confirm: agent)
    session.submit("running")
    time.sleep(0.1)
    session.submit("queued-1")
    session.submit("queued-2")
    assert session.clear_queue() == 2
    time.sleep(0.8)
    assert agent.seen == ["running"]


def test_a_failing_turn_becomes_an_event_not_a_crash():
    session = Session(lambda confirm: _Agent(boom=True))
    events: list[dict] = []
    session.emit = lambda kind, **data: events.append({"kind": kind, **data})
    session.start_turn("go")
    time.sleep(0.5)
    kinds = [e["kind"] for e in events]
    assert "turn_error" in kinds
    assert "model exploded" in next(e for e in events if e["kind"] == "turn_error")["error"]


def test_the_stream_forwards_agent_events():
    session = Session(lambda confirm: _Agent())
    collected: list[dict] = []
    ready = threading.Event()

    def listen():
        for chunk in session.listen():
            if chunk.startswith("data: "):
                event = json.loads(chunk[6:])
                collected.append(event)
                if event["kind"] == "hello":
                    ready.set()
                if event["kind"] == "turn_end":
                    return

    threading.Thread(target=listen, daemon=True).start()
    ready.wait(2)
    session.start_turn("hello")
    time.sleep(0.6)
    kinds = [e["kind"] for e in collected]
    assert kinds[0] == "hello"
    assert "turn_start" in kinds and "classified" in kinds and "turn_end" in kinds


def test_a_slow_listener_does_not_block_the_agent():
    """A backgrounded tab must not be able to stall a turn."""
    import queue

    session = Session(lambda confirm: _Agent())
    stuck: queue.Queue = queue.Queue(maxsize=2)
    session._listeners.append(stuck)
    for i in range(50):
        session.emit("noise", i=i)  # would block forever on a full queue
    assert stuck.full()


# -- endpoints -------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ITSBOB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ITSBOB_EMBED_OFFLINE", "true")
    for key in ("GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    from itsbob.gui.app import create_app

    return create_app(tmp_path / "home").test_client()


def test_the_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"EventSource" in response.data  # the live stream is wired up


def test_favicon_is_served_not_404(client):
    assert client.get("/favicon.ico").status_code == 200


def test_status_describes_every_subsystem(client):
    body = client.get("/api/status").get_json()
    assert set(body) >= {"policy", "tools", "tiers", "memory", "tasks", "busy"}
    assert any(t["name"] == "run_shell" for t in body["tools"])


def test_an_empty_message_is_rejected(client):
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400


def test_approving_something_that_already_timed_out_is_410(client):
    response = client.post("/api/approve", json={"id": "gone", "approved": True})
    assert response.status_code == 410
    assert "no longer waiting" in response.get_json()["error"]


def test_tasks_can_be_created_listed_and_removed(client):
    created = client.post(
        "/api/task", json={"name": "t", "prompt": "do it", "schedule": "every 30m"}
    ).get_json()
    assert created["task"]["schedule"] == "every 30m"
    assert client.get("/api/status").get_json()["tasks"]
    assert client.post("/api/task/remove", json={"id": created["task"]["id"]}).get_json()["ok"]


def test_a_bad_schedule_is_a_400_with_a_hint(client):
    response = client.post("/api/task", json={"name": "t", "prompt": "x", "schedule": "soon"})
    assert response.status_code == 400
    assert "could not parse" in response.get_json()["error"]


def test_an_incomplete_task_is_rejected(client):
    assert client.post("/api/task", json={"name": "t"}).status_code == 400


def test_acting_on_a_missing_task_is_404(client):
    assert client.post("/api/task/remove", json={"id": "ghost"}).status_code == 404


def test_memory_can_be_added_searched_and_forgotten(client):
    added = client.post("/api/memory", json={"content": "the spare key is under the pot"})
    assert added.status_code == 200
    hits = client.get("/api/memory?q=spare key").get_json()["hits"]
    assert hits and "spare key" in hits[0]["content"]
    assert client.post("/api/memory/forget", json={"id": added.get_json()["id"]}).get_json()["ok"]


def test_empty_memory_reports_zero_rather_than_missing(client):
    """LongTermMemory defines __len__, so truthiness would hide it entirely."""
    assert client.get("/api/status").get_json()["memory"]["records"] == 0


def test_reset_clears_the_conversation(client):
    assert client.post("/api/reset").get_json()["ok"] is True
    assert client.get("/api/status").get_json()["turns"] == 0
