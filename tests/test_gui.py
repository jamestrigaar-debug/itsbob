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


# -- the messages window and the API panel ---------------------------------


def test_the_messages_window_is_its_own_page(client):
    response = client.get("/messages")
    assert response.status_code == 200
    body = response.data
    assert b"/api/messages/stream" in body  # live
    assert b"/api/chat" not in body  # and genuinely separate from the conversation


def test_messages_are_listed_and_can_be_marked_read(client, tmp_path):
    from itsbob.daemon.notify import FileSink, Notification

    sink = FileSink(path=tmp_path / "home" / "notifications.jsonl")
    sink.send(Notification(title="backup finished", body="all good", task="nightly"))
    sink.send(Notification(title="disk is nearly full", body="4%", urgency="high"))

    body = client.get("/api/messages").get_json()
    assert [m["title"] for m in body["messages"]] == [
        "backup finished", "disk is nearly full"
    ]
    assert body["unread"] == 2
    assert client.get("/api/status").get_json()["unread"] == 2

    marked = client.post("/api/messages/read", json={"all": True}).get_json()
    assert marked["unread"] == 0
    assert client.get("/api/messages?unread=1").get_json()["messages"] == []


def test_the_status_names_which_apis_are_live(client, monkeypatch):
    """The API panel exists so a task can be written against a key that is set."""
    body = client.get("/api/status").get_json()
    assert isinstance(body["apis"], list)
    for row in body["apis"]:
        assert {"name", "configured", "key_env", "base_url"} <= set(row)


def test_discord_reports_that_it_is_not_configured_rather_than_failing(client):
    body = client.get("/api/discord").get_json()
    assert body["configured"] is False and "DISCORD_BOT_TOKEN" in body["hint"]
    assert client.post("/api/discord", json={"enabled": True}).status_code == 409


def test_the_page_links_to_the_messages_window(client):
    body = client.get("/").data
    assert b"/messages" in body
    for panel in (b"activity", b"memory", b"tasks", b"tokens", b"messages", b"system"):
        assert b'data-panel="' + panel + b'"' in body


# -- the header must never sit on "connecting…" ----------------------------


def test_the_autonomous_endpoint_does_not_deadlock_against_itself(client):
    """A real hang, reported as a page stuck on "connecting…".

    `autonomous()` needed the task store and the notification sink, both built
    behind the same lock it had already taken. A plain `threading.Lock` is not
    reentrant, so the first request to reach it blocked forever *holding the
    lock* — and every later request touching any other lazily-built subsystem
    blocked behind it. `/api/status` was one of those, so the header never got
    its first answer, while chat and the event stream (which take no such lock)
    carried on working perfectly and made it look like a front-end bug.
    """
    import threading

    for path in ("/api/autonomous", "/api/status", "/api/autonomous"):
        outcome: list[int] = []

        def call(p=path, into=outcome):
            into.append(client.get(p).status_code)

        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        worker.join(timeout=15)
        assert outcome, f"{path} never answered — the lock deadlocked again"
        assert outcome[0] == 200


def test_one_broken_subsystem_does_not_blank_the_whole_panel(client, monkeypatch):
    """Status feeds the header, the chips, the tools and the tasks at once."""
    from itsbob.gui import app as app_module

    def explode():
        raise RuntimeError("the disk fell off")

    problems: dict[str, str] = {}
    assert app_module._safe(problems, "memory", explode, {"fallback": True}) == {
        "fallback": True
    }
    assert "the disk fell off" in problems["memory"]

    body = client.get("/api/status").get_json()
    assert body["problems"] == {}
    # Everything the header needs is present even on a bare install.
    assert {"tiers", "policy", "tools", "memory", "apis", "services"} <= set(body)


def test_status_names_what_you_could_switch_on_not_only_what_is_on(client):
    """The catalog holds only APIs whose key is set, so it cannot show the rest."""
    body = client.get("/api/status").get_json()
    names = {s["name"] for s in body["services"]}
    assert {"weather", "news", "gnews", "football"} <= names
    for service in body["services"]:
        assert service["key_env"], "a service with no named variable cannot be enabled"


def test_the_page_bounds_its_requests(client):
    """`fetch` waits forever by default, which is how a hang became a silent one."""
    body = client.get("/").data
    assert b"AbortController" in body
    assert b"status unavailable" in body  # a named failure, not "connecting…"


def test_the_tasks_panel_has_its_own_endpoint(client):
    """It was the only panel reading its data out of /api/status.

    That coupling is why it went dark alongside the header when the status
    endpoint deadlocked: memory, scripts and audit each have their own route
    and kept working. A panel should depend on what it shows and nothing else.
    """
    created = client.post(
        "/api/task", json={"name": "briefing", "prompt": "brief me", "schedule": "daily at 07:00"}
    ).get_json()
    body = client.get("/api/tasks").get_json()
    assert [t["name"] for t in body["tasks"]] == ["briefing"]
    assert body["next_due"]
    # And it says whether anything will actually run them, which is the most
    # common "my task never fired".
    assert body["runner"]["autonomous"] is False
    assert client.post("/api/task/remove", json={"id": created["task"]["id"]}).get_json()["ok"]
    assert client.get("/api/tasks").get_json()["tasks"] == []


def test_the_page_loads_tasks_from_the_tasks_route(client):
    body = client.get("/").data
    assert b'api("/api/tasks")' in body
    # A named failure in place, not a silently empty panel.
    assert b"Could not load this panel" in body


# -- the console: token tracking, serving status, network policy -----------


def test_the_token_panel_reports_money_not_only_counts(client):
    """A million cheap tokens and a million premium ones differ ~40x in price,
    so a bare token count invites exactly the wrong conclusion."""
    body = client.get("/api/tokens").get_json()
    assert set(body) == {"today", "all_time", "recent"}
    for key in ("usd", "tokens", "calls", "local_share", "by_model", "by_purpose"):
        assert key in body["today"], key


def test_spend_is_split_by_what_the_call_was_for():
    from itsbob.llm.pricing import Ledger
    from itsbob.llm.router import UsageRecord, UsageTracker
    from itsbob.llm.base import Usage

    tracker = UsageTracker()
    tracker.record(UsageRecord(provider="google", model="gemini-pro-latest", ok=True,
                               usage=Usage(10_000, 2_000), purpose="agent.step.s"))
    tracker.record(UsageRecord(provider="ollama", model="qwen2.5:1.5b", ok=True,
                               usage=Usage(5_000, 500), purpose="gatekeeper.local"))
    summary = Ledger(tracker).summary()
    assert summary["calls"] == 2
    # Pro: 10k prompt at $1.25/M + 2k completion at $10/M = $0.0325
    assert 0.03 < summary["usd"] < 0.035
    assert summary["local_tokens"] == 5_500
    assert summary["by_purpose"]["gatekeeper"]["usd"] == 0.0  # local is free
    assert summary["by_model"]["ollama/qwen2.5:1.5b"]["local"] is True


def test_an_unpriced_model_is_counted_as_unknown_not_guessed():
    """A made-up number next to real ones is indistinguishable from a real one."""
    from itsbob.llm.base import Usage
    from itsbob.llm.pricing import estimate, price_for
    from itsbob.llm.router import UsageRecord

    assert price_for("some-model-nobody-has-heard-of") is None
    got = estimate([UsageRecord(provider="x", model="mystery", ok=True,
                                usage=Usage(1000, 1000))])
    assert got["usd"] == 0.0 and got["unpriced_calls"] == 1
    # And the cheap tier is not billed at the expensive tier's rate.
    assert price_for("gemini-3.5-flash-lite").prompt < price_for("gemini-3.5-flash").prompt


def test_serving_status_tells_stopped_apart_from_never_started(tmp_path):
    import json
    import time

    from itsbob.daemon import daemon_status

    assert daemon_status(tmp_path)["running"] is False
    assert "no daemon has run" in daemon_status(tmp_path)["reason"]

    beat = tmp_path / "daemon.json"
    beat.write_text(json.dumps({"pid": 7, "at": time.time(), "started_at": time.time() - 60}))
    live = daemon_status(tmp_path)
    assert live["running"] is True and live["pid"] == 7

    beat.write_text(json.dumps({"pid": 7, "at": time.time() - 4000}))
    dead = daemon_status(tmp_path)
    assert dead["running"] is False and "it stopped" in dead["reason"]


def test_the_status_says_whether_the_daemon_is_serving(client):
    body = client.get("/api/status").get_json()
    assert "serving" in body and body["serving"]["running"] is False
    assert "spend" in body and "usd" in body["spend"]


def test_the_console_has_every_panel_and_the_status_strip(client):
    body = client.get("/").data.decode()
    for panel in ("activity", "memory", "tasks", "tokens", "messages", "system"):
        assert f'data-panel="{panel}"' in body
    assert "serving:" in body      # is the daemon running
    assert "discord:" in body      # is it reachable in Discord
    assert "today:" in body        # what it has cost
    assert "AbortController" in body  # every request bounded
    # `display:flex` beats the `hidden` attribute; this is what stops a hidden
    # element rendering anyway, which a screenshot caught.
    assert "[hidden]{display:none !important}" in body


def test_recalled_memories_carry_their_attribution_to_the_panel(client):
    """The panel shows whose memory it is; the API has to send it."""
    client.post("/api/memory", json={"content": "prefers dark roast"})
    hit = client.get("/api/memory").get_json()["hits"][0]
    # Short by default: everything starts in the working set.
    assert hit["subject"] == "user" and hit["horizon"] == "short"

    # And can be kept by hand, which is the manual form of being recalled again.
    assert client.post("/api/memory/keep", json={"id": hit["id"]}).get_json()["ok"]
    assert client.get("/api/memory").get_json()["hits"][0]["horizon"] == "long"
    assert client.post("/api/memory/keep", json={"id": "nope"}).status_code == 404
