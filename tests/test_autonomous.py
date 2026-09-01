"""Continuous mode: scheduled work and live chat sharing one agent."""

from __future__ import annotations

import time


from itsbob.agent.context import Turn
from itsbob.daemon.tasks import TaskStore
from itsbob.gui.autonomous import Autonomous
from itsbob.gui.session import Session


class _Agent:
    def __init__(self, delay: float = 0.05, boom: bool = False):
        self.delay = delay
        self.boom = boom
        self.seen: list[str] = []
        self.conversation = None

    def chat(self, message, *, on_event=None, context=None):
        self.seen.append(message)
        time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("model down")
        return Turn(message=message, final=f"did: {message}")


class _NeverNotify:
    def judge(self, **kwargs):
        return None


def _runner(agent, store, **kwargs):
    session = Session(lambda _confirm: agent)
    session.emit = lambda *a, **k: None
    # The scheduling tests must not depend on the audit machine's battery,
    # temperature or free disk. Health-gate behaviour is covered explicitly.
    kwargs.setdefault("health_gate", False)
    return session, Autonomous(session, store, gate=_NeverNotify(), poll_seconds=0.05, **kwargs)


def test_due_tasks_are_run_and_recorded():
    store = TaskStore(":memory:")
    agent = _Agent()
    session, runner = _runner(agent, store)
    now = time.time()
    store.create("watch", "check the disk", "every 15m", now=now)

    runner._poll(now=now)
    time.sleep(0.6)
    task = store.find("watch")
    assert agent.seen == ["check the disk"]
    assert task.run_count == 1 and task.last_status == "ok"
    assert runner.runs == 1


def test_a_failed_task_is_recorded_as_failed():
    store = TaskStore(":memory:")
    session, runner = _runner(_Agent(boom=True), store)
    now = time.time()
    store.create("bad", "explode", "every 15m", now=now)
    runner._poll(now=now)
    time.sleep(0.6)
    assert store.find("bad").last_status == "failed"


def test_a_task_already_running_is_not_submitted_twice():
    """Without this a slow task is resubmitted on every poll while it runs."""
    store = TaskStore(":memory:")
    agent = _Agent(delay=0.5)
    session, runner = _runner(agent, store)
    now = time.time()
    store.create("slow", "take a while", "every 15m", now=now)
    runner._poll(now=now)
    runner._poll(now=now)
    runner._poll(now=now)
    time.sleep(1.0)
    assert agent.seen.count("take a while") == 1


def test_a_typed_message_goes_ahead_of_queued_tasks():
    store = TaskStore(":memory:")
    agent = _Agent(delay=0.3)
    session, runner = _runner(agent, store)
    now = time.time()
    for name in ("one", "two", "three"):
        store.create(name, f"task {name}", "every 15m", now=now)
    runner._poll(now=now)
    time.sleep(0.05)
    session.submit("my question", source="user")
    queued = [q["text"] for q in session.queued_messages()]
    assert queued[0] == "my question", queued


def test_work_is_deferred_not_failed_when_the_machine_is_unfit(monkeypatch):
    """A flat battery says nothing about whether the task works, so counting it
    as a failure would disable a good task after five bad mornings."""
    import itsbob.scripts.system_monitor as monitor

    store = TaskStore(":memory:")
    agent = _Agent()
    session, runner = _runner(agent, store, defer_seconds=60, health_gate=True)
    monkeypatch.setattr(
        monitor, "read_system",
        lambda *a, **k: type("S", (), {"concerns": ["on battery at 9%"]})(),
    )
    now = time.time()
    store.create("heavy", "do a big job", "every 15m", now=now)
    runner._poll(now=now)
    time.sleep(0.3)

    task = store.find("heavy")
    assert agent.seen == []
    assert task.fail_count == 0, "a deferral is not a failure"
    assert task.enabled is True
    assert task.next_run > now
    assert runner.deferrals == 1


def test_a_task_can_opt_out_of_the_health_gate(monkeypatch):
    import itsbob.scripts.system_monitor as monitor

    store = TaskStore(":memory:")
    agent = _Agent()
    session, runner = _runner(agent, store, health_gate=True)
    monkeypatch.setattr(
        monitor, "read_system",
        lambda *a, **k: type("S", (), {"concerns": ["overheating"]})(),
    )
    now = time.time()
    store.create("light", "quick check", "every 15m", now=now, metadata={"ignore_health": True})
    runner._poll(now=now)
    time.sleep(0.4)
    assert agent.seen == ["quick check"]


def test_a_broken_monitor_does_not_stop_the_work(monkeypatch):
    import itsbob.scripts.system_monitor as monitor

    def explode(*args, **kwargs):
        raise RuntimeError("no /proc")

    store = TaskStore(":memory:")
    agent = _Agent()
    session, runner = _runner(agent, store)
    monkeypatch.setattr(monitor, "read_system", explode)
    now = time.time()
    store.create("t", "do it", "every 15m", now=now)
    runner._poll(now=now)
    time.sleep(0.4)
    assert agent.seen == ["do it"]


def test_starting_twice_is_harmless_and_stopping_works():
    store = TaskStore(":memory:")
    session, runner = _runner(_Agent(), store)
    assert runner.start() is True
    assert runner.start() is False
    assert runner.running is True
    assert runner.stop() is True
    time.sleep(0.2)
    assert runner.stop() is False


def test_status_reports_what_it_is_doing():
    store = TaskStore(":memory:")
    session, runner = _runner(_Agent(), store)
    status = runner.status()
    assert set(status) >= {"running", "runs", "deferrals", "next_due", "health_gate"}
