"""Schedules, the task store, and the always-on loop. Fully offline."""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from itsbob.agent.context import Turn
from itsbob.daemon.notify import (
    FileSink,
    MultiSink,
    NoticeGate,
    Notification,
)
from itsbob.daemon.schedule import ScheduleError, parse_schedule
from itsbob.daemon.service import Daemon
from itsbob.daemon.tasks import MAX_CONSECUTIVE_FAILURES, TaskStore
from itsbob.tools import Mode, build_toolbox


def _at(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").timestamp()


# -- schedules -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,seconds",
    [("every 30s", 30), ("every 15m", 900), ("every 2 hours", 7200),
     ("every 1 day", 86400), ("hourly", 3600)],
)
def test_intervals_parse(text, seconds):
    assert parse_schedule(text).seconds == seconds


def test_clock_schedule_picks_the_next_occurrence():
    schedule = parse_schedule("daily at 08:30")
    when = schedule.next_after(_at("2026-08-30 14:05"))
    assert datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M") == "2026-08-31 08:30"


def test_a_time_later_today_fires_today():
    when = parse_schedule("daily at 18:00").next_after(_at("2026-08-30 14:05"))
    assert datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M") == "2026-08-30 18:00"


def test_weekdays_skips_the_weekend():
    # 30 Aug 2026 is a Sunday.
    when = parse_schedule("weekdays at 09:00").next_after(_at("2026-08-30 14:05"))
    assert datetime.fromtimestamp(when).weekday() == 0  # Monday


def test_a_named_day_finds_that_day():
    when = parse_schedule("friday at 17:00").next_after(_at("2026-08-30 14:05"))
    assert datetime.fromtimestamp(when).weekday() == 4


def test_a_one_shot_fires_once_then_never():
    schedule = parse_schedule("at 2026-09-01T06:00")
    assert schedule.next_after(_at("2026-08-30 14:05")) == _at("2026-09-01 06:00")
    assert schedule.next_after(_at("2026-09-02 00:00")) is None


def test_an_interval_task_starts_immediately():
    """A task that appears to do nothing after being created reads as broken."""
    now = _at("2026-08-30 14:05")
    assert parse_schedule("every 15m").first_run(now) == now
    assert parse_schedule("daily at 08:30").first_run(now) > now


@pytest.mark.parametrize("text", ["whenever", "", "every 3 fortnights", "daily at 25:00", "at nonsense"])
def test_unparseable_schedules_raise_with_a_hint(text):
    with pytest.raises(ScheduleError):
        parse_schedule(text)


def test_a_too_frequent_interval_is_refused():
    with pytest.raises(ScheduleError, match="minimum"):
        parse_schedule("every 1s")


# -- the store -------------------------------------------------------------


def test_tasks_persist_across_reopen(tmp_path):
    path = tmp_path / "tasks.sqlite"
    store = TaskStore(path)
    store.create("nightly", "do the thing", "daily at 02:00")
    store.close()
    assert [t.name for t in TaskStore(path).all()] == ["nightly"]


def test_creating_with_a_bad_schedule_fails_loudly(tmp_path):
    store = TaskStore(":memory:")
    with pytest.raises(ScheduleError):
        store.create("bad", "x", "sometimes")
    assert len(store) == 0


def test_find_accepts_an_id_or_a_name():
    store = TaskStore(":memory:")
    task = store.create("Inbox Check", "x", "every 15m")
    assert store.find(task.id).id == task.id
    assert store.find("inbox check").id == task.id
    assert store.find("nope") is None


def test_due_returns_only_what_is_ready():
    now = _at("2026-08-30 14:05")
    store = TaskStore(":memory:")
    store.create("now", "x", "every 15m", now=now)
    store.create("later", "x", "daily at 23:00", now=now)
    assert [t.name for t in store.due(now)] == ["now"]


def test_a_disabled_task_is_never_due():
    now = _at("2026-08-30 14:05")
    store = TaskStore(":memory:")
    task = store.create("t", "x", "every 15m", now=now)
    store.set_enabled(task.id, False)
    assert store.due(now) == []


def test_recording_a_run_advances_the_schedule():
    from itsbob.daemon.tasks import TaskRun

    now = _at("2026-08-30 14:05")
    store = TaskStore(":memory:")
    task = store.create("t", "x", "every 15m", now=now)
    store.record_run(task, TaskRun(task.id, now, 10.0, "ok", "fine"), now=now)
    assert store.get(task.id).next_run == pytest.approx(now + 900)
    assert store.get(task.id).run_count == 1


def test_repeated_failures_disable_the_task():
    from itsbob.daemon.tasks import TaskRun

    now = _at("2026-08-30 14:05")
    store = TaskStore(":memory:")
    task = store.create("t", "x", "every 15m", now=now)
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        task = store.record_run(task, TaskRun(task.id, now, 1.0, "failed", "boom"), now=now)
    stored = store.get(task.id)
    assert stored.enabled is False
    assert "consecutive failures" in stored.last_status


def test_one_success_resets_the_failure_streak():
    from itsbob.daemon.tasks import TaskRun

    now = _at("2026-08-30 14:05")
    store = TaskStore(":memory:")
    task = store.create("t", "x", "every 15m", now=now)
    task = store.record_run(task, TaskRun(task.id, now, 1.0, "failed", ""), now=now)
    task = store.record_run(task, TaskRun(task.id, now, 1.0, "ok", ""), now=now)
    assert store.get(task.id).fail_count == 0


def test_max_runs_retires_a_task():
    from itsbob.daemon.tasks import TaskRun

    now = _at("2026-08-30 14:05")
    store = TaskStore(":memory:")
    task = store.create("once-only", "x", "every 15m", max_runs=1, now=now)
    store.record_run(task, TaskRun(task.id, now, 1.0, "ok", ""), now=now)
    assert store.get(task.id).enabled is False


def test_removing_a_task_removes_its_history():
    from itsbob.daemon.tasks import TaskRun

    store = TaskStore(":memory:")
    task = store.create("t", "x", "every 15m")
    store.record_run(task, TaskRun(task.id, time.time(), 1.0, "ok", ""))
    assert store.remove(task.id) is True
    assert store.runs(task.id) == []


# -- notification sinks ----------------------------------------------------


def test_file_sink_appends_jsonl(tmp_path):
    sink = FileSink(path=tmp_path / "n.jsonl")
    sink.send(Notification(title="one", body="a"))
    sink.send(Notification(title="two", body="b"))
    lines = (tmp_path / "n.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["title"] for line in lines] == ["one", "two"]


def test_multi_sink_survives_a_broken_sink(tmp_path):
    class Broken:
        def send(self, notification):
            raise RuntimeError("webhook down")

    good = FileSink(path=tmp_path / "n.jsonl")
    assert MultiSink(sinks=[Broken(), good]).send(Notification(title="t", body="b")) is True
    assert (tmp_path / "n.jsonl").exists()


def test_multi_sink_reports_total_failure():
    class Broken:
        def send(self, notification):
            return False

    assert MultiSink(sinks=[Broken()]).send(Notification(title="t", body="b")) is False


# -- the proactive gate ----------------------------------------------------


class _Gatekeeper:
    def __init__(self, verdict):
        self.verdict = verdict
        self.prompts = []

    def complete_json(self, tier, request, *, purpose="", default=None):
        self.prompts.append(request.messages[-1].content)
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict, None


def test_the_gate_stays_silent_on_a_routine_result():
    gate = NoticeGate(brain=_Gatekeeper({"notify": False}))
    assert gate.judge(task_name="t", prompt="p", result="all quiet") is None


def test_the_gate_speaks_when_it_matters():
    gate = NoticeGate(
        brain=_Gatekeeper({"notify": True, "title": "Disk full", "body": "97%", "urgency": "high"})
    )
    notification = gate.judge(task_name="t", prompt="p", result="CRITICAL 97%")
    assert notification.title == "Disk full" and notification.urgency == "high"


def test_a_broken_gate_stays_silent_rather_than_spamming():
    gate = NoticeGate(brain=_Gatekeeper(RuntimeError("model down")))
    assert gate.judge(task_name="t", prompt="p", result="something") is None


def test_the_previous_result_is_shown_so_repeats_can_be_suppressed():
    brain = _Gatekeeper({"notify": False})
    NoticeGate(brain=brain).judge(task_name="t", prompt="p", result="same", previous="same")
    assert "last run" in brain.prompts[0]


def test_an_empty_result_is_never_notified():
    gate = NoticeGate(brain=_Gatekeeper({"notify": True, "title": "x", "body": "y"}))
    assert gate.judge(task_name="t", prompt="p", result="   ") is None


def test_an_invalid_urgency_falls_back_to_normal():
    gate = NoticeGate(brain=_Gatekeeper({"notify": True, "title": "t", "body": "b", "urgency": "SCREAMING"}))
    assert gate.judge(task_name="t", prompt="p", result="x").urgency == "normal"


# -- the loop --------------------------------------------------------------


class _Agent:
    """Stands in for the agent: records prompts, returns scripted turns."""

    def __init__(self, replies=None, toolbox=None, memory=None):
        self.replies = list(replies or [])
        self.prompts = []
        self.toolbox = toolbox
        self.memory = memory
        self.brain = _Gatekeeper({"notify": False})
        self.conversation = None

    def chat(self, message, **kwargs):
        self.prompts.append(message)
        if self.replies:
            item = self.replies.pop(0)
            if isinstance(item, Exception):
                raise item
            return Turn(message=message, final=item)
        return Turn(message=message, final="done")


def _daemon(tmp_path, agent=None, store=None, **kwargs):
    toolbox = build_toolbox(workspace=tmp_path / "ws", mode=Mode.GUARDED, env={})
    agent = agent or _Agent(toolbox=toolbox)
    agent.toolbox = toolbox
    return Daemon(
        agent=agent,
        tasks=store or TaskStore(":memory:"),
        sink=FileSink(path=tmp_path / "n.jsonl"),
        home=tmp_path,
        **kwargs,
    )


def test_a_tick_runs_everything_due(tmp_path):
    now = _at("2026-08-30 14:05")
    daemon = _daemon(tmp_path)
    daemon.tasks.create("a", "prompt a", "every 15m", now=now)
    daemon.tasks.create("b", "prompt b", "every 15m", now=now)
    assert len(daemon.tick(now=now)) == 2
    assert daemon.agent.prompts == ["prompt a", "prompt b"]


def test_a_tick_with_nothing_due_does_nothing(tmp_path):
    now = _at("2026-08-30 14:05")
    daemon = _daemon(tmp_path)
    daemon.tasks.create("later", "x", "daily at 23:00", now=now)
    assert daemon.tick(now=now) == []


def test_one_failing_task_does_not_stop_the_others(tmp_path):
    now = _at("2026-08-30 14:05")
    agent = _Agent(replies=[RuntimeError("model exploded"), "second ran"])
    daemon = _daemon(tmp_path, agent=agent)
    daemon.tasks.create("boom", "a", "every 15m", now=now)
    daemon.tasks.create("fine", "b", "every 15m", now=now)
    runs = daemon.tick(now=now)
    assert [r.status for r in runs] == ["failed", "ok"]


def test_a_failure_is_recorded_against_the_task(tmp_path):
    now = _at("2026-08-30 14:05")
    daemon = _daemon(tmp_path, agent=_Agent(replies=[RuntimeError("nope")]))
    task = daemon.tasks.create("boom", "a", "every 15m", now=now)
    daemon.tick(now=now)
    assert daemon.tasks.get(task.id).fail_count == 1
    assert "RuntimeError" in daemon.tasks.get(task.id).last_output


def test_each_run_gets_a_fresh_conversation(tmp_path):
    now = _at("2026-08-30 14:05")
    daemon = _daemon(tmp_path)
    task = daemon.tasks.create("t", "x", "every 15m", now=now)
    daemon.run_task(task, now=now)
    first = daemon.agent.conversation
    daemon.run_task(task, now=now)
    assert daemon.agent.conversation is not first


def test_run_now_ignores_the_schedule(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.tasks.create("nightly", "the prompt", "daily at 03:00")
    assert daemon.run_now("nightly").status == "ok"
    assert daemon.agent.prompts == ["the prompt"]


def test_run_now_on_a_missing_task_returns_none(tmp_path):
    assert _daemon(tmp_path).run_now("ghost") is None


def test_notify_false_skips_the_gate_entirely(tmp_path):
    now = _at("2026-08-30 14:05")
    agent = _Agent()
    agent.brain = _Gatekeeper({"notify": True, "title": "t", "body": "b"})
    daemon = _daemon(tmp_path, agent=agent)
    daemon.gate = NoticeGate(brain=agent.brain)
    daemon.tasks.create("quiet", "x", "every 15m", notify=False, now=now)
    assert daemon.tick(now=now)[0].notified is False


def test_a_noteworthy_result_is_delivered(tmp_path):
    now = _at("2026-08-30 14:05")
    daemon = _daemon(tmp_path)
    daemon.gate = NoticeGate(brain=_Gatekeeper({"notify": True, "title": "Alert", "body": "bad"}))
    daemon.tasks.create("watch", "x", "every 15m", now=now)
    assert daemon.tick(now=now)[0].notified is True
    assert "Alert" in (tmp_path / "n.jsonl").read_text()


def test_sleep_is_bounded_at_both_ends(tmp_path):
    now = _at("2026-08-30 14:05")
    daemon = _daemon(tmp_path, max_sleep=60.0)
    assert daemon._sleep_for(now) == 60.0  # nothing scheduled
    daemon.tasks.create("soon", "x", "daily at 23:00", now=now)
    assert 1.0 <= daemon._sleep_for(now) <= 60.0


def test_the_daemon_reports_whether_it_can_run_commands(tmp_path):
    """Unattended, confirm-gated tools are denied — that must be visible."""
    daemon = _daemon(tmp_path)
    assert daemon.describe()["can_run_commands"] is False
    assert daemon.describe()["policy_mode"] == "guarded"


def test_a_repeated_failure_is_written_to_memory(tmp_path):
    class Store:
        def __init__(self):
            self.records = []

        def add(self, record):
            self.records.append(record)
            return record

    now = _at("2026-08-30 14:05")
    memory = Store()
    agent = _Agent(replies=[RuntimeError("backup failed")])
    agent.memory = memory
    daemon = _daemon(tmp_path, agent=agent)
    daemon.tasks.create("backup", "x", "every 15m", now=now)
    daemon.tick(now=now)
    assert len(memory.records) == 1
    assert "backup" in memory.records[0].content.lower()


def test_a_successful_run_is_not_written_to_memory(tmp_path):
    class Store:
        def __init__(self):
            self.records = []

        def add(self, record):
            self.records.append(record)
            return record

    now = _at("2026-08-30 14:05")
    agent = _Agent()
    agent.memory = Store()
    daemon = _daemon(tmp_path, agent=agent)
    daemon.tasks.create("fine", "x", "every 15m", now=now)
    daemon.tick(now=now)
    assert agent.memory.records == []


def test_events_are_emitted(tmp_path):
    now = _at("2026-08-30 14:05")
    seen = []
    daemon = _daemon(tmp_path, on_event=lambda e: seen.append(e.kind))
    daemon.tasks.create("t", "x", "every 15m", now=now)
    daemon.tick(now=now)
    assert seen == ["running", "finished"]


# -- did the task actually get done ----------------------------------------


class _Judge:
    """A stand-in completion check with a scripted set of verdicts."""

    def __init__(self, *verdicts):
        from itsbob.daemon.completion import Completion

        self.verdicts = [
            Completion(complete=ok, missing=missing, checked=True)
            for ok, missing in verdicts
        ]
        self.seen = []

    def judge(self, *, prompt, output, status="ok"):
        from itsbob.daemon.completion import Completion

        self.seen.append((prompt, output, status))
        return self.verdicts.pop(0) if self.verdicts else Completion(True, "", checked=True)


def test_an_incomplete_task_is_retried_at_a_higher_grade(tmp_path):
    """Nobody is awake at 07:00 to say "no, all of them"."""
    now = _at("2026-08-30 14:05")
    agent = _Agent(replies=["I found several fixtures today.", "1. Hull v Leeds\n2. ..."])
    judge = _Judge((False, "it counted the matches instead of listing them"))
    daemon = _daemon(tmp_path, agent=agent, completion=judge)
    daemon.tasks.create("fixtures", "list today's fixtures", "every 15m", now=now)

    runs = daemon.tick(now=now)
    assert [r.status for r in runs] == ["ok"]
    assert runs[0].output.startswith("1. Hull v Leeds")
    assert daemon.escalations == 1

    # The second attempt carries the first forward rather than starting over:
    # the first try is usually most of the answer.
    assert len(agent.prompts) == 2
    retry = agent.prompts[1]
    assert "list today's fixtures" in retry
    assert "it counted the matches instead of listing them" in retry
    assert "I found several fixtures today." in retry


def test_a_complete_task_is_not_run_twice(tmp_path):
    now = _at("2026-08-30 14:05")
    agent = _Agent(replies=["1. Hull v Leeds"])
    daemon = _daemon(tmp_path, agent=agent, completion=_Judge((True, "")))
    daemon.tasks.create("fixtures", "list today's fixtures", "every 15m", now=now)

    daemon.tick(now=now)
    assert len(agent.prompts) == 1
    assert daemon.escalations == 0


def test_escalation_stops_at_the_attempt_limit(tmp_path):
    """Never satisfied is not a reason to run forever."""
    now = _at("2026-08-30 14:05")
    agent = _Agent(replies=["thin", "still thin", "thin again", "and again"])
    judge = _Judge((False, "no"), (False, "no"), (False, "no"))
    daemon = _daemon(tmp_path, agent=agent, completion=judge)
    daemon.tasks.create("x", "do it", "every 15m", now=now, attempts=3)

    daemon.tick(now=now)
    assert len(agent.prompts) == 3
    assert daemon.escalations == 2


def test_the_check_can_be_turned_off(tmp_path):
    now = _at("2026-08-30 14:05")
    agent = _Agent(replies=["thin"])
    daemon = _daemon(tmp_path, agent=agent, completion=False)
    daemon.tasks.create("x", "do it", "every 15m", now=now)
    daemon.tick(now=now)
    assert len(agent.prompts) == 1


def test_a_broken_judge_never_blocks_delivery(tmp_path):
    """A verification pass that can wedge every task is worse than none."""
    from itsbob.daemon.completion import CompletionCheck

    class Exploding:
        def complete_json(self, *a, **k):
            raise RuntimeError("no model")

    verdict = CompletionCheck(brain=Exploding()).judge(
        prompt="list the fixtures", output="a reasonably long answer about fixtures"
    )
    assert verdict.complete is True
    assert verdict.checked is False, "a guess must not be recorded as a judgement"


def test_a_short_answer_is_not_automatically_incomplete():
    """"Nothing was scheduled today" is 27 characters and finished.

    A length threshold is a bad proxy for completeness, so only genuinely
    empty output is decided without asking.
    """
    from itsbob.daemon.completion import CompletionCheck

    class Asked:
        def __init__(self):
            self.calls = 0

        def complete_json(self, tier, request, **k):
            self.calls += 1
            return {"complete": True, "missing": ""}, None

    brain = Asked()
    check = CompletionCheck(brain=brain)
    assert check.judge(prompt="anything on today?", output="Nothing scheduled.").complete
    assert brain.calls == 1

    empty = check.judge(prompt="anything on?", output="   ")
    assert empty.complete is False and brain.calls == 1


def test_a_failed_run_is_incomplete_without_asking():
    from itsbob.daemon.completion import CompletionCheck

    class NeverAsked:
        def complete_json(self, *a, **k):
            raise AssertionError("asked a model about a run that already failed")

    verdict = CompletionCheck(brain=NeverAsked()).judge(
        prompt="x", output="some output", status="failed"
    )
    assert verdict.complete is False


def test_escalation_climbs_from_where_the_turn_actually_ran(tmp_path):
    """Not from the floor: the classifier may already have started higher."""
    from itsbob.daemon.completion import next_grade
    from itsbob.router.tiers import Tier

    assert next_grade(Tier.C) is Tier.B
    assert next_grade(Tier.B) is Tier.A
    assert next_grade(Tier.A) is Tier.S
    # S is the ceiling — a task that cannot be done at S will not be done at S
    # twice, so escalation stops rather than looping.
    assert next_grade(Tier.S) is Tier.S


def test_a_graded_task_floors_the_tier_but_does_not_cap_it(tmp_path):
    """A floor, not an override — the classifier may still go higher."""
    from itsbob.router.tiers import Tier

    now = _at("2026-08-30 14:05")
    seen = {}

    class Recording(_Agent):
        def chat(self, message, **kwargs):
            seen.update(kwargs)
            return super().chat(message, **kwargs)

    daemon = _daemon(tmp_path, agent=Recording(), completion=False)
    daemon.tasks.create("r", "write the review", "every 15m", now=now, grade="S")
    daemon.tick(now=now)
    assert seen["min_tier"] is Tier.S
    assert seen["thorough"] is True


def test_an_ungraded_task_leaves_the_classifier_alone(tmp_path):
    now = _at("2026-08-30 14:05")
    seen = {}

    class Recording(_Agent):
        def chat(self, message, **kwargs):
            seen.update(kwargs)
            return super().chat(message, **kwargs)

    daemon = _daemon(tmp_path, agent=Recording(), completion=False)
    daemon.tasks.create("r", "check the disk", "every 15m", now=now)
    daemon.tick(now=now)
    assert seen["min_tier"] is None
    # But effort still defaults to full: nobody is watching a scheduled run.
    assert seen["thorough"] is True


def test_a_quick_task_says_so(tmp_path):
    now = _at("2026-08-30 14:05")
    seen = {}

    class Recording(_Agent):
        def chat(self, message, **kwargs):
            seen.update(kwargs)
            return super().chat(message, **kwargs)

    daemon = _daemon(tmp_path, agent=Recording(), completion=False)
    daemon.tasks.create("r", "is the disk full", "every 15m", now=now, effort="quick")
    daemon.tick(now=now)
    assert seen["thorough"] is False


def test_a_grade_can_be_changed_without_losing_the_history(tmp_path):
    """A task that keeps coming back thin wants a higher grade, not a new identity."""
    from itsbob.daemon.tasks import TaskStore

    store = TaskStore(":memory:")
    task = store.create("r", "write the review", "daily at 07:00")
    assert task.grade == "auto" and task.effort == "full"

    store.update(task.id, grade="s")
    again = store.get(task.id)
    assert again.grade == "S", "a lowercase grade should be accepted"
    assert again.id == task.id and again.created_at == task.created_at

    # Nonsense falls back to auto rather than storing a grade nothing honours.
    store.update(task.id, grade="platinum")
    assert store.get(task.id).grade == "auto"

    # A bad schedule fails here, not silently at 07:00 tomorrow.
    import pytest

    from itsbob.daemon.schedule import ScheduleError

    with pytest.raises(ScheduleError):
        store.update(task.id, schedule="whenever I feel like it")
    assert store.get(task.id).schedule == "daily at 07:00", "a rejected edit still changed it"


def test_an_older_task_list_still_opens(tmp_path):
    """The columns are new; somebody's tasks.sqlite is not."""
    import sqlite3

    path = tmp_path / "tasks.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, prompt TEXT NOT NULL,
            schedule TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL, next_run REAL, last_run REAL,
            last_status TEXT, last_output TEXT, run_count INTEGER NOT NULL DEFAULT 0,
            fail_count INTEGER NOT NULL DEFAULT 0, notify INTEGER NOT NULL DEFAULT 1,
            max_runs INTEGER, metadata TEXT NOT NULL DEFAULT '{}'
        );
        INSERT INTO tasks (id, name, prompt, schedule, created_at)
        VALUES ('old1', 'briefing', 'do the briefing', 'daily at 07:00', 0);
        """
    )
    connection.commit()
    connection.close()

    from itsbob.daemon.tasks import TaskStore

    store = TaskStore(path)
    task = store.find("briefing")
    assert task is not None
    assert task.grade == "auto" and task.effort == "full" and task.attempts == 2
    # And it can still be written back.
    store.update(task.id, grade="A")
    assert store.get("old1").grade == "A"
