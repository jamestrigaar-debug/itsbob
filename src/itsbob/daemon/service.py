"""The always-on loop.

Sleep until something is due, run it, decide whether to say anything, repeat.

Three properties it is built for:

**It survives restarts.** Schedules and run history live in SQLite, so the
laptop can reboot and the 08:30 task still fires at 08:30.

**It cannot silently gain permissions.** The daemon runs with nobody to ask,
so :class:`~itsbob.tools.policy.Policy` denies every confirm-gated tool — by
construction, not by convention. A task that needs to run commands has to be
granted that explicitly (``auto_allow``, or trusted mode), and
:meth:`Daemon.describe` reports which it is. An unattended agent quietly
acquiring the ability to run anything is the failure this design refuses.

**One bad task cannot take the loop down.** Every run is wrapped; a failure is
recorded against that task and the loop continues. Five consecutive failures
disable the task, because at that point it is broken rather than unlucky and
running it hourly forever helps nobody.

Each run gets a fresh conversation and the shared long-term memory, so a task
starts clean but can still notice "third morning in a row".
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent import Agent, build_agent
from ..agent.context import Conversation
from ..memory.base import MemoryKind, MemoryRecord
from .notify import MultiSink, Notification, NoticeGate, default_sink
from .tasks import Task, TaskRun, TaskStore

__all__ = ["Daemon", "DaemonEvent"]


@dataclass
class DaemonEvent:
    kind: str  #: started | waiting | running | finished | notified | error | stopped
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


EventFn = Callable[[DaemonEvent], None]


class Daemon:
    """Runs scheduled tasks through the agent, and decides what to surface."""

    def __init__(
        self,
        *,
        agent: Agent,
        tasks: TaskStore,
        sink: Any = None,
        gate: NoticeGate | None = None,
        home: Path | None = None,
        tick_seconds: float = 15.0,
        max_sleep: float = 60.0,
        on_event: EventFn | None = None,
        remember_runs: bool = True,
    ) -> None:
        self.agent = agent
        self.tasks = tasks
        self.home = Path(home) if home else Path.home() / ".itsbob"
        self.sink = sink if sink is not None else default_sink(self.home)
        self.gate = gate or NoticeGate(brain=agent.brain)
        self.tick_seconds = tick_seconds
        #: Never sleep longer than this even with nothing scheduled, so a task
        #: added by another process is picked up within a minute.
        self.max_sleep = max_sleep
        self.on_event = on_event
        self.remember_runs = remember_runs
        self._stop = threading.Event()
        self.started_at: float | None = None
        self.runs_completed = 0
        self.notifications_sent = 0

    # -- lifecycle ---------------------------------------------------------

    def run_forever(self) -> None:
        """Block until :meth:`stop` is called or the process is interrupted."""
        self.started_at = time.time()
        self._emit("started", tasks=len(self.tasks), policy=self.agent.toolbox.policy.mode.value)
        try:
            while not self._stop.is_set():
                self.tick()
                self._stop.wait(self._sleep_for())
        except KeyboardInterrupt:  # pragma: no cover - interactive
            pass
        finally:
            self._emit("stopped", runs=self.runs_completed, notified=self.notifications_sent)

    def stop(self) -> None:
        self._stop.set()

    def _sleep_for(self, now: float | None = None) -> float:
        """Sleep until the next task is due, bounded at both ends."""
        now = time.time() if now is None else now
        next_due = self.tasks.next_due_at()
        if next_due is None:
            return self.max_sleep
        return max(1.0, min(self.max_sleep, next_due - now))

    # -- work --------------------------------------------------------------

    def tick(self, now: float | None = None) -> list[TaskRun]:
        """Run everything currently due. Returns what ran."""
        now = time.time() if now is None else now
        due = self.tasks.due(now)
        if not due:
            self._emit("waiting", next_due=self.tasks.next_due_at())
            return []
        return [self.run_task(task, now=now) for task in due]

    def run_task(self, task: Task, *, now: float | None = None) -> TaskRun:
        """One task, start to finish, never raising."""
        now = time.time() if now is None else now
        started = time.perf_counter()
        self._emit("running", task=task.name, id=task.id)
        previous = task.last_output

        status, output, tools = "ok", "", ()
        try:
            turn = self._run_prompt(task.prompt)
            output = turn.final
            tools = tuple(turn.tools_used)
            if turn.error:
                status = "failed"
        except Exception as exc:  # noqa: BLE001 - one task must not stop the loop
            status = "failed"
            output = f"{type(exc).__name__}: {exc}"
            self._emit("error", task=task.name, error=output)

        run = TaskRun(
            task_id=task.id,
            started_at=now,
            duration_ms=(time.perf_counter() - started) * 1000,
            status=status,
            output=output,
            tools=tools,
        )

        if task.notify and status != "skipped":
            notification = self.gate.judge(
                task_name=task.name, prompt=task.prompt, result=output, previous=previous
            )
            if notification is not None and self._deliver(notification):
                run.notified = True
                self.notifications_sent += 1

        if self.remember_runs and status == "failed":
            # Only failures are written. A successful nightly run is not a
            # durable fact; a repeatedly failing one is exactly the pattern
            # worth surfacing on the third morning.
            self._remember_failure(task, output)

        self.tasks.record_run(task, run, now=now)
        self.runs_completed += 1
        self._emit("finished", task=task.name, status=status, notified=run.notified,
                   duration_ms=round(run.duration_ms, 1))
        return run

    def run_now(self, needle: str) -> TaskRun | None:
        """Run a task immediately by id or name, off-schedule."""
        task = self.tasks.find(needle)
        return None if task is None else self.run_task(task)

    def _run_prompt(self, prompt: str):
        # A fresh conversation per run: a nightly task should not inherit
        # yesterday's context. Long-term memory is shared, so it still can.
        self.agent.conversation = Conversation()
        return self.agent.chat(prompt)

    def _deliver(self, notification: Notification) -> bool:
        try:
            delivered = bool(self.sink.send(notification))
        except Exception:  # noqa: BLE001
            return False
        if delivered:
            self._emit("notified", title=notification.title, urgency=notification.urgency)
        return delivered

    def _remember_failure(self, task: Task, output: str) -> None:
        if self.agent.memory is None:
            return
        try:
            self.agent.memory.add(
                MemoryRecord(
                    content=f"Scheduled task '{task.name}' failed: {output[:300]}",
                    kind=MemoryKind.OBSERVATION,
                    importance=0.55,
                    tags=("task", "failure", task.name.lower().replace(" ", "-")),
                    metadata={"source": "daemon", "task_id": task.id},
                )
            )
        except Exception:  # noqa: BLE001 - bookkeeping is never load-bearing
            pass

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(DaemonEvent(kind=kind, data=data))
        except Exception:  # noqa: BLE001
            pass

    # -- introspection -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        policy = self.agent.toolbox.policy
        return {
            "running": not self._stop.is_set() and self.started_at is not None,
            "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "tasks": len(self.tasks),
            "enabled_tasks": len(self.tasks.all(enabled_only=True)),
            "next_due": self.tasks.next_due_at(),
            "runs_completed": self.runs_completed,
            "notifications_sent": self.notifications_sent,
            "policy_mode": policy.mode.value,
            "can_run_commands": (
                policy.mode.value == "trusted" or "run_shell" in policy.auto_allow
            ),
            "workspace": str(policy.workspace),
        }


def build_daemon(
    *,
    home: str | Path | None = None,
    agent: Agent | None = None,
    tasks: TaskStore | None = None,
    sink: Any = None,
    console: bool = True,
    on_event: EventFn | None = None,
    **agent_kwargs: Any,
) -> Daemon:
    """Assemble a daemon over the standard home directory."""
    from ..agent import default_home

    root = Path(home).expanduser() if home else default_home()
    root.mkdir(parents=True, exist_ok=True)
    agent = agent or build_agent(home=root, **agent_kwargs)
    return Daemon(
        agent=agent,
        tasks=tasks or TaskStore(root / "tasks.sqlite"),
        sink=sink if sink is not None else default_sink(root, console=console),
        home=root,
        on_event=on_event,
    )
