"""Continuously on: scheduled work and live chat sharing one agent.

``itsbob serve`` runs the daemon in its own process, which is right for a
machine you leave alone. But when the browser interface is open, you want both
at once — itsbob getting on with its scheduled work *and* answering you — and
two processes cannot share one agent, because the agent carries conversation
state and a SQLite handle.

So autonomous mode does not start a second daemon. It polls the same task store
and hands each due task to :class:`~itsbob.gui.session.Session` as a queued
message. That makes the session's queue the single point where everything
serializes, which buys three things that would otherwise need arranging
separately: scheduled work and chat cannot run at once; a question you type
goes ahead of pending scheduled work; and a task's steps stream into the same
activity panel as everything else, so "what is it doing right now" has one
answer.

The costs are real and worth stating. Closing the tab does not stop the work,
but quitting the server does — this is not a substitute for ``itsbob serve``
under systemd, it is the interactive version of it. And a long task delays your
next message until it finishes, since nothing preempts a running turn.
"""

from __future__ import annotations

import threading
import time
from typing import Any

__all__ = ["Autonomous"]


class Autonomous:
    """Polls for due tasks and feeds them through the chat queue."""

    def __init__(
        self,
        session: Any,
        tasks: Any,
        *,
        sink: Any = None,
        gate: Any = None,
        poll_seconds: float = 15.0,
        health_gate: bool = True,
        defer_seconds: float = 600.0,
    ) -> None:
        self.session = session
        self.tasks = tasks
        self.sink = sink
        self.gate = gate
        self.poll_seconds = poll_seconds
        self.health_gate = health_gate
        self.defer_seconds = defer_seconds

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.started_at: float | None = None
        self.runs = 0
        self.deferrals = 0
        self.last_reason: str | None = None
        #: Tasks handed to the queue but not yet finished. Without this a slow
        #: task would be submitted again on every poll while it was still running.
        self._in_flight: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop.clear()
            self.started_at = time.time()
            self._thread = threading.Thread(
                target=self._loop, name="itsbob-autonomous", daemon=True
            )
            self._thread.start()
            self.session.emit("autonomous", running=True)
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self._stop.set()
            self.session.emit("autonomous", running=False)
            return True

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at) if self.started_at else 0,
            "runs": self.runs,
            "deferrals": self.deferrals,
            "in_flight": sorted(self._in_flight),
            "next_due": self.tasks.next_due_at(),
            "health_gate": self.health_gate,
            "last_reason": self.last_reason,
        }

    # -- the loop ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll()
            except Exception as exc:  # noqa: BLE001 - a bad poll must not end the loop
                self.last_reason = f"{type(exc).__name__}: {exc}"[:200]
                self.session.emit("autonomous_error", error=self.last_reason)
            self._stop.wait(self.poll_seconds)

    def _poll(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        submitted: list[str] = []
        for task in self.tasks.due(now):
            if task.id in self._in_flight:
                continue
            held = self._health_hold(task)
            if held is not None:
                self._defer(task, held, now=now)
                continue
            if self._submit(task, now=now):
                submitted.append(task.name)
        return submitted

    def _health_hold(self, task: Any) -> str | None:
        if not self.health_gate or task.metadata.get("ignore_health"):
            return None
        try:
            from ..scripts.system_monitor import read_system

            state = read_system(sample_seconds=0.1)
        except Exception:  # noqa: BLE001 - a broken monitor must not stop the work
            return None
        return "; ".join(state.concerns) if state.concerns else None

    def _defer(self, task: Any, reason: str, *, now: float) -> None:
        """Push it back rather than running or failing it.

        A flat battery says nothing about whether the task works, so counting
        it as a failure would eventually disable a perfectly good task.
        """
        self.deferrals += 1
        self.last_reason = reason
        task.next_run = now + self.defer_seconds
        self.tasks.add(task)
        self.session.emit(
            "deferred", task=task.name, reason=reason, retry_in_s=self.defer_seconds
        )

    def _submit(self, task: Any, *, now: float) -> bool:

        started = time.time()
        self._in_flight.add(task.id)

        def finished(turn: Any, error: str | None) -> None:
            try:
                self._record(task, turn, error, started=started, now=now)
            finally:
                self._in_flight.discard(task.id)

        result = self.session.submit(
            task.prompt,
            source="task",
            label=task.name,
            on_done=finished,
        )
        if not result["accepted"]:
            # The queue is full. Leave the schedule alone so it is picked up on
            # the next poll rather than silently skipped.
            self._in_flight.discard(task.id)
            self.last_reason = result.get("error")
            return False
        return True

    def _record(self, task: Any, turn: Any, error: str | None, *, started: float, now: float) -> None:
        from ..daemon.tasks import TaskRun

        output = ""
        tools: tuple[str, ...] = ()
        status = "ok"
        if error or turn is None:
            status = "failed"
            output = error or "no result"
        else:
            output = turn.final
            tools = tuple(turn.tools_used)
            if turn.error:
                status = "failed"

        previous = task.last_output
        run = TaskRun(
            task_id=task.id,
            started_at=started,
            duration_ms=(time.time() - started) * 1000,
            status=status,
            output=output,
            tools=tools,
        )

        if task.notify and self.gate is not None:
            notification = self.gate.judge(
                task_name=task.name, prompt=task.prompt, result=output, previous=previous
            )
            if notification is not None:
                run.notified = self._deliver(notification)

        self.tasks.record_run(task, run, now=time.time())
        self.runs += 1
        self.session.emit(
            "task_finished",
            task=task.name,
            status=status,
            notified=run.notified,
            duration_ms=round(run.duration_ms, 1),
        )

    def _deliver(self, notification: Any) -> bool:
        if self.sink is None:
            return False
        try:
            delivered = bool(self.sink.send(notification))
        except Exception:  # noqa: BLE001
            return False
        if delivered:
            self.session.emit(
                "notified", title=notification.title, body=notification.body,
                urgency=notification.urgency,
            )
        return delivered
