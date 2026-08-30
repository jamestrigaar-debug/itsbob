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

**Nor can a slow one.** Tasks run on a worker thread with a hard wall-clock
deadline. Without it a single task that wedges — a tool waiting on a socket
that never answers — stalls every other schedule indefinitely, and the symptom
("my 8am briefing stopped arriving") points nowhere near the cause. Python
cannot kill a thread, so an overrunning task is abandoned rather than stopped:
it is recorded as failed, the loop moves on, and the orphan finishes into a
locked database where it can do no damage. That is a real cost, and it is
still much better than blocking the daemon.

**Ctrl-C and SIGTERM stop it cleanly.** A service manager stopping the daemon
mid-task would otherwise lose the run record entirely.

**It does not start heavy work on a machine that cannot take it.** Before each
run the daemon reads :func:`~itsbob.scripts.system_monitor.read_system`, and if
the laptop is on a nearly-flat battery, thermally throttled, or out of disk, the
task is *deferred* rather than run or failed — deferred, because "the battery
was low at 8:30" is not a failure of the task and should not count toward
disabling it. After an hour of being held back it says so once, since silently
never running is its own kind of broken. A task can opt out with
``metadata={"ignore_health": True}`` when it is light enough not to care.

Each run gets a fresh conversation and the shared long-term memory, so a task
starts clean but can still notice "third morning in a row".
"""

from __future__ import annotations

import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent import Agent, build_agent
from ..agent.context import Conversation
from ..memory.base import MemoryKind, MemoryRecord
from .notify import Notification, NoticeGate, default_sink
from .tasks import Task, TaskRun, TaskStore

__all__ = ["Daemon", "DaemonEvent", "TaskTimeout"]


class TaskTimeout(TimeoutError):
    """A task ran past the daemon's per-task deadline and was abandoned."""


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
        task_timeout: float = 600.0,
        handle_signals: bool = True,
        health_gate: bool = True,
        defer_seconds: float = 600.0,
        max_defers: int = 6,
        discord: Any = None,
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
        #: Hard wall-clock bound per task. Generous, because a task may
        #: legitimately do a lot of work; finite, because nothing may wedge the
        #: loop. Set to 0 to disable (and accept that a stuck task stops
        #: everything).
        self.task_timeout = task_timeout
        self.handle_signals = handle_signals
        #: Hold work back when the machine is in no state for it.
        self.health_gate = health_gate
        self.defer_seconds = defer_seconds
        #: After this many consecutive deferrals (about an hour at the default
        #: interval) say so once — work that is never run is also a failure.
        self.max_defers = max_defers
        #: Two-way Discord, when configured. Messages typed in the channel are
        #: queued here and run between ticks, on this same thread — one
        #: consumer for the agent, so a chat message and a scheduled task can
        #: never interleave into each other's conversation.
        self.discord = discord
        self._inbox: queue.Queue = queue.Queue(maxsize=50)
        self._stop = threading.Event()
        self._abandoned = 0
        self._defers: dict[str, int] = {}
        self._defer_notified: set[str] = set()
        self.started_at: float | None = None
        self.runs_completed = 0
        self.notifications_sent = 0

    # -- lifecycle ---------------------------------------------------------

    def run_forever(self) -> None:
        """Block until :meth:`stop` is called, or a stop signal arrives."""
        self.started_at = time.time()
        self._install_signal_handlers()
        self._start_discord()
        self._emit("started", tasks=len(self.tasks), policy=self.agent.toolbox.policy.mode.value)
        try:
            while not self._stop.is_set():
                self.tick()
                self.drain_inbox()
                if self._stop.is_set():
                    break
                # Capped when Discord is listening: a person waiting on a reply
                # measures the delay differently to a nightly backup does.
                wait = self._sleep_for()
                if self.discord is not None:
                    wait = min(wait, 3.0)
                self._stop.wait(wait)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            pass
        finally:
            if self.discord is not None:
                self.discord.stop()
            self._emit("stopped", runs=self.runs_completed, notified=self.notifications_sent)

    def _start_discord(self) -> None:
        if self.discord is None:
            return
        if self.discord.start():
            self._emit("discord_started", channel=self.discord.client.channel_id)
        else:
            self._emit("discord_failed", error=self.discord.last_error or "unknown")
            self.discord = None

    def submit_message(self, text: str, **extra: Any) -> bool:
        """Queue an outside message (from Discord) to run between ticks."""
        try:
            self._inbox.put_nowait((text, extra))
        except queue.Full:
            return False
        return True

    def drain_inbox(self) -> int:
        """Answer everything queued from outside. Returns how many ran."""
        answered = 0
        while not self._stop.is_set():
            try:
                text, extra = self._inbox.get_nowait()
            except queue.Empty:
                return answered
            self._emit("message", source=extra.get("source", "external"), text=text[:200])
            turn = None
            error: str | None = None
            try:
                turn = self._run_prompt(text)
            except Exception as exc:  # noqa: BLE001 - one bad message, not the loop
                error = f"{type(exc).__name__}: {exc}"
                self._emit("message_failed", error=error)
            callback = extra.get("on_done")
            if callback is not None:
                try:
                    callback(turn, error)
                except Exception:  # noqa: BLE001
                    pass
            answered += 1
        return answered

    def _install_signal_handlers(self) -> None:
        """Turn SIGTERM/SIGINT into a clean stop, when we can.

        ``signal.signal`` only works on the main thread, and a caller embedding
        the daemon may want its own handlers — so a failure here is ignored
        rather than fatal.
        """
        if not self.handle_signals:
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: self.stop())
            except (ValueError, OSError, AttributeError):  # not main thread, or unsupported
                return

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

        runs: list[TaskRun] = []
        for task in due:
            held = self._health_hold(task)
            if held is not None:
                self._defer(task, held, now=now)
                continue
            self._defers.pop(task.id, None)
            self._defer_notified.discard(task.id)
            runs.append(self.run_task(task, now=now))
        return runs

    def _health_hold(self, task: Task) -> str | None:
        """Why this task should wait, or None to go ahead."""
        if not self.health_gate or task.metadata.get("ignore_health"):
            return None
        try:
            from ..scripts.system_monitor import read_system

            state = read_system()
        except Exception:  # noqa: BLE001 - a broken monitor must not stop the work
            return None
        return "; ".join(state.concerns) if state.concerns else None

    def _defer(self, task: Task, reason: str, *, now: float) -> None:
        """Push a task back rather than running or failing it.

        Deferred, not failed: "the battery was flat at 08:30" says nothing about
        whether the task works, and counting it as a failure would disable a
        perfectly good task after five bad mornings.
        """
        count = self._defers.get(task.id, 0) + 1
        self._defers[task.id] = count
        task.next_run = (now or time.time()) + self.defer_seconds
        self.tasks.add(task)
        self._emit("deferred", task=task.name, reason=reason, count=count,
                   retry_in_s=self.defer_seconds)

        if count >= self.max_defers and task.id not in self._defer_notified:
            self._defer_notified.add(task.id)
            held_for = count * self.defer_seconds / 60
            self._deliver(
                Notification(
                    title=f"'{task.name}' is being held back",
                    body=(
                        f"Not run for about {held_for:.0f} minutes because: {reason}. "
                        "It will start on its own once the machine is in better shape."
                    ),
                    task=task.name,
                    urgency="low",
                )
            )

    def run_task(self, task: Task, *, now: float | None = None) -> TaskRun:
        """One task, start to finish, never raising."""
        now = time.time() if now is None else now
        started = time.perf_counter()
        self._emit("running", task=task.name, id=task.id)
        previous = task.last_output

        status, output, tools = "ok", "", ()
        try:
            turn = self._run_bounded(task)
            output = turn.final
            tools = tuple(turn.tools_used)
            if turn.error:
                status = "failed"
        except TaskTimeout:
            status = "failed"
            self._abandoned += 1
            output = (
                f"exceeded the {self.task_timeout:g}s task limit and was abandoned. "
                "It may still be running in the background; the daemon moved on so "
                "other schedules are not held up."
            )
            self._emit("error", task=task.name, error=output)
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

    def _run_bounded(self, task: Task):
        """Run one task with a hard deadline, on its own thread.

        A fresh thread per run, not a shared worker pool. With a pool of one, a
        task that wedges holds the only worker and every *subsequent* task times
        out behind it — which turns one broken task into a broken daemon, the
        exact failure the deadline exists to prevent. A pool sized above one
        just moves the ceiling. Daemon threads, so an abandoned run never keeps
        the process alive at exit.
        """
        if not self.task_timeout:
            return self._run_prompt(task.prompt)

        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = self._run_prompt(task.prompt)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc

        thread = threading.Thread(
            target=target, name=f"itsbob-task-{task.id}", daemon=True
        )
        thread.start()
        thread.join(self.task_timeout)
        if thread.is_alive():
            raise TaskTimeout(f"task {task.name!r} exceeded {self.task_timeout:g}s")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _run_prompt(self, prompt: str):
        # A fresh conversation per run: a nightly task should not inherit
        # yesterday's context. Long-term memory is shared, so it still can.
        self.agent.conversation = Conversation()
        # The agent's own budget sits just inside the daemon's, so an overrun
        # normally ends as a proper turn ("I ran out of time, here is what I
        # found") rather than as an abandoned thread.
        # getattr, because the agent is injectable and a stand-in need not
        # carry every field the real one does.
        current = getattr(self.agent, "max_seconds", None)
        if self.task_timeout and isinstance(current, (int, float)):
            self.agent.max_seconds = min(current, self.task_timeout * 0.9)
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
            "abandoned_runs": self._abandoned,
            "task_timeout_s": self.task_timeout,
            "health_gate": self.health_gate,
            "deferred_now": {name: count for name, count in self._defers.items() if count},
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
    daemon = Daemon(
        agent=agent,
        tasks=tasks or TaskStore(root / "tasks.sqlite"),
        sink=sink if sink is not None else _default_sink_with_discord(root, console=console),
        home=root,
        on_event=on_event,
    )
    daemon.discord = _build_bridge(daemon)
    return daemon


def _build_bridge(daemon: Daemon) -> Any:
    """The Discord bridge, wired to answer through the daemon's own inbox."""
    from ..integrations.discord import DiscordBridge, DiscordClient

    client = DiscordClient.from_env()
    if client is None:
        return None
    return DiscordBridge(client=client, submit=daemon.submit_message)


def _default_sink_with_discord(root: Path, *, console: bool) -> Any:
    """The usual sinks, plus the Discord channel when one is configured.

    Appended, never substituted: `notifications.jsonl` is what the messages
    window reads, and a configured channel must not empty that page.
    """
    from ..daemon.notify import MultiSink
    from ..integrations.discord import DiscordClient, DiscordSink

    base = default_sink(root, console=console)
    client = DiscordClient.from_env()
    if client is None:
        return base
    return MultiSink(sinks=[*base.sinks, DiscordSink(client=client)])
