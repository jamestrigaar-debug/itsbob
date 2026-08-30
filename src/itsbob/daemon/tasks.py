"""Standing instructions the daemon runs on a schedule.

A task is a prompt plus a schedule plus a memory of how it has gone. It is
stored in SQLite next to the memory database, so the daemon can be restarted,
the laptop rebooted, and the schedule survives — which is the whole difference
between "a script I run" and "an assistant that is running".

Each run gets a fresh conversation but the *shared* long-term memory. Fresh,
because a nightly task should not inherit yesterday's context; shared, because
noticing "this is the third morning the backup has failed" is exactly the
thing that makes a proactive assistant worth having.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..store import Database
from .schedule import Schedule, parse_schedule

__all__ = ["Task", "TaskStore", "TaskRun"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    schedule     TEXT NOT NULL,
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   REAL NOT NULL,
    next_run     REAL,
    last_run     REAL,
    last_status  TEXT,
    last_output  TEXT,
    run_count    INTEGER NOT NULL DEFAULT 0,
    fail_count   INTEGER NOT NULL DEFAULT 0,
    notify       INTEGER NOT NULL DEFAULT 1,
    max_runs     INTEGER,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tasks_next ON tasks(next_run);

CREATE TABLE IF NOT EXISTS task_runs (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    started_at  REAL NOT NULL,
    duration_ms REAL NOT NULL,
    status      TEXT NOT NULL,
    output      TEXT NOT NULL DEFAULT '',
    tools       TEXT NOT NULL DEFAULT '[]',
    notified    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON task_runs(task_id, started_at);
"""

#: A task that keeps failing is usually broken rather than unlucky. It gets
#: disabled instead of failing forever, and the daemon says so once.
MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class Task:
    name: str
    prompt: str
    schedule: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    next_run: float | None = None
    last_run: float | None = None
    last_status: str | None = None
    last_output: str = ""
    run_count: int = 0
    fail_count: int = 0
    #: Whether a result may interrupt the user. The proactive gate still has
    #: to agree it is worth saying.
    notify: bool = True
    max_runs: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def parsed(self) -> Schedule:
        return parse_schedule(self.schedule)

    def is_due(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.enabled and self.next_run is not None and self.next_run <= now

    def is_spent(self) -> bool:
        return self.max_runs is not None and self.run_count >= self.max_runs

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "last_status": self.last_status,
            "run_count": self.run_count,
            "fail_count": self.fail_count,
            "notify": self.notify,
        }

    def describe(self) -> str:
        state = "on " if self.enabled else "off"
        when = time.strftime("%a %H:%M", time.localtime(self.next_run)) if self.next_run else "—"
        status = self.last_status or "never run"
        return f"[{state}] {self.id}  {self.name:<24} {self.schedule:<22} next {when:<10} {status}"


@dataclass
class TaskRun:
    task_id: str
    started_at: float
    duration_ms: float
    status: str  # ok | failed | skipped
    output: str = ""
    tools: tuple[str, ...] = ()
    notified: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class TaskStore:
    """SQLite-backed task list. Survives restarts, which is the point."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self._db = Database(database, schema=_SCHEMA)
        self.database = self._db.path

    def add(self, task: Task, *, now: float | None = None, first: bool = False) -> Task:
        if task.next_run is None:
            schedule = task.parsed()
            task.next_run = schedule.first_run(now) if first else schedule.next_after(now)
        self._db.execute(
            "INSERT OR REPLACE INTO tasks (id, name, prompt, schedule, enabled, created_at, "
            "next_run, last_run, last_status, last_output, run_count, fail_count, notify, "
            "max_runs, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                task.id, task.name, task.prompt, task.schedule, int(task.enabled),
                task.created_at, task.next_run, task.last_run, task.last_status,
                task.last_output, task.run_count, task.fail_count, int(task.notify),
                task.max_runs, json.dumps(task.metadata),
            ),
        )
        return task

    def create(
        self, name: str, prompt: str, schedule: str, *, now: float | None = None, **kwargs: Any
    ) -> Task:
        parse_schedule(schedule)  # validate before storing, so a typo fails loudly
        return self.add(
            Task(name=name, prompt=prompt, schedule=schedule, **kwargs), now=now, first=True
        )

    def get(self, task_id: str) -> Task | None:
        row = self._db.one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return _from_row(row) if row else None

    def find(self, needle: str) -> Task | None:
        """By id or by name, so the CLI can take either."""
        task = self.get(needle)
        if task is not None:
            return task
        row = self._db.one("SELECT * FROM tasks WHERE name = ? COLLATE NOCASE", (needle,))
        return _from_row(row) if row else None

    def all(self, *, enabled_only: bool = False) -> list[Task]:
        sql = "SELECT * FROM tasks"
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY next_run IS NULL, next_run ASC"
        return [_from_row(row) for row in self._db.query(sql)]

    def due(self, now: float | None = None) -> list[Task]:
        now = time.time() if now is None else now
        rows = self._db.query(
            "SELECT * FROM tasks WHERE enabled = 1 AND next_run IS NOT NULL AND next_run <= ? "
            "ORDER BY next_run ASC",
            (now,),
        )
        return [_from_row(row) for row in rows]

    def next_due_at(self) -> float | None:
        return self._db.scalar(
            "SELECT MIN(next_run) FROM tasks WHERE enabled = 1 AND next_run IS NOT NULL"
        )

    def set_enabled(self, task_id: str, enabled: bool, *, now: float | None = None) -> bool:
        task = self.get(task_id)
        if task is None:
            return False
        task.enabled = enabled
        if enabled and task.next_run is None:
            task.next_run = task.parsed().next_after(now)
        self.add(task)
        return True

    def remove(self, task_id: str) -> bool:
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
            return cursor.rowcount > 0

    def record_run(self, task: Task, run: TaskRun, *, now: float | None = None) -> Task:
        """Store the outcome and compute the next fire time."""
        now = time.time() if now is None else now
        task.last_run = run.started_at
        task.last_status = run.status
        task.last_output = run.output[:4000]
        task.run_count += 1
        task.fail_count = 0 if run.status == "ok" else task.fail_count + 1

        if task.is_spent():
            task.enabled = False
            task.next_run = None
        elif task.fail_count >= MAX_CONSECUTIVE_FAILURES:
            task.enabled = False
            task.next_run = None
            task.last_status = f"disabled after {task.fail_count} consecutive failures"
        else:
            task.next_run = task.parsed().next_after(now)
            if task.next_run is None:  # a spent one-shot
                task.enabled = False

        self._db.execute(
            "INSERT OR REPLACE INTO task_runs (id, task_id, started_at, duration_ms, status, "
            "output, tools, notified) VALUES (?,?,?,?,?,?,?,?)",
            (
                run.id, run.task_id, run.started_at, run.duration_ms, run.status,
                run.output[:8000], json.dumps(list(run.tools)), int(run.notified),
            ),
        )
        return self.add(task)

    def runs(self, task_id: str | None = None, *, limit: int = 20) -> list[dict[str, Any]]:
        sql = "SELECT * FROM task_runs"
        params: tuple[Any, ...] = ()
        if task_id:
            sql += " WHERE task_id = ?"
            params = (task_id,)
        sql += " ORDER BY started_at DESC LIMIT ?"
        return [dict(row) for row in self._db.query(sql, (*params, limit))]

    def __len__(self) -> int:
        return int(self._db.scalar("SELECT COUNT(*) FROM tasks", default=0))

    def close(self) -> None:
        self._db.close()


def _from_row(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        name=row["name"],
        prompt=row["prompt"],
        schedule=row["schedule"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        next_run=row["next_run"],
        last_run=row["last_run"],
        last_status=row["last_status"],
        last_output=row["last_output"] or "",
        run_count=row["run_count"],
        fail_count=row["fail_count"],
        notify=bool(row["notify"]),
        max_runs=row["max_runs"],
        metadata=json.loads(row["metadata"] or "{}"),
    )
