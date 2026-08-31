"""Letting itsbob schedule its own work.

This is deliberately a *wrapper*, not a scheduler. itsbob already has one — the
daemon's :class:`~itsbob.daemon.tasks.TaskStore` with SQLite persistence, run
history, failure backoff and the notification gate — and a second one would
mean two competing answers to "what runs when", which is the kind of split that
ends with a task that is enabled in one place and disabled in the other. So
these tools read and write the same store ``itsbob serve`` runs from, and
anything scheduled here is picked up by a daemon already running, without a
restart.

One capability is missing on purpose: there is no "run this task now" tool.
A task's instruction is executed *by the agent*, so a tool that triggers one
would let a turn start another turn — and the first thing an agent does with
that is schedule a task that runs itself. ``itsbob task run <name>`` exists for
people, where a human is deciding.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["tools", "store_for"]


def store_for(ctx: ToolContext):
    """The same task database the daemon uses.

    Taken from the context when the agent was given one, so tests and alternate
    homes work; otherwise the configured home, which is what the daemon reads.
    """
    from ..config import itsbob_home
    from ..daemon.tasks import TaskStore

    existing = ctx.extras.get("task_store") if ctx.extras else None
    if existing is not None:
        return existing
    home = ctx.extras.get("home") if ctx.extras else None
    return TaskStore(Path(home or itsbob_home()) / "tasks.sqlite")


def _when(timestamp: float | None) -> str:
    if not timestamp:
        return "—"
    delta = timestamp - time.time()
    if delta < 0:
        return f"{_span(-delta)} ago"
    return f"in {_span(delta)}"


def _span(seconds: float) -> str:
    for limit, unit, size in ((90, "s", 1), (5400, "m", 60), (172800, "h", 3600)):
        if seconds < limit:
            return f"{int(seconds // size)}{unit}"
    return f"{int(seconds // 86400)}d"


def _list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = store_for(ctx)
    tasks = store.all()
    if not tasks:
        return ToolResult(
            ok=True,
            output="no scheduled tasks yet",
            data={"tasks": [], "count": 0},
        )
    rows = []
    for task in tasks:
        state = "on " if task.enabled else "off"
        rows.append(
            f"  [{state}] {task.name:<20} {task.schedule:<20} next {_when(task.next_run):<10} "
            f"grade {task.grade:<4} {task.run_count} run(s), last: {task.last_status or 'never'}"
        )
    return ToolResult(
        ok=True,
        output="\n".join(rows),
        data={"tasks": [t.as_dict() for t in tasks], "count": len(tasks)},
    )


def _schedule(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from ..daemon.schedule import ScheduleError, parse_schedule

    store = store_for(ctx)
    name = str(params["name"]).strip()
    instruction = str(params["instruction"]).strip()
    schedule = str(params["schedule"]).strip()

    if store.find(name) is not None:
        raise ToolError(
            f"a task called {name!r} already exists. Remove it first, or pick another name — "
            "quietly replacing a schedule someone else set is not something to do by accident."
        )
    try:
        parse_schedule(schedule)
    except ScheduleError as exc:
        raise ToolError(str(exc)) from exc

    grade = str(params.get("grade") or "auto").strip()
    grade = grade.upper() if grade.upper() in ("C", "B", "A", "S") else "auto"
    task = store.create(
        name,
        instruction,
        schedule,
        notify=bool(params.get("notify", True)),
        grade=grade,
        effort="quick" if str(params.get("effort", "")).strip().lower() == "quick" else "full",
    )
    return ToolResult(
        ok=True,
        output=(
            f"scheduled '{task.name}' ({task.schedule}), first run {_when(task.next_run)}"
            f" — grade {task.grade}, {task.effort} effort"
        ),
        data=task.as_dict(),
    )


def _update(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = store_for(ctx)
    task = store.find(str(params["name"]))
    if task is None:
        raise ToolError(f"no task called {params['name']!r}")

    fields = {
        key: params[key]
        for key in ("prompt", "schedule", "grade", "effort", "attempts", "notify")
        if key in params and params[key] is not None
    }
    if not fields:
        raise ToolError(
            "nothing to change — pass at least one of prompt, schedule, grade, "
            "effort, attempts or notify"
        )
    try:
        updated = store.update(task.id, **fields)
    except Exception as exc:  # noqa: BLE001 - a bad schedule is the usual cause
        raise ToolError(str(exc)) from exc
    return ToolResult(
        ok=True,
        output=(
            f"updated '{updated.name}' — grade {updated.grade}, {updated.effort} effort, "
            f"{updated.schedule}, next run {_when(updated.next_run)}"
        ),
        data=updated.as_dict(),
    )


def _remove(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = store_for(ctx)
    task = store.find(str(params["name"]))
    if task is None:
        raise ToolError(f"no task called {params['name']!r}")
    store.remove(task.id)
    return ToolResult(ok=True, output=f"removed '{task.name}' and its run history",
                      data={"removed": task.as_dict()})


def _set_enabled(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = store_for(ctx)
    task = store.find(str(params["name"]))
    if task is None:
        raise ToolError(f"no task called {params['name']!r}")
    enabled = bool(params.get("enabled", True))
    store.set_enabled(task.id, enabled)
    return ToolResult(
        ok=True,
        output=f"{'resumed' if enabled else 'paused'} '{task.name}'",
        data={"name": task.name, "enabled": enabled},
    )


def _history(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = store_for(ctx)
    task = store.find(str(params["name"]))
    if task is None:
        raise ToolError(f"no task called {params['name']!r}")
    runs = store.runs(task.id, limit=int(params.get("limit", 10)))
    if not runs:
        return ToolResult(ok=True, output=f"'{task.name}' has not run yet", data={"runs": []})
    rows = [
        f"  {time.strftime('%d %b %H:%M', time.localtime(r['started_at']))}  "
        f"{r['status']:<7} {r['duration_ms'] / 1000:>6.1f}s  {r['output'][:70]}"
        for r in runs
    ]
    return ToolResult(ok=True, output="\n".join(rows), data={"runs": runs})


def tools() -> list[Tool]:
    return [
        Tool(
            name="list_scheduled_tasks",
            description=(
                "What itsbob is scheduled to do on its own, when each next runs, and how "
                "the last run went. Check before scheduling anything, to avoid duplicates."
            ),
            run=_list,
            risk=Risk.READ,
            parameters={"type": "object", "properties": {}},
        ),
        Tool(
            name="schedule_task",
            description=(
                "Add recurring work for itsbob to do by itself. The instruction is plain "
                "language, run by the agent exactly as if you had typed it. Schedules look "
                "like 'every 30m', 'weekdays at 08:30', 'friday at 17:00', "
                "'at 2026-09-01T06:00'. Only runs while `itsbob serve` is running. "
                "Scheduled work runs at full effort and is checked afterwards for "
                "whether it actually did what was asked, retrying at a higher grade "
                "if not — so ask for the whole job here, not a sketch of it."
            ),
            run=_schedule,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short unique label."},
                    "instruction": {"type": "string", "description": "What to do, in plain language."},
                    "schedule": {"type": "string", "description": "e.g. 'every 30m' or 'weekdays at 08:30'."},
                    "notify": {"type": "boolean", "description": "May the result interrupt the user? Default true."},
                    "grade": {
                        "type": "string",
                        "description": (
                            "Floor on how hard to think: C cheap, B light, A standard, "
                            "S strongest. Default 'auto' lets each run be classified. "
                            "Set A or S for anything that has to be right or complete."
                        ),
                    },
                    "effort": {
                        "type": "string",
                        "description": (
                            "'full' (default) finishes the job — a report comes back as a "
                            "report. 'quick' for a one-line check where that is all it is."
                        ),
                    },
                },
                "required": ["name", "instruction", "schedule"],
            },
        ),
        Tool(
            name="update_task",
            description=(
                "Change an existing task without losing its history: its grade (how hard "
                "to think), effort, schedule, or the instruction itself. Use this rather "
                "than removing and re-adding — a task that keeps coming back thin wants a "
                "higher grade, not a new identity."
            ),
            run=_update,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Task id or name."},
                    "prompt": {"type": "string", "description": "New instruction."},
                    "schedule": {"type": "string", "description": "New schedule."},
                    "grade": {
                        "type": "string",
                        "description": "C, B, A, S — or 'auto' to classify each run.",
                    },
                    "effort": {"type": "string", "description": "'full' or 'quick'."},
                    "attempts": {
                        "type": "integer",
                        "description": "Tries allowed at a higher grade when a run falls short. 1 disables.",
                    },
                    "notify": {"type": "boolean"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="pause_task",
            description="Stop a scheduled task running, keeping it and its history. Set enabled=true to resume.",
            run=_set_enabled,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "enabled": {"type": "boolean", "description": "false to pause (default), true to resume."},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="remove_task",
            description="Delete a scheduled task and its run history. Prefer pause_task if it might be wanted later.",
            run=_remove,
            risk=Risk.DESTRUCTIVE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        Tool(
            name="task_history",
            description="Recent runs of one scheduled task: when, how long, and what it produced.",
            run=_history,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["name"],
            },
        ),
    ]
