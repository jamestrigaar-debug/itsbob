"""The always-on half: scheduled tasks, and deciding what is worth saying.

    from itsbob.daemon import build_daemon

    bob = build_daemon()
    bob.tasks.create("inbox", "Check ~/inbox for new CSVs and summarise them", "every 30m")
    bob.run_forever()
"""

from __future__ import annotations

from .notify import (
    ConsoleSink,
    DesktopSink,
    FileSink,
    MultiSink,
    NoticeGate,
    Notification,
    WebhookSink,
    default_sink,
)
from .schedule import Schedule, ScheduleError, parse_schedule
from .service import Daemon, DaemonEvent, build_daemon
from .tasks import Task, TaskRun, TaskStore

__all__ = [
    "ConsoleSink",
    "Daemon",
    "DaemonEvent",
    "DesktopSink",
    "FileSink",
    "MultiSink",
    "NoticeGate",
    "Notification",
    "Schedule",
    "ScheduleError",
    "Task",
    "TaskRun",
    "TaskStore",
    "WebhookSink",
    "build_daemon",
    "default_sink",
    "parse_schedule",
]
