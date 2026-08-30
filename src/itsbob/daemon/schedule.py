"""When a task should next run.

Cron is the obvious choice and the wrong one here: the schedules a person
actually wants from an assistant are "every 20 minutes", "weekdays at 08:30",
"once, tomorrow at 6" — and ``30 8 * * 1-5`` is a worse way to say the second
of those than "weekdays at 08:30" is. So schedules are written in words and
parsed here.

Everything is computed from an explicit ``now``, and :meth:`Schedule.next_after`
is pure. That is what makes the behaviour testable without waiting for a clock,
and it is why the daemon can be fast-forwarded in tests.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

__all__ = ["Schedule", "ScheduleError", "parse_schedule"]


class ScheduleError(ValueError):
    """The schedule text could not be understood."""


_UNITS = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
          "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
          "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
          "d": 86400, "day": 86400, "days": 86400,
          "w": 604800, "week": 604800, "weeks": 604800}

_DAYS = {"mon": 0, "monday": 0, "tue": 1, "tuesday": 1, "wed": 2, "wednesday": 2,
         "thu": 3, "thursday": 3, "fri": 4, "friday": 4, "sat": 5, "saturday": 5,
         "sun": 6, "sunday": 6}

_EVERY = re.compile(r"^every\s+(\d+)\s*([a-z]+)$")
_AT = re.compile(r"^(daily|every\s+day|weekdays|weekends|" + "|".join(_DAYS) + r")\s+at\s+(\d{1,2}):(\d{2})$")
_ONCE = re.compile(r"^(?:once\s+)?at\s+(.+)$")


@dataclass(frozen=True)
class Schedule:
    """A parsed schedule. ``kind`` is ``interval``, ``clock``, or ``once``."""

    kind: str
    text: str
    seconds: float = 0.0
    hour: int = 0
    minute: int = 0
    #: Weekday numbers (Mon=0) this may fire on. Empty means any day.
    days: tuple[int, ...] = ()
    at: float = 0.0

    def next_after(self, now: float | None = None) -> float | None:
        """The next fire time strictly after ``now``. ``None`` for a spent one-shot."""
        now = time.time() if now is None else now
        if self.kind == "interval":
            return now + self.seconds
        if self.kind == "once":
            return self.at if self.at > now else None

        moment = datetime.fromtimestamp(now)
        candidate = moment.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
        if candidate.timestamp() <= now:
            candidate += timedelta(days=1)
        # Walk forward to the next permitted weekday. Bounded at 8 so a
        # malformed day set can never spin.
        for _ in range(8):
            if not self.days or candidate.weekday() in self.days:
                return candidate.timestamp()
            candidate += timedelta(days=1)
        return None

    def first_run(self, now: float | None = None) -> float | None:
        """When a newly created task should first fire.

        An interval task starts straight away: someone who writes "every 15m"
        wants it working now, not in fifteen minutes, and a task that appears
        to do nothing after being created reads as broken. Clock and one-shot
        schedules name their own time, so they wait for it.
        """
        now = time.time() if now is None else now
        return now if self.kind == "interval" else self.next_after(now)

    def describe(self) -> str:
        return self.text

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text}


def parse_schedule(text: str) -> Schedule:
    """Parse a human schedule.

    Accepted::

        every 30s / every 15 minutes / every 2 hours / hourly / daily
        daily at 08:30 / weekdays at 09:00 / friday at 17:00 / weekends at 10:00
        at 2026-09-01T06:00 / once at 2026-09-01 06:00
    """
    raw = " ".join(str(text).strip().lower().split())
    if not raw:
        raise ScheduleError("schedule is empty")

    if raw in ("hourly", "every hour"):
        return Schedule(kind="interval", text="hourly", seconds=3600)
    if raw in ("daily", "every day"):
        return Schedule(kind="clock", text="daily at 09:00", hour=9, minute=0)
    if raw in ("minutely", "every minute"):
        return Schedule(kind="interval", text="every minute", seconds=60)

    match = _EVERY.match(raw)
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        if unit not in _UNITS:
            raise ScheduleError(f"unknown time unit {unit!r} (try s, m, h, d, w)")
        seconds = amount * _UNITS[unit]
        if seconds < 5:
            raise ScheduleError("minimum interval is 5 seconds")
        return Schedule(kind="interval", text=raw, seconds=seconds)

    match = _AT.match(raw)
    if match:
        which, hour, minute = match.group(1), int(match.group(2)), int(match.group(3))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ScheduleError(f"{hour:02d}:{minute:02d} is not a valid time")
        if which in ("daily", "every day"):
            days: tuple[int, ...] = ()
        elif which == "weekdays":
            days = (0, 1, 2, 3, 4)
        elif which == "weekends":
            days = (5, 6)
        else:
            days = (_DAYS[which],)
        return Schedule(kind="clock", text=raw, hour=hour, minute=minute, days=days)

    match = _ONCE.match(raw)
    if match:
        when = _parse_datetime(match.group(1))
        if when is None:
            raise ScheduleError(f"could not read a date and time from {match.group(1)!r}")
        return Schedule(kind="once", text=raw, at=when)

    raise ScheduleError(
        f"could not parse schedule {text!r}. Try 'every 15m', 'daily at 08:30', "
        "'weekdays at 09:00', or 'at 2026-09-01T06:00'."
    )


def _parse_datetime(text: str) -> float | None:
    text = text.strip().replace("/", "-")
    for fmt in ("%Y-%m-%dt%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dt%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None
