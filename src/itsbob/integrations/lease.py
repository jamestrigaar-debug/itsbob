"""One consumer for a shared channel, across processes.

Discord is polled, and polling is only safe if exactly one thing does it. Both
``itsbob serve`` and the browser's continuous mode build a bridge, and with both
running each has its own cursor, sees the same message, runs its own turn and
posts its own answer. The symptom is two replies to every message — not
identical ones either, since they are genuinely separate turns with separate
conversation state, so one may recommend a film the other has never heard of.
The second is pure waste: a whole turn's tokens spent producing something nobody
asked for.

A lease fixes it where the problem is, rather than by asking people to run only
one thing. It is a file, because the two claimants are separate processes on one
machine and a file is the smallest thing both can see:

* **Whoever holds it polls.** Everyone else stays in standby, still running, so
  a crash hands over rather than stopping Discord entirely.
* **It expires.** A holder renews on every poll; a holder that dies stops
  renewing and the lease is free again after ``ttl``. No cleanup step to forget.
* **It remembers what was answered.** The ids of recently answered messages ride
  along, so a handover — or a restart — does not answer the same message twice.

Deliberately not a lock file with a pid check. A pid tells you a process exists,
not that it is still polling Discord, and pids are reused.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DiscordLease"]

#: How many answered ids to carry. Enough to cover a handover and a restart,
#: small enough that the file stays a single cheap write.
REMEMBERED = 60


@dataclass
class DiscordLease:
    """A renewable claim on being the one that answers a channel."""

    path: Path
    #: Seconds a claim survives without renewal. Comfortably longer than one
    #: poll, so an ordinary slow turn does not lose the lease mid-answer.
    ttl: float = 90.0
    #: What this claimant is, for the status panel: "daemon" or "browser".
    role: str = "unknown"
    owner: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    #: Set when someone else holds it, naming who, for reporting.
    held_by: str | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()

    # -- the file ----------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        # ValueError as well as OSError: a path with a null byte in it — from a
        # mangled ITSBOB_HOME — raises that rather than an OS error, and a
        # bookkeeping file must never be able to stop the bridge polling.
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self.path)
        except (OSError, ValueError):
            # A home directory that cannot be written is a reason to carry on
            # polling, not to stop: one process answering twice is a smaller
            # failure than none answering at all.
            return False
        return True

    @staticmethod
    def _fresh(data: dict[str, Any], ttl: float, now: float) -> bool:
        return bool(data) and (now - float(data.get("at") or 0)) < ttl

    # -- claiming ----------------------------------------------------------

    def hold(self, *, now: float | None = None) -> bool:
        """Take or renew the claim. ``False`` means someone else has it."""
        now = time.time() if now is None else now
        current = self._read()
        mine = current.get("owner") == self.owner

        if not mine and self._fresh(current, self.ttl, now):
            self.held_by = f"{current.get('role', 'another process')} (pid {current.get('pid')})"
            return False

        self.held_by = None
        if not self._write(
            {
                **current,
                "owner": self.owner,
                "role": self.role,
                "pid": os.getpid(),
                "at": now,
            }
        ):
            # Unwritable: behave as the holder rather than silently going quiet.
            return True
        return True

    def release(self) -> None:
        """Give it up on a clean stop, so a handover is immediate."""
        current = self._read()
        if current.get("owner") != self.owner:
            return
        # The answered ids outlive the claim: whoever picks it up next needs
        # them, or a restart re-answers whatever was in flight.
        current.pop("owner", None)
        current["at"] = 0.0
        self._write(current)

    # -- not answering the same thing twice --------------------------------

    def already_answered(self, message_id: str) -> bool:
        return str(message_id) in set(self._read().get("answered") or [])

    def mark_answered(self, message_id: str) -> None:
        current = self._read()
        answered = [str(x) for x in (current.get("answered") or [])]
        if str(message_id) in answered:
            return
        answered.append(str(message_id))
        current["answered"] = answered[-REMEMBERED:]
        self._write(current)

    def last_answered(self) -> str | None:
        """The most recent message the holder dealt with, for a standby cursor.

        Appended in order, so the tail is the high-water mark. A process in
        standby can follow this instead of polling Discord for the newest id:
        it costs nothing, and it does not skip past messages the holder has
        not answered yet — which is the difference between a clean handover
        and quietly losing whatever was in flight when the holder died.
        """
        answered = self._read().get("answered") or []
        return str(answered[-1]) if answered else None

    def describe(self) -> dict[str, Any]:
        current = self._read()
        return {
            "role": self.role,
            "owner": self.owner,
            "holder": current.get("role"),
            "holder_pid": current.get("pid"),
            "mine": current.get("owner") == self.owner,
            "held_by": self.held_by,
            "answered": len(current.get("answered") or []),
        }
