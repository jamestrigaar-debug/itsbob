"""Append-only JSONL with rotation, shared by the audit log and notifications.

Both were plain ``open(..., "a")`` with nothing bounding them. That is fine for
an afternoon and wrong for the thing this is meant to be: an audit log grows
~350KB per 500 tool calls, and a daemon that runs for months on a laptop has no
reason ever to stop growing it. The failure is quiet — you notice when the disk
does.

Rotation is by size, oldest dropped: ``audit.jsonl`` becomes ``audit.1.jsonl``,
that becomes ``audit.2.jsonl``, and beyond ``keep`` they are deleted. Reading
walks the backups oldest-first so history is continuous across a rotation
rather than appearing to start over.

Writes are guarded by a lock per resolved path, for the same reason the
database is: the daemon and the GUI append to the same files from different
threads, and interleaved partial lines are unparseable forever.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator

__all__ = ["JsonlFile"]

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


class JsonlFile:
    """One JSON object per line, rotated by size."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 5_000_000,
        keep: int = 3,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_bytes = max(0, max_bytes)
        self.keep = max(0, keep)
        self._lock = _lock_for(str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, default=str)
        with self._lock:
            self._rotate_if_needed(len(line) + 2)
            with self.path.open("a", encoding="utf-8") as handle:
                if self._needs_newline():
                    # The previous write was cut short — a crash, a full disk, a
                    # kill mid-append. Without this, the next record is glued
                    # onto the torn one and *both* become unparseable, so one
                    # interruption quietly costs two entries instead of one.
                    handle.write("\n")
                handle.write(line + "\n")

    def _needs_newline(self) -> bool:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return False
        if size == 0:
            return False
        with self.path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"

    def _rotate_if_needed(self, incoming: int) -> None:
        if not self.max_bytes:
            return
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size + incoming <= self.max_bytes:
            return

        if self.keep == 0:
            self.path.unlink(missing_ok=True)
            return
        # Walk down so nothing is overwritten before it has been shifted.
        for index in range(self.keep, 0, -1):
            source = self.path if index == 1 else self._backup(index - 1)
            target = self._backup(index)
            if source.exists():
                target.unlink(missing_ok=True)
                os.replace(source, target)
        self._backup(self.keep + 1).unlink(missing_ok=True)

    def _backup(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.stem}.{index}{self.path.suffix}")

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Entries oldest first, across rotations. Skips unparseable lines."""
        entries: list[dict[str, Any]] = []
        for candidate in [*reversed([self._backup(i) for i in range(1, self.keep + 1)]), self.path]:
            if not candidate.exists():
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # a torn line from a crash; skip it, keep the rest
        return entries[-limit:] if limit is not None else entries

    def size(self) -> int:
        total = 0
        for candidate in [self.path, *(self._backup(i) for i in range(1, self.keep + 1))]:
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def clear(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)
            for index in range(1, self.keep + 2):
                self._backup(index).unlink(missing_ok=True)
