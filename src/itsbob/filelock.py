"""Small cross-process locks for the assistant's local state files.

The project deliberately keeps its runtime state as ordinary files and SQLite
databases.  SQLite supplies its own locking; JSON files need a tiny equivalent
when the GUI and daemon are separate processes.  This module uses only the
standard library and also serializes threads in one process (``flock`` alone
does not provide that guarantee on every platform).
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import threading
from pathlib import Path
from typing import Iterator

__all__ = ["exclusive_file_lock"]


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = _THREAD_LOCKS[key] = threading.Lock()
        return lock


def _lock(handle) -> None:
    """Lock one byte on Windows or the whole file on POSIX."""
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        import msvcrt

        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    """Hold an exclusive, process-safe lock associated with ``path``.

    The lock is advisory, as is standard for local POSIX applications: every
    itsbob writer uses it, while a manually edited file remains possible.
    """
    lock_path = Path(path).expanduser()
    with _thread_lock(lock_path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            _lock(handle)
            try:
                yield
            finally:
                _unlock(handle)
