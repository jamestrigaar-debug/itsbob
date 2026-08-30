"""One safe way to talk to SQLite, shared by memory and the task list.

Both stores were opened with ``check_same_thread=False`` and then used from
several threads at once — Flask serves requests on threads, and the daemon
runs alongside it. That combination does not raise reliably; it *loses writes*.
Six threads writing 150 memories landed 34 of them, with
``OperationalError: cannot start a transaction within a transaction`` and a
bare ``SystemError`` from the sqlite3 module along the way.

Three things fix it, and all three belong together:

**One lock per database file, held for the whole statement.** Keyed on the
resolved path at class level, so two :class:`~itsbob.memory.long_term.LongTermMemory`
objects opened on the same file in one process serialize against each other —
which they must, because they share the file, not the object.

**WAL journalling.** Readers no longer block on a writer, which is what
``itsbob serve`` and ``itsbob gui`` running together need. It also survives a
crash mid-write without corrupting the file.

**A busy timeout.** Cross-process contention waits its turn instead of failing
instantly with "database is locked".

The lock is re-entrant because the stores legitimately nest: ``forget()``
deletes from three tables inside one guarded block, and ``prune()`` calls
``forget()`` in a loop.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

__all__ = ["Database", "IN_MEMORY"]

IN_MEMORY = ":memory:"

#: One lock per database *file*, not per object. Two stores opened on the same
#: path in one process must serialize against each other.
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.RLock()
        return lock


class Database:
    """A thread-safe SQLite connection with sane pragmas."""

    def __init__(
        self,
        path: str | Path = IN_MEMORY,
        *,
        schema: str | None = None,
        busy_timeout_ms: int = 10_000,
    ) -> None:
        self.path = str(path)
        self.is_memory = self.path in (IN_MEMORY, "") or "mode=memory" in self.path

        if not self.is_memory:
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
            key = str(Path(self.path).resolve())
        else:
            # Each in-memory database is genuinely private, so it gets its own
            # lock rather than contending with every other test's.
            key = f"memory-{id(self)}"

        self._lock = _lock_for(key)
        # Per instance, not per class. A class-level threading.local is shared
        # by every Database, so a transaction opened on one database while
        # another's was open read the wrong depth, concluded it was nested, and
        # never committed — losing the write silently to every other reader.
        # Which is the exact bug this module exists to prevent.
        self._depth = threading.local()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, timeout=busy_timeout_ms / 1000
        )
        self._conn.row_factory = sqlite3.Row

        with self._lock:
            self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            if not self.is_memory:
                # WAL is a property of the file, so it only needs setting once,
                # but setting it again is free and makes every open correct
                # regardless of which process got there first.
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
            if schema:
                self._conn.executescript(schema)
                self._conn.commit()

    # -- access ------------------------------------------------------------

    @property
    def lock(self) -> threading.RLock:
        """For callers that need to hold several statements together."""
        return self._lock

    @property
    def connection(self) -> sqlite3.Connection:
        """The raw connection. Only touch it inside :attr:`lock`."""
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Everything inside commits together, or none of it does.

        Nesting is safe: only the outermost block commits, so a helper that
        opens its own transaction still composes into a larger one.
        """
        with self._lock:
            depth = getattr(self._depth, "value", 0)
            self._depth.value = depth + 1
            try:
                yield self._conn
            except Exception:
                if depth == 0:
                    self._conn.rollback()
                raise
            else:
                if depth == 0:
                    self._conn.commit()
            finally:
                self._depth.value = depth

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Run one statement under the lock, committing if it wrote."""
        with self.transaction() as conn:
            return conn.execute(sql, params)

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        with self.transaction() as conn:
            return conn.executemany(sql, rows)

    def executescript(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)
            self._conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Read rows, fully materialized inside the lock.

        Returning a cursor would let a caller iterate it after the lock was
        released, which is exactly the pattern that made this unsafe before.
        """
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    def columns(self, table: str) -> set[str]:
        return {row["name"] for row in self.query(f"PRAGMA table_info({table})")}

    def supports_fts5(self) -> bool:
        try:
            self.executescript(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x);"
                "DROP TABLE IF EXISTS _fts5_probe;"
            )
        except sqlite3.OperationalError:
            return False
        return True

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.ProgrammingError:  # pragma: no cover - already closed
                pass

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<Database {self.path}>"
