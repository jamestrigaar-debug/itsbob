"""Long-term memory: SQLite-backed, survives the process.

Retrieval blends relevance, importance and recency (see
:func:`itsbob.memory.base.score_record`). SQL does the coarse filtering, Python
does the scoring — a shape that swaps cleanly for a vector index later without
changing any caller.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .base import (
    DEFAULT_WEIGHTS,
    MemoryKind,
    MemoryRecord,
    RetrievalWeights,
    keyword_relevance,
    rank,
    render_records,
    tokenize,
)

__all__ = ["LongTermMemory"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    kind             TEXT NOT NULL,
    importance       REAL NOT NULL,
    tick             INTEGER NOT NULL,
    created_at       REAL NOT NULL,
    last_access_tick INTEGER NOT NULL,
    access_count     INTEGER NOT NULL,
    salience         REAL NOT NULL,
    tags             TEXT NOT NULL,
    metadata         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_tick ON memories(tick);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
"""


class LongTermMemory:
    """Durable memory store.

    ``database=":memory:"`` gives an ephemeral store with identical semantics,
    which is what the tests and one-off runs use.
    """

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        weights: RetrievalWeights = DEFAULT_WEIGHTS,
        relevance_fn: Callable[[str, MemoryRecord], float] = keyword_relevance,
        candidate_limit: int = 400,
    ) -> None:
        self.database = str(database)
        self.weights = weights
        self.relevance_fn = relevance_fn
        self.candidate_limit = candidate_limit

        if self.database not in (":memory:", "") and "mode=memory" not in self.database:
            Path(self.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.database, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writing -----------------------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._conn.execute(
            """
            INSERT INTO memories
                (id, content, kind, importance, tick, created_at,
                 last_access_tick, access_count, salience, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content,
                importance = excluded.importance,
                last_access_tick = excluded.last_access_tick,
                access_count = excluded.access_count,
                salience = excluded.salience
            """,
            (
                record.id,
                record.content,
                record.kind.value,
                record.importance,
                record.tick,
                record.created_at,
                record.last_access_tick,
                record.access_count,
                record.salience,
                json.dumps(list(record.tags)),
                json.dumps(record.metadata),
            ),
        )
        self._conn.commit()
        return record

    def add_many(self, records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        stored = [self.add(record) for record in records]
        return stored

    def forget(self, record_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def prune(self, max_records: int) -> int:
        """Keep the ``max_records`` most valuable memories; drop the rest.

        Value here is importance first, recency second — the same ordering the
        character would use if asked what it would hate to lose.
        """
        total = len(self)
        if total <= max_records:
            return 0
        cursor = self._conn.execute(
            """
            DELETE FROM memories WHERE id IN (
                SELECT id FROM memories
                ORDER BY importance ASC, last_access_tick ASC, tick ASC
                LIMIT ?
            )
            """,
            (total - max_records,),
        )
        self._conn.commit()
        return cursor.rowcount

    # -- reading -----------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        tick: int | None = None,
        kinds: Sequence[MemoryKind] | None = None,
        min_importance: float = 0.0,
    ) -> list[MemoryRecord]:
        candidates = self._candidates(query, kinds=kinds, min_importance=min_importance)
        now = tick if tick is not None else self.latest_tick
        results = rank(
            candidates,
            query,
            now_tick=now,
            limit=limit,
            weights=self.weights,
            relevance_fn=self.relevance_fn,
        )
        for record in results:
            record.touch(now)
            self._touch(record)
        return results

    def _candidates(
        self,
        query: str,
        *,
        kinds: Sequence[MemoryKind] | None,
        min_importance: float,
    ) -> list[MemoryRecord]:
        """Narrow the table before scoring in Python.

        Keyword hits come first; the recent tail is unioned in so a query with
        no lexical matches still retrieves *something* on importance/recency.
        """
        clauses = ["importance >= ?"]
        params: list[object] = [min_importance]
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            clauses.append(f"kind IN ({placeholders})")
            params.extend(k.value for k in kinds)

        where = " AND ".join(clauses)
        found: dict[str, MemoryRecord] = {}

        terms = tokenize(query)[:8]
        if terms:
            like = " OR ".join("content LIKE ?" for _ in terms)
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE {where} AND ({like}) "
                f"ORDER BY tick DESC LIMIT ?",
                (*params, *(f"%{t}%" for t in terms), self.candidate_limit),
            ).fetchall()
            found.update({row["id"]: _from_row(row) for row in rows})

        remaining = max(0, self.candidate_limit - len(found))
        if remaining:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY tick DESC LIMIT ?",
                (*params, remaining),
            ).fetchall()
            for row in rows:
                found.setdefault(row["id"], _from_row(row))
        return list(found.values())

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (record_id,)
        ).fetchone()
        return _from_row(row) if row else None

    def all(self, *, limit: int | None = None) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories ORDER BY tick ASC, created_at ASC"
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [_from_row(row) for row in self._conn.execute(sql, params)]

    def by_kind(self, kind: MemoryKind, *, limit: int = 20) -> list[MemoryRecord]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE kind = ? ORDER BY tick DESC LIMIT ?",
            (kind.value, limit),
        )
        return [_from_row(row) for row in rows]

    @property
    def latest_tick(self) -> int:
        row = self._conn.execute("SELECT MAX(tick) AS t FROM memories").fetchone()
        return int(row["t"] or 0)

    def _touch(self, record: MemoryRecord) -> None:
        self._conn.execute(
            "UPDATE memories SET last_access_tick = ?, access_count = ? WHERE id = ?",
            (record.last_access_tick, record.access_count, record.id),
        )
        self._conn.commit()

    def render(self, limit: int = 10) -> str:
        return render_records(self.all(limit=limit))

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "LongTermMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        kind=MemoryKind(row["kind"]),
        importance=row["importance"],
        tick=row["tick"],
        created_at=row["created_at"],
        last_access_tick=row["last_access_tick"],
        access_count=row["access_count"],
        salience=row["salience"],
        tags=tuple(json.loads(row["tags"])),
        metadata=json.loads(row["metadata"]),
    )
