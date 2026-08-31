"""Long-term memory: SQLite-backed, hybrid lexical + semantic recall.

Two retrieval signals, fused:

* **Lexical** — SQLite FTS5 with BM25 ranking. Exact, free, and the only thing
  that reliably finds a proper noun, an error code, or a path.
* **Semantic** — cosine similarity over embeddings (see
  :mod:`itsbob.llm.embeddings`). Finds the memory that says the same thing in
  different words, which is most of what a person actually asks for.

Neither alone is good enough: BM25 misses "what do I like to drink?" →
"prefers dark roast coffee", and pure vector search misses
``ERR_CONN_REFUSED``. They are fused on *normalized scores*, then nudged by
importance and wall-clock recency.

Two details that decide whether this works at all:

**Cosine is normalized within the candidate set, not thresholded.** Gemini
embeddings put unrelated text around 0.45–0.60 and related text around
0.65–0.85 — the useful signal is the *spread*, and its absolute position moves
with the model. Min-max scaling across the candidates makes ranking
independent of that calibration; ``min_relative_score`` then trims the tail
using a fraction of the best score rather than a magic constant.

**Relevance dominates.** An early version blended rank-based RRF at 1.0
against importance at 0.35, which on a small corpus meant every query
returned the most *important* memory rather than the most relevant one — RRF
spreads ranks 1..6 across 1/61..1/66, so the retrieval signal was nearly flat
while importance was not. Importance and recency are tiebreakers here, and
weighted like tiebreakers.

Both signals degrade independently and loudly: no FTS5 in this SQLite build
falls back to ``LIKE``; no embedding API falls back to
:class:`~itsbob.llm.embeddings.HashingEmbedder`; no embedder at all leaves
lexical-only recall. :meth:`LongTermMemory.stats` reports which of those is
the case.
"""

from __future__ import annotations

import heapq
import json
import math
import sqlite3
import time
from array import array
from operator import mul
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..store import Database
from .base import (
    DEFAULT_WEIGHTS,
    Horizon,
    MemoryKind,
    MemoryRecord,
    RetrievalWeights,
    Subject,
    keyword_relevance,
    render_records,
    tokenize,
)

__all__ = ["LongTermMemory", "RecallHit", "HybridWeights"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    kind             TEXT NOT NULL,
    importance       REAL NOT NULL,
    tick             INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    last_access_tick INTEGER NOT NULL DEFAULT 0,
    last_access_at   REAL NOT NULL DEFAULT 0,
    access_count     INTEGER NOT NULL DEFAULT 0,
    salience         REAL NOT NULL DEFAULT 0.5,
    tags             TEXT NOT NULL DEFAULT '[]',
    metadata         TEXT NOT NULL DEFAULT '{}',
    source           TEXT NOT NULL DEFAULT 'agent',
    expires_at       REAL,
    subject          TEXT NOT NULL DEFAULT 'user',
    horizon          TEXT NOT NULL DEFAULT 'long'
);
CREATE TABLE IF NOT EXISTS vectors (
    memory_id  TEXT NOT NULL,
    signature  TEXT NOT NULL,
    dims       INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (memory_id, signature)
);
"""

# Kept apart from _SCHEMA and applied *after* _migrate(): a database written by
# an older itsbob has a `memories` table that CREATE TABLE IF NOT EXISTS won't
# touch, so indexing expires_at before the migration adds it fails the whole
# open with "no such column".
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_memories_tick       ON memories(tick);
CREATE INDEX IF NOT EXISTS idx_memories_kind       ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_created    ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_expires    ON memories(expires_at);
CREATE INDEX IF NOT EXISTS idx_memories_subject    ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_memories_horizon    ON memories(horizon, created_at);
CREATE INDEX IF NOT EXISTS idx_vectors_signature   ON vectors(signature);
"""

# A standalone (not external-content) FTS table. External-content tables are
# leaner, but they need triggers kept in lockstep with every write path, and a
# desynced index fails by silently returning nothing. This corpus is small
# enough that storing the text twice is the cheaper trade.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(id UNINDEXED, content, tags, tokenize='unicode61 remove_diacritics 2');
"""

class HybridWeights:
    """How the recall signals are blended.

    ``semantic``/``lexical`` split the relevance budget between the two
    retrieval signals and are renormalized when only one of them produced
    anything, so a lexical-only store doesn't score uniformly lower than a
    hybrid one. ``importance`` and ``recency`` are deliberately small — they
    break ties between comparably relevant memories, they do not outrank
    relevance.
    """

    def __init__(
        self,
        *,
        semantic: float = 0.65,
        lexical: float = 0.35,
        importance: float = 0.12,
        recency: float = 0.10,
        recency_half_life_days: float = 30.0,
        min_relative_score: float = 0.35,
    ) -> None:
        self.semantic = semantic
        self.lexical = lexical
        self.importance = importance
        self.recency = recency
        self.recency_half_life_days = max(0.001, recency_half_life_days)
        #: Drop hits scoring below this fraction of the best hit. Relative, so
        #: it holds across embedding models with different cosine baselines.
        #: 0 keeps everything.
        self.min_relative_score = min_relative_score

    def recency_score(self, created_at: float, now: float) -> float:
        age_days = max(0.0, (now - created_at) / 86400.0)
        return 0.5 ** (age_days / self.recency_half_life_days)


class RecallHit:
    """One recalled memory plus why it surfaced — the explainable half of recall."""

    __slots__ = ("record", "score", "lexical_rank", "vector_rank", "vector_score", "reason")

    def __init__(
        self,
        record: MemoryRecord,
        score: float,
        *,
        lexical_rank: int | None = None,
        vector_rank: int | None = None,
        vector_score: float = 0.0,
    ) -> None:
        self.record = record
        self.score = score
        self.lexical_rank = lexical_rank
        self.vector_rank = vector_rank
        self.vector_score = vector_score
        parts = []
        if lexical_rank is not None:
            parts.append(f"keyword #{lexical_rank + 1}")
        if vector_rank is not None:
            parts.append(f"semantic #{vector_rank + 1} (cos {vector_score:.2f})")
        if not parts:
            parts.append("recency/importance")
        self.reason = ", ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "content": self.record.content,
            "kind": self.record.kind.value,
            "subject": self.record.subject.value,
            "horizon": self.record.horizon.value,
            "importance": round(self.record.importance, 3),
            "tags": list(self.record.tags),
            "created_at": self.record.created_at,
            "score": round(self.score, 4),
            "why": self.reason,
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<RecallHit {self.score:.3f} {self.record.content[:50]!r} ({self.reason})>"


class LongTermMemory:
    """Durable memory store with hybrid recall.

    ``database=":memory:"`` gives an ephemeral store with identical semantics.
    Pass ``embedder=None`` to run lexical-only (no network, no vectors).
    """

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        weights: RetrievalWeights = DEFAULT_WEIGHTS,
        relevance_fn: Callable[[str, MemoryRecord], float] = keyword_relevance,
        candidate_limit: int = 400,
        embedder: Any | None = None,
        hybrid_weights: HybridWeights | None = None,
        vector_scan_limit: int = 5000,
        auto_embed: bool = True,
    ) -> None:
        self.database = str(database)
        self.weights = weights
        self.relevance_fn = relevance_fn
        self.candidate_limit = candidate_limit
        self.embedder = embedder
        self.hybrid = hybrid_weights or HybridWeights()
        #: Cap on vectors compared per recall when numpy is absent. With numpy
        #: the whole table is scanned regardless — the cap exists so pure-Python
        #: cosine stays inside a few tens of milliseconds, not to bound memory.
        self.vector_scan_limit = vector_scan_limit
        self.auto_embed = auto_embed
        self.embed_errors = 0
        self.last_embed_error: str | None = None

        self._db = Database(self.database, schema=_SCHEMA)
        self.database = self._db.path
        self._migrate()
        self._db.executescript(_INDEXES)
        self.fts_enabled = self._try_enable_fts()

        self._np = _try_numpy()
        self._vec_cache: tuple[str, list[str], Any] | None = None
        self._vec_norms: list[float] | None = None

    # -- schema ----------------------------------------------------------------

    def _migrate(self) -> None:
        """Add columns introduced after the original tick-only schema.

        Old databases (written by the character simulation) open and keep
        working; they simply have ``source='agent'`` and no expiry until
        something rewrites them.
        """
        existing = self._db.columns("memories")
        for column, ddl in (
            ("last_access_at", "ALTER TABLE memories ADD COLUMN last_access_at REAL NOT NULL DEFAULT 0"),
            ("source", "ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'agent'"),
            ("expires_at", "ALTER TABLE memories ADD COLUMN expires_at REAL"),
            ("subject", "ALTER TABLE memories ADD COLUMN subject TEXT NOT NULL DEFAULT 'user'"),
            ("horizon", "ALTER TABLE memories ADD COLUMN horizon TEXT NOT NULL DEFAULT 'long'"),
        ):
            if column not in existing:
                self._db.execute(ddl)

    def _try_enable_fts(self) -> bool:
        try:
            self._db.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            return False  # SQLite built without FTS5; LIKE fallback covers it
        # Backfill anything written before the index existed (or by an older
        # itsbob), so an upgrade doesn't leave half the corpus unsearchable.
        if int(self._db.scalar("SELECT COUNT(*) FROM memories_fts", default=0)) == 0:
            self._db.executemany(
                "INSERT INTO memories_fts (id, content, tags) VALUES (?, ?, ?)",
                [
                    (r["id"], r["content"], " ".join(json.loads(r["tags"] or "[]")))
                    for r in self._db.query("SELECT id, content, tags FROM memories")
                ],
            )
        return True

    # -- writing ---------------------------------------------------------------

    def add(self, record: MemoryRecord, *, embed: bool | None = None) -> MemoryRecord:
        source = str(record.metadata.get("source", "agent"))
        expires_at = record.expires_at
        with self._db.transaction() as conn:
            conn.execute(
                """
            INSERT OR REPLACE INTO memories (
                id, content, kind, importance, tick, created_at,
                last_access_tick, last_access_at, access_count, salience,
                tags, metadata, source, expires_at, subject, horizon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.id,
                    record.content,
                    record.kind.value,
                    record.importance,
                    record.tick,
                    record.created_at,
                    record.last_access_tick,
                    record.metadata.get("last_access_at", record.created_at),
                    record.access_count,
                    record.salience,
                    json.dumps(list(record.tags)),
                    json.dumps(record.metadata),
                    source,
                    float(expires_at) if expires_at is not None else None,
                    record.subject.value,
                    record.horizon.value,
                ),
            )
            if self.fts_enabled:
                conn.execute("DELETE FROM memories_fts WHERE id = ?", (record.id,))
                conn.execute(
                    "INSERT INTO memories_fts (id, content, tags) VALUES (?, ?, ?)",
                    (record.id, record.content, " ".join(record.tags)),
                )

        if embed if embed is not None else self.auto_embed:
            self._embed_records([record])
        return record

    def add_many(self, records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        stored = [self.add(record, embed=False) for record in records]
        if self.auto_embed and stored:
            self._embed_records(stored)
        return stored

    def update(self, record_id: str, **fields: Any) -> bool:
        """Patch one record in place. Re-embeds if the content changed."""
        allowed = {
            "content", "importance", "tags", "metadata", "kind",
            "expires_at", "subject", "horizon",
        }
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("tags", "metadata"):
                value = json.dumps(list(value) if key == "tags" else dict(value))
            elif key == "kind":
                value = MemoryKind(value).value
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            return False
        params.append(record_id)
        with self._db.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params
            )
            if not cursor.rowcount:
                return False
            reindex = "content" in fields or "tags" in fields
            if reindex:
                conn.execute("DELETE FROM vectors WHERE memory_id = ?", (record_id,))
                if self.fts_enabled:
                    conn.execute("DELETE FROM memories_fts WHERE id = ?", (record_id,))

        if reindex:
            record = self.get(record_id)
            if record is not None:
                if self.fts_enabled:
                    self._db.execute(
                        "INSERT INTO memories_fts (id, content, tags) VALUES (?, ?, ?)",
                        (record_id, record.content, " ".join(record.tags)),
                    )
                self._invalidate_vector_cache()
                if self.auto_embed:
                    self._embed_records([record])
        return True

    def forget(self, record_id: str) -> bool:
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
            conn.execute("DELETE FROM vectors WHERE memory_id = ?", (record_id,))
            if self.fts_enabled:
                conn.execute("DELETE FROM memories_fts WHERE id = ?", (record_id,))
            removed = cursor.rowcount > 0
        self._invalidate_vector_cache()
        return removed

    def expire(self, *, now: float | None = None) -> int:
        """Drop records past their ``expires_at``. Returns how many went."""
        now = time.time() if now is None else now
        ids = [
            row["id"]
            for row in self._db.query(
                "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
        ]
        for record_id in ids:
            self.forget(record_id)
        return len(ids)

    #: How many short-horizon rows survive. "A few replies" in the request that
    #: prompted this: enough to carry a thread, not enough to become the corpus.
    short_term_capacity: int = 24
    #: And how long one lives regardless of how quiet it has been since.
    short_term_ttl_seconds: float = 6 * 3600.0

    def prune_short_term(self, *, keep: int | None = None, now: float | None = None) -> int:
        """Expire the short-horizon working set, by clock then by count.

        Two limits rather than one because they fail differently. The clock
        alone leaves a burst of forty rows from one busy hour all live at once;
        the count alone lets a single row from last Tuesday sit in the working
        set forever because nothing has pushed it out.

        Long-horizon rows are never touched here — the whole point of promoting
        something to long-term is that this method cannot reach it.
        """
        now = time.time() if now is None else now
        keep = self.short_term_capacity if keep is None else max(0, keep)
        cutoff = now - self.short_term_ttl_seconds

        doomed = [
            row["id"]
            for row in self._db.query(
                "SELECT id FROM memories WHERE horizon = 'short' AND created_at < ?",
                (cutoff,),
            )
        ]
        survivors = [
            row["id"]
            for row in self._db.query(
                "SELECT id FROM memories WHERE horizon = 'short' AND created_at >= ? "
                "ORDER BY created_at DESC",
                (cutoff,),
            )
        ]
        doomed.extend(survivors[keep:])
        for record_id in doomed:
            self.forget(record_id)
        return len(doomed)

    #: A short-horizon row recalled this many times has proved it matters.
    #: Two rather than one: surfacing once can be the query being vague.
    promote_after_recalls: int = 2
    #: Or written down as clearly mattering in the first place.
    promote_above_importance: float = 0.85

    def consolidate(self, *, now: float | None = None) -> list[str]:
        """Promote short-horizon rows that have earned permanence.

        Being recalled *again* is the signal, and it is a good one: something
        that surfaced in a later conversation is being used, which is the only
        evidence available that a memory was worth keeping. The alternative —
        deciding at write time — is what fills a store with forty rows nobody
        ever reads, because at write time everything looks like it might matter.

        Run once per turn, before pruning, so a row about to be dropped gets its
        chance first.
        """
        rows = self._db.query(
            "SELECT id FROM memories WHERE horizon = 'short' "
            "AND (access_count >= ? OR importance >= ?)",
            (self.promote_after_recalls, self.promote_above_importance),
        )
        promoted = [row["id"] for row in rows]
        for record_id in promoted:
            self.promote(record_id)
        return promoted

    def promote(self, record_id: str, *, importance: float | None = None) -> bool:
        """Move a short-horizon row into long-term memory.

        Consolidation is a real event, not a side effect of scoring: something
        that started as "what we are doing right now" turned out to be worth
        keeping, and clearing ``expires_at`` is what makes that stick.
        """
        fields: dict[str, Any] = {"horizon": Horizon.LONG.value, "expires_at": None}
        if importance is not None:
            fields["importance"] = importance
        return self.update(record_id, **fields)

    def by_subject(self, subject: Subject, *, limit: int = 20) -> list[MemoryRecord]:
        rows = self._db.query(
            "SELECT * FROM memories WHERE subject = ? ORDER BY created_at DESC LIMIT ?",
            (Subject.coerce(subject).value, limit),
        )
        return [_from_row(row) for row in rows]

    def counts_by(self, column: str) -> dict[str, int]:
        """Row counts grouped by ``subject`` or ``horizon``, for the status panel."""
        if column not in ("subject", "horizon", "kind"):
            raise ValueError(f"cannot group by {column!r}")
        return {
            str(row[column]): int(row["n"])
            for row in self._db.query(
                f"SELECT {column}, COUNT(*) AS n FROM memories GROUP BY {column}"  # noqa: S608
            )
        }

    def prune(self, max_records: int) -> int:
        """Keep the ``max_records`` most valuable; drop the rest.

        Value is importance first, then recency — the same ordering recall
        falls back to when a query matches nothing.
        """
        total = len(self)
        if total <= max_records:
            return 0
        doomed = [
            row["id"]
            for row in self._db.query(
                "SELECT id FROM memories ORDER BY importance ASC, created_at ASC LIMIT ?",
                (total - max_records,),
            )
        ]
        for record_id in doomed:
            self.forget(record_id)
        return len(doomed)

    # -- embeddings ------------------------------------------------------------

    def _embed_records(self, records: Sequence[MemoryRecord]) -> int:
        """Embed and store vectors. Never raises — recall survives without them."""
        if self.embedder is None or not records:
            return 0
        texts = [self._embed_text(r) for r in records]
        try:
            vectors = self.embedder.embed(texts)
            if len(vectors) != len(records):
                # Checked here, inside the guard, rather than by strict=True on
                # the zip below: a provider returning the wrong number of
                # vectors must be recorded as an embedding failure like any
                # other, not raised through a write. The memory is the thing
                # being protected; the vector is an optimization.
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(records)} records"
                )
        except Exception as exc:  # noqa: BLE001 - lexical recall still works
            self.embed_errors += 1
            self.last_embed_error = f"{type(exc).__name__}: {exc}"[:200]
            return 0
        signature = self.embedder.signature
        now = time.time()
        self._db.executemany(
            "INSERT OR REPLACE INTO vectors (memory_id, signature, dims, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (r.id, signature, len(v), array("f", v).tobytes(), now)
                for r, v in zip(records, vectors, strict=True)  # length checked above
            ],
        )
        self._invalidate_vector_cache()
        return len(records)

    @staticmethod
    def _embed_text(record: MemoryRecord) -> str:
        """What actually gets vectorized.

        Tags carry meaning the content often leaves implicit ("preference",
        "deadline"), so they go in — but after the content, so they never
        dominate a short memory.
        """
        tags = " ".join(record.tags)
        return f"{record.content}\n[{record.kind.value}] {tags}".strip()

    def reindex(self, *, batch_size: int = 64) -> int:
        """Embed every record missing a vector for the *current* signature.

        Run this after changing embedding model or dimensions: old vectors
        carry the old signature, are ignored by recall rather than
        mis-compared, and this is what replaces them.
        """
        if self.embedder is None:
            return 0
        signature = self.embedder.signature
        rows = self._db.query(
            "SELECT m.* FROM memories m LEFT JOIN vectors v "
            "  ON v.memory_id = m.id AND v.signature = ? "
            "WHERE v.memory_id IS NULL",
            (signature,),
        )
        done = 0
        for start in range(0, len(rows), batch_size):
            batch = [_from_row(r) for r in rows[start : start + batch_size]]
            done += self._embed_records(batch)
        return done

    def _invalidate_vector_cache(self) -> None:
        self._vec_cache = None
        self._vec_norms = None

    def _load_vectors(self, signature: str) -> tuple[list[str], Any]:
        """Ids and vectors for one signature, cached until the next write."""
        if self._vec_cache is not None and self._vec_cache[0] == signature:
            return self._vec_cache[1], self._vec_cache[2]

        sql = (
            "SELECT v.memory_id, v.vector FROM vectors v "
            "JOIN memories m ON m.id = v.memory_id "
            "WHERE v.signature = ? ORDER BY m.created_at DESC"
        )
        params: tuple[Any, ...] = (signature,)
        if self._np is None:
            sql += " LIMIT ?"
            params = (signature, self.vector_scan_limit)

        ids: list[str] = []
        raw: list[bytes] = []
        for row in self._db.query(sql, params):
            ids.append(row["memory_id"])
            raw.append(row["vector"])

        if self._np is not None:
            matrix = (
                self._np.frombuffer(b"".join(raw), dtype=self._np.float32).reshape(len(ids), -1)
                if ids
                else self._np.zeros((0, 0), dtype=self._np.float32)
            )
        else:
            # array('f'), not .tolist(). A list of 768 Python floats costs
            # ~30KB (24 bytes per float object plus a pointer); the same
            # numbers in an array('f') cost ~3KB. At 400 memories that was the
            # difference between 10.7MB and ~1MB per search, and it scaled
            # linearly into hundreds of megabytes on a store that had been
            # used for a while.
            matrix = [(v := array("f"), v.frombytes(blob), v)[2] for blob in raw]
            # Norms never change once a vector is stored, so computing them
            # here means each search does one dot product per candidate rather
            # than a dot and two norms.
            self._vec_norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in matrix]

        self._vec_cache = (signature, ids, matrix)
        return ids, matrix

    def _vector_ranking(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        """(memory_id, cosine) best first. Empty if embeddings are unavailable."""
        if self.embedder is None:
            return []
        try:
            query_vector = self.embedder.embed([query])[0]
        except Exception as exc:  # noqa: BLE001
            self.embed_errors += 1
            self.last_embed_error = f"{type(exc).__name__}: {exc}"[:200]
            return []

        ids, matrix = self._load_vectors(self.embedder.signature)
        if not ids:
            return []

        if self._np is not None:
            np = self._np
            q = np.asarray(query_vector, dtype=np.float32)
            if matrix.shape[1] != q.shape[0]:
                return []  # width changed under us; reindex will fix it
            norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(q))
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = np.where(norms > 0, matrix @ q / norms, 0.0)
            order = np.argsort(-sims)[:top_k]
            return [(ids[int(i)], float(sims[int(i)])) for i in order]

        qn = math.sqrt(sum(x * x for x in query_vector))
        if qn == 0.0:
            return []
        width = len(query_vector)
        norms = getattr(self, "_vec_norms", None) or [1.0] * len(ids)
        scored: list[tuple[str, float]] = []
        for memory_id, vector, vn in zip(ids, matrix, norms, strict=True):
            if len(vector) != width:
                continue
            dot = sum(map(mul, vector, query_vector))
            scored.append((memory_id, dot / (vn * qn)))
        # nlargest beats a full sort when top_k is small relative to the
        # candidate set, which it always is here.
        return heapq.nlargest(top_k, scored, key=lambda pair: pair[1])

    # -- reading ---------------------------------------------------------------

    def _lexical_ranking(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        """(memory_id, score) best first, where a higher score is a better match.

        Returns BM25 *scores* rather than positions on purpose. Scoring by
        reciprocal rank instead (1, 1/2, 1/3, ...) makes the gap between the
        first and second hit larger than the entire importance and recency
        budget put together, so two equally-good matches get ordered by
        whichever the index happened to return first and nothing downstream
        can break the tie. Equal matches must produce equal scores.
        """
        terms = tokenize(query)
        if not terms:
            return []
        if self.fts_enabled:
            # Quote every term: an unescaped '-' or ':' is FTS5 syntax and
            # raises rather than matching. OR keeps it a ranked search rather
            # than an all-terms filter.
            match = " OR ".join(f'"{t}"' for t in terms[:16])
            try:
                rows = self._db.query(
                    "SELECT id, bm25(memories_fts, 1.0, 0.5) AS score FROM memories_fts "
                    "WHERE memories_fts MATCH ? ORDER BY score LIMIT ?",
                    (match, top_k),
                )
                # SQLite's bm25() is negated (more negative = better match),
                # so flip it into "bigger is better" like every other signal.
                return [(row["id"], -float(row["score"])) for row in rows]
            except sqlite3.OperationalError:
                pass  # malformed query for this tokenizer; fall through to LIKE

        # LIKE has no notion of match quality, so every hit scores the same
        # and importance/recency decide the order among them.
        like = " OR ".join("content LIKE ?" for _ in terms[:8])
        rows = self._db.query(
            f"SELECT id FROM memories WHERE {like} ORDER BY created_at DESC LIMIT ?",
            (*(f"%{t}%" for t in terms[:8]), top_k),
        )
        return [(row["id"], 1.0) for row in rows]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        kinds: Sequence[MemoryKind] | None = None,
        min_importance: float = 0.0,
        tags: Sequence[str] | None = None,
        now: float | None = None,
        touch: bool = True,
    ) -> list[RecallHit]:
        """Hybrid recall, with the reason each hit surfaced.

        :meth:`recall` is the plain-records version of this; use ``search``
        when you want to show *why* something was remembered.
        """
        now = time.time() if now is None else now
        pool = max(limit * 8, 40)

        lexical = self._lexical_ranking(query, top_k=pool)
        vector = self._vector_ranking(query, top_k=pool) if query.strip() else []

        lex_rank = {mid: i for i, (mid, _) in enumerate(lexical)}
        lex_score = dict(lexical)
        vec_rank = {mid: i for i, (mid, _) in enumerate(vector)}
        vec_score = dict(vector)

        candidates = set(lex_rank) | set(vec_rank)
        fallback = False
        if not candidates:
            # Nothing matched either way — fall back to what mattered and what
            # just happened, so recall is never simply empty.
            fallback = True
            candidates = {
                row["id"]
                for row in self._db.query(
                    "SELECT id FROM memories ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (pool,),
                )
            }

        records = self._fetch(candidates, kinds=kinds, min_importance=min_importance, tags=tags)
        if not records:
            return []

        # Min-max scale cosine across the surviving candidates. Absolute cosine
        # is not comparable between embedding models (and barely between
        # queries); its spread within one result set is.
        sem_present = [vec_score[r.id] for r in records if r.id in vec_score]
        lex_present = [lex_score[r.id] for r in records if r.id in lex_score]
        sem_lo, sem_hi = _span(sem_present)
        lex_lo, lex_hi = _span(lex_present)

        # Split the relevance budget over whichever signals actually fired, so
        # a lexical-only store isn't uniformly penalized against a hybrid one.
        w_sem = self.hybrid.semantic if sem_present else 0.0
        w_lex = self.hybrid.lexical if lex_present else 0.0
        total_w = w_sem + w_lex
        if total_w:
            w_sem, w_lex = w_sem / total_w, w_lex / total_w

        hits: list[RecallHit] = []
        for record in records:
            semantic = _scale(vec_score.get(record.id), sem_lo, sem_hi)
            lexical_score = _scale(lex_score.get(record.id), lex_lo, lex_hi)
            relevance = w_sem * semantic + w_lex * lexical_score
            score = (
                relevance
                + self.hybrid.importance * record.importance
                + self.hybrid.recency * self.hybrid.recency_score(record.created_at, now)
            )
            hits.append(
                RecallHit(
                    record,
                    score,
                    lexical_rank=lex_rank.get(record.id),
                    vector_rank=vec_rank.get(record.id),
                    vector_score=vec_score.get(record.id, 0.0),
                )
            )

        # Stable tiebreak keeps repeated recalls (and seeded runs) identical.
        hits.sort(key=lambda h: (-h.score, -h.record.created_at, h.record.id))

        cutoff = self.hybrid.min_relative_score
        if cutoff > 0 and hits and not fallback:
            floor = hits[0].score * cutoff
            hits = [h for h in hits if h.score >= floor]
        hits = hits[:limit]

        if touch and hits:
            self._touch_many([h.record for h in hits], now=now)
        return hits

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        tick: int | None = None,
        kinds: Sequence[MemoryKind] | None = None,
        min_importance: float = 0.0,
    ) -> list[MemoryRecord]:
        """Records only — the :class:`~itsbob.memory.base.MemoryStore` contract.

        ``tick`` is honoured for callers still on the simulation's tick clock;
        it only affects the ``last_access_tick`` stamped on what comes back.
        """
        hits = self.search(query, limit=limit, kinds=kinds, min_importance=min_importance)
        if tick is not None:
            with self._db.transaction() as conn:
                for hit in hits:
                    hit.record.touch(tick)
                    conn.execute(
                        "UPDATE memories SET last_access_tick = ? WHERE id = ?",
                        (hit.record.last_access_tick, hit.record.id),
                    )
        return [hit.record for hit in hits]

    def _fetch(
        self,
        ids: Iterable[str],
        *,
        kinds: Sequence[MemoryKind] | None = None,
        min_importance: float = 0.0,
        tags: Sequence[str] | None = None,
    ) -> list[MemoryRecord]:
        ids = list(ids)
        if not ids:
            return []
        out: list[MemoryRecord] = []
        # SQLite caps host parameters (999 on older builds), so chunk rather
        # than assuming the candidate set is small.
        for start in range(0, len(ids), 400):
            chunk = ids[start : start + 400]
            clauses = [f"id IN ({', '.join('?' for _ in chunk)})", "importance >= ?"]
            params: list[Any] = [*chunk, min_importance]
            if kinds:
                clauses.append(f"kind IN ({', '.join('?' for _ in kinds)})")
                params.extend(k.value if isinstance(k, MemoryKind) else str(k) for k in kinds)
            rows = self._db.query(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)}", params
            )
            out.extend(_from_row(row) for row in rows)
        if tags:
            wanted = {t.strip().lower() for t in tags if t.strip()}
            out = [r for r in out if wanted & set(r.tags)]
        return out

    def _touch_many(self, records: Sequence[MemoryRecord], *, now: float) -> None:
        for record in records:
            record.access_count += 1
            record.metadata["last_access_at"] = now
        self._db.executemany(
            "UPDATE memories SET access_count = ?, last_access_at = ? WHERE id = ?",
            [(r.access_count, now, r.id) for r in records],
        )

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self._db.one("SELECT * FROM memories WHERE id = ?", (record_id,))
        return _from_row(row) if row else None

    def all(self, *, limit: int | None = None) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories ORDER BY created_at ASC, tick ASC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [_from_row(row) for row in self._db.query(sql, params)]

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        return [
            _from_row(row)
            for row in self._db.query(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        ]

    def by_kind(self, kind: MemoryKind, *, limit: int = 20) -> list[MemoryRecord]:
        rows = self._db.query(
            "SELECT * FROM memories WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
            (kind.value, limit),
        )
        return [_from_row(row) for row in rows]

    def by_tag(self, tag: str, *, limit: int = 20) -> list[MemoryRecord]:
        needle = f'"{tag.strip().lower()}"'
        rows = self._db.query(
            "SELECT * FROM memories WHERE tags LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{needle}%", limit),
        )
        return [_from_row(row) for row in rows]

    @property
    def latest_tick(self) -> int:
        return int(self._db.scalar("SELECT MAX(tick) FROM memories", default=0) or 0)

    def render(self, limit: int = 10) -> str:
        return render_records(self.recent(limit=limit))

    def stats(self) -> dict[str, Any]:
        """What recall can actually do right now, and what it's falling back to."""
        signature = self.embedder.signature if self.embedder is not None else None
        # Whether the vectors came from a real embedding model or the offline
        # hashing fallback. Both produce recall; only one produces *semantic*
        # recall, and a caller reporting "semantic recall: on" for the fallback
        # would be overstating what the store can do.
        active = getattr(self.embedder, "active", self.embedder)
        offline = getattr(active, "name", None) == "hashing"
        vectorized = 0
        if signature:
            vectorized = int(
                self._db.scalar(
                    "SELECT COUNT(*) FROM vectors WHERE signature = ?", (signature,), default=0
                )
            )
        total = len(self)
        return {
            "database": self.database,
            "records": total,
            "fts5": self.fts_enabled,
            "numpy": self._np is not None,
            "embedder": signature,
            "embedded": vectorized,
            "unembedded": max(0, total - vectorized) if signature else total,
            "embed_errors": self.embed_errors,
            "last_embed_error": self.last_embed_error,
            "offline_embedder": offline,
            "semantic_recall": bool(signature and vectorized and not offline),
            "degraded": bool(getattr(self.embedder, "degraded", False)),
            "by_subject": self.counts_by("subject"),
            "by_horizon": self.counts_by("horizon"),
        }

    def __len__(self) -> int:
        return int(self._db.scalar("SELECT COUNT(*) FROM memories", default=0))

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "LongTermMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _span(values: Sequence[float]) -> tuple[float, float]:
    return (min(values), max(values)) if values else (0.0, 0.0)


def _scale(value: float | None, lo: float, hi: float) -> float:
    """Min-max a raw signal into ``[0, 1]``; absent means 0, all-tied means 1.

    Equal raw scores must map to equal scaled scores — that is what lets
    importance and recency act as tiebreakers instead of being overruled by an
    arbitrary index ordering.
    """
    if value is None:
        return 0.0
    if hi - lo <= 0:
        return 1.0
    return (value - lo) / (hi - lo)


def _try_numpy() -> Any | None:
    try:
        import numpy  # noqa: PLC0415 - optional accelerator
    except ImportError:
        return None
    return numpy


def _from_row(row: sqlite3.Row) -> MemoryRecord:
    keys = row.keys()
    metadata = json.loads(row["metadata"] or "{}")
    if "source" in keys:
        metadata.setdefault("source", row["source"])
    if "expires_at" in keys and row["expires_at"] is not None:
        metadata.setdefault("expires_at", row["expires_at"])
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        kind=MemoryKind.coerce(row["kind"]),
        subject=Subject.coerce(row["subject"] if "subject" in keys else None),
        horizon=Horizon.coerce(row["horizon"] if "horizon" in keys else None),
        expires_at=row["expires_at"] if "expires_at" in keys else None,
        importance=row["importance"],
        tick=row["tick"],
        created_at=row["created_at"],
        last_access_tick=row["last_access_tick"],
        access_count=row["access_count"],
        salience=row["salience"],
        tags=tuple(json.loads(row["tags"] or "[]")),
        metadata=metadata,
    )
