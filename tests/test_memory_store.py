"""Hybrid recall, schema migration, and the failure modes that matter.

Deliberately offline: every test drives :class:`HashingEmbedder` or no
embedder at all, so the suite proves the *fusion* logic rather than any
particular vendor's embedding quality.
"""

from __future__ import annotations

import json
import sqlite3
import time


from itsbob.llm.embeddings import Embedder, EmbeddingRouter, HashingEmbedder
from itsbob.memory.base import MemoryKind, MemoryRecord
from itsbob.memory.long_term import HybridWeights, LongTermMemory


def _store(**kwargs) -> LongTermMemory:
    kwargs.setdefault("embedder", EmbeddingRouter([HashingEmbedder(dims=128)]))
    return LongTermMemory(":memory:", **kwargs)


def _fill(store: LongTermMemory, items) -> None:
    store.add_many(
        [
            MemoryRecord(content=c, kind=k, importance=i, tags=t)
            for c, k, i, t in items
        ]
    )


SAMPLE = [
    ("I prefer dark roast coffee", MemoryKind.FACT, 0.5, ("preference",)),
    ("The deploy failed with ERR_CONN_REFUSED", MemoryKind.OBSERVATION, 0.9, ("bug",)),
    ("Rufus is a border collie", MemoryKind.FACT, 0.3, ("people",)),
]


def test_exact_token_recall_beats_an_unrelated_but_important_memory():
    """The regression that motivated dropping rank-fusion for score-fusion."""
    store = _store()
    _fill(store, SAMPLE)
    hits = store.search("ERR_CONN_REFUSED", limit=3)
    assert hits[0].record.content.startswith("The deploy failed")
    assert hits[0].lexical_rank == 0


def test_importance_does_not_outrank_relevance():
    store = _store()
    _fill(store, SAMPLE)
    top = store.search("border collie", limit=1)[0]
    assert top.record.content == "Rufus is a border collie"  # importance 0.3, still wins


def test_importance_breaks_ties_between_equally_relevant_memories():
    store = _store()
    _fill(
        store,
        [
            ("alpha beta gamma", MemoryKind.FACT, 0.1, ()),
            ("alpha beta gamma", MemoryKind.FACT, 0.9, ()),
        ],
    )
    assert store.search("alpha beta gamma", limit=1)[0].record.importance == 0.9


def test_relative_cutoff_trims_the_irrelevant_tail():
    store = _store()
    _fill(store, SAMPLE)
    assert len(store.search("ERR_CONN_REFUSED", limit=10)) < len(SAMPLE)


def test_cutoff_of_zero_keeps_everything():
    store = _store(hybrid_weights=HybridWeights(min_relative_score=0.0))
    _fill(store, SAMPLE)
    assert len(store.search("ERR_CONN_REFUSED", limit=10)) == len(SAMPLE)


def test_recall_is_never_empty_when_nothing_matches():
    """A query with no lexical or semantic hits still returns the best guess."""
    store = LongTermMemory(":memory:", embedder=None)
    _fill(store, SAMPLE)
    assert store.search("zzzzz", limit=2)


def test_empty_store_returns_nothing_rather_than_raising():
    assert _store().search("anything", limit=5) == []


def test_lexical_only_store_still_ranks():
    store = LongTermMemory(":memory:", embedder=None)
    _fill(store, SAMPLE)
    hits = store.search("coffee", limit=1)
    assert hits[0].record.content == "I prefer dark roast coffee"
    assert hits[0].vector_rank is None


def test_fts_query_with_punctuation_does_not_raise():
    """Bare '-' and ':' are FTS5 operators; unquoted terms used to blow up."""
    store = _store()
    _fill(store, SAMPLE)
    for query in ["ERR-CONN: refused", "a - b", 'quote " here', "OR AND NOT"]:
        assert isinstance(store.search(query, limit=2), list)


def test_vectors_are_isolated_by_signature():
    """Vectors from a different model must be ignored, never compared."""
    store = _store()
    _fill(store, SAMPLE)
    assert store.stats()["embedded"] == 3
    store.embedder = EmbeddingRouter([HashingEmbedder(dims=64)])  # different width
    assert store.stats()["embedded"] == 0
    assert store.search("coffee", limit=1)  # degrades to lexical, does not crash


def test_reindex_backfills_the_current_signature():
    store = _store()
    _fill(store, SAMPLE)
    store.embedder = EmbeddingRouter([HashingEmbedder(dims=64)])
    assert store.reindex() == 3
    assert store.stats()["embedded"] == 3


def test_add_without_embedder_still_stores_and_recalls():
    store = LongTermMemory(":memory:", embedder=None)
    record = store.add(MemoryRecord(content="hello world"))
    assert store.get(record.id) is not None
    assert store.stats()["semantic_recall"] is False


def test_embedding_failure_does_not_lose_the_memory():
    class _Boom(Embedder):
        def __init__(self):
            super().__init__(name="boom", model="x", dims=8)

        def embed(self, texts):
            raise RuntimeError("no network")

    store = LongTermMemory(":memory:", embedder=_Boom())
    store.add(MemoryRecord(content="important thing"))
    assert len(store) == 1
    assert store.stats()["embed_errors"] == 1
    assert store.search("important", limit=1)[0].record.content == "important thing"


def test_forget_removes_record_vector_and_index():
    store = _store()
    record = store.add(MemoryRecord(content="ephemeral note"))
    assert store.forget(record.id) is True
    assert store.forget(record.id) is False
    assert len(store) == 0
    assert store.stats()["embedded"] == 0
    assert store.search("ephemeral", limit=5) == []


def test_update_rewrites_content_index_and_vector():
    store = _store()
    record = store.add(MemoryRecord(content="likes tea"))
    assert store.update(record.id, content="likes espresso") is True
    assert store.get(record.id).content == "likes espresso"
    assert store.search("espresso", limit=1)[0].record.id == record.id
    # The old wording is gone from the index: "tea" no longer matches lexically.
    assert store.search("tea", limit=1)[0].lexical_rank is None


def test_update_rejects_unknown_fields():
    store = _store()
    record = store.add(MemoryRecord(content="x"))
    assert store.update(record.id, nonsense=1) is False
    assert store.update("no-such-id", content="y") is False


def test_expire_drops_only_lapsed_records():
    store = _store()
    now = time.time()
    store.add(MemoryRecord(content="gone", metadata={"expires_at": now - 1}))
    store.add(MemoryRecord(content="stays", metadata={"expires_at": now + 3600}))
    store.add(MemoryRecord(content="forever"))
    assert store.expire() == 1
    assert {r.content for r in store.all()} == {"stays", "forever"}


def test_prune_keeps_the_most_important():
    store = _store()
    _fill(store, SAMPLE)
    assert store.prune(2) == 1
    assert "Rufus is a border collie" not in {r.content for r in store.all()}


def test_prune_is_a_noop_below_the_cap():
    store = _store()
    _fill(store, SAMPLE)
    assert store.prune(10) == 0


def test_recency_favours_the_newer_of_two_identical_memories():
    store = _store()
    old = MemoryRecord(content="same text", created_at=time.time() - 400 * 86400)
    new = MemoryRecord(content="same text")
    store.add_many([old, new])
    assert store.search("same text", limit=1)[0].record.id == new.id


def test_kind_and_importance_filters_apply():
    store = _store()
    _fill(store, SAMPLE)
    assert all(h.record.kind is MemoryKind.FACT for h in store.search("a", limit=5, kinds=[MemoryKind.FACT]))
    assert all(h.record.importance >= 0.8 for h in store.search("a", limit=5, min_importance=0.8))


def test_tag_filter_applies():
    store = _store()
    _fill(store, SAMPLE)
    assert {h.record.content for h in store.search("a", limit=5, tags=["bug"])} == {
        "The deploy failed with ERR_CONN_REFUSED"
    }


def test_search_records_why_each_hit_surfaced():
    store = _store()
    _fill(store, SAMPLE)
    hit = store.search("coffee", limit=1)[0]
    assert "keyword" in hit.reason or "semantic" in hit.reason
    assert set(hit.as_dict()) >= {"id", "content", "score", "why"}


def test_recall_returns_plain_records_and_stamps_the_tick():
    store = _store()
    _fill(store, SAMPLE)
    records = store.recall("coffee", limit=1, tick=42)
    assert isinstance(records[0], MemoryRecord)
    assert store.get(records[0].id).last_access_tick == 42


def test_search_touches_access_counts():
    store = _store()
    record = store.add(MemoryRecord(content="touched"))
    store.search("touched", limit=1)
    assert store.get(record.id).access_count == 1


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "m.sqlite"
    with LongTermMemory(path, embedder=HashingEmbedder(dims=64)) as store:
        store.add(MemoryRecord(content="durable fact about llamas"))
    with LongTermMemory(path, embedder=HashingEmbedder(dims=64)) as store:
        assert len(store) == 1
        assert store.search("llamas", limit=1)[0].record.content.endswith("llamas")
        assert store.stats()["embedded"] == 1


def test_opens_a_legacy_tick_only_database(tmp_path):
    """Databases written by the character simulation must keep working."""
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, kind TEXT NOT NULL,
            importance REAL NOT NULL, tick INTEGER NOT NULL, created_at REAL NOT NULL,
            last_access_tick INTEGER NOT NULL, access_count INTEGER NOT NULL,
            salience REAL NOT NULL, tags TEXT NOT NULL, metadata TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("old1", "a memory from the old schema", "observation", 0.5, 3,
         time.time(), 3, 0, 0.7, json.dumps(["legacy"]), "{}"),
    )
    conn.commit()
    conn.close()

    with LongTermMemory(path, embedder=None) as store:
        assert len(store) == 1
        # Backfilled into the FTS index that did not exist when it was written.
        assert store.search("old schema", limit=1)[0].record.id == "old1"
        store.add(MemoryRecord(content="a new one"))
        assert len(store) == 2


def test_stats_reports_the_degraded_path():
    store = LongTermMemory(":memory:", embedder=None)
    stats = store.stats()
    assert stats["fts5"] is True
    assert stats["embedder"] is None
    assert stats["semantic_recall"] is False
