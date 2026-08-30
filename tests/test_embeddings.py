"""Embedding chain behaviour. Offline only — no test here touches the network."""

from __future__ import annotations

import pytest

from itsbob.llm.embeddings import (
    EmbeddingError,
    EmbeddingRouter,
    Embedder,
    HashingEmbedder,
    cosine,
    default_embedder,
)


class _Boom(Embedder):
    """An embedder that always fails, for exercising failover."""

    def __init__(self, dims: int = 8) -> None:
        super().__init__(name="boom", model="always-fails", dims=dims)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        raise RuntimeError("nope")


class _Fixed(Embedder):
    def __init__(self, value: float, dims: int = 8) -> None:
        super().__init__(name="fixed", model=f"fixed-{value}", dims=dims)
        self.value = value
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[self.value] * self.dims for _ in texts]


def test_hashing_is_deterministic_across_instances():
    a = HashingEmbedder().embed_one("the quick brown fox")
    b = HashingEmbedder().embed_one("the quick brown fox")
    assert a == b


def test_hashing_is_normalized_and_separates_near_duplicates():
    h = HashingEmbedder()
    near, other = h.embed(["the cat sat on the mat", "quantum chromodynamics lecture"])
    same = h.embed_one("a cat sat on a mat")
    assert cosine(near, same) > cosine(near, other)
    assert abs(sum(v * v for v in near) - 1.0) < 1e-6


def test_hashing_handles_empty_text_without_dividing_by_zero():
    assert HashingEmbedder().embed_one("") == [0.0] * 768


def test_cosine_rejects_mismatched_widths_instead_of_raising():
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
    assert cosine([], []) == 0.0


def test_router_falls_through_to_the_next_embedder():
    boom, good = _Boom(), _Fixed(0.5)
    router = EmbeddingRouter([boom, good])
    assert router.embed(["x"]) == [[0.5] * 8]
    assert boom.calls == 1
    assert router.active is good
    assert router.degraded is True
    assert "boom" in (router.last_error or "")


def test_router_keeps_the_failover_reason_after_a_later_success():
    """A silent degrade to a weaker embedder is the bug this guards against."""
    router = EmbeddingRouter([_Boom(), _Fixed(0.5)])
    router.embed(["x"])
    router.embed(["y"])
    assert router.degraded is True
    assert router.last_error


def test_router_raises_when_every_embedder_fails():
    router = EmbeddingRouter([_Boom(), _Boom()])
    with pytest.raises(EmbeddingError):
        router.embed(["x"])


def test_router_caches_per_signature_not_per_text():
    good = _Fixed(0.5)
    router = EmbeddingRouter([good])
    router.embed(["a", "b"])
    router.embed(["a", "b"])
    assert good.calls == 1  # second call served entirely from cache


def test_signature_tracks_the_embedder_that_actually_answered():
    router = EmbeddingRouter([_Boom(), _Fixed(0.5)])
    router.embed(["x"])
    assert router.signature == "fixed:fixed-0.5:8"


def test_signature_includes_dims_so_a_width_change_invalidates_rows():
    assert HashingEmbedder(dims=256).signature != HashingEmbedder(dims=768).signature


def test_default_embedder_is_offline_only_without_keys():
    router = default_embedder(env={}, allow_api=False)
    assert [e.name for e in router.embedders] == ["hashing"]


def test_embed_preserves_input_order_and_duplicates():
    router = EmbeddingRouter([HashingEmbedder(dims=64)])
    out = router.embed(["a", "b", "a"])
    assert len(out) == 3
    assert out[0] == out[2]
    assert out[0] != out[1]
