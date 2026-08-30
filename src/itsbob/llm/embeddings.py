"""Vector embeddings — the semantic half of memory recall.

Memory recall fuses two signals: SQLite FTS5 (lexical, exact, free) and
cosine similarity over embeddings (semantic, catches paraphrase). This module
supplies the second.

Embeddings come from an API by default (Google's ``gemini-embedding-2``), so
there is no model download on the laptop. But a recall that *fails* because
the network is down is worse than a recall that is merely less clever, so
:class:`HashingEmbedder` is always the last link in the chain: a deterministic
signed-hash projection of word and bigram counts that needs nothing but the
standard library.

The critical invariant: **vectors from different models are not comparable.**
Every stored vector carries the model id that produced it, and recall only
ever compares vectors within one model id. Mixing them silently returns
nonsense rather than an error, so it is enforced here rather than trusted to
callers — see :meth:`Embedder.signature`.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..config import ProviderConfig
from .base import LLMError, ProviderNotConfigured, ProviderUnavailable

__all__ = [
    "EmbeddingError",
    "Embedder",
    "OpenAICompatibleEmbedder",
    "GoogleEmbedder",
    "HashingEmbedder",
    "EmbeddingRouter",
    "cosine",
    "default_embedder",
    "GOOGLE_EMBEDDING",
    "GOOGLE_NATIVE_DIMS",
    "DEFAULT_EMBED_DIMS",
]


class EmbeddingError(LLMError):
    """Every embedder in the chain failed."""


# Google's OpenAI-compatible shim serves /embeddings at the same base URL as
# chat. gemini-embedding-2 and -001 both return 3072 dimensions; -001 is kept
# as the fallback because new model ids get gated to existing accounts more
# often than old ones get retired.
GOOGLE_EMBEDDING = ProviderConfig(
    name="google-embed",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key_env="GOOGLE_API_KEY",
    default_model="gemini-embedding-2",
    fallback_models=("gemini-embedding-001",),
    requests_per_minute=100,
    rate_limit_scope="model",
    timeout=30.0,
)


@dataclass
class Embedder(ABC):
    """One way of turning text into vectors."""

    name: str
    model: str
    dims: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Vector per input, same order. Raise on failure."""

    @property
    def signature(self) -> str:
        """Identity a stored vector is tagged with.

        Two vectors may only be compared when their signatures match. Includes
        the dimension so a vendor silently changing output width invalidates
        the old rows instead of producing quiet garbage.
        """
        return f"{self.name}:{self.model}:{self.dims}"

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAICompatibleEmbedder(Embedder):
    """Any vendor exposing OpenAI's ``/embeddings`` route."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        model: str | None = None,
        dims: int = 768,
        env: Mapping[str, str] | None = None,
        batch_size: int = 64,
        request_dimensions: bool = True,
    ) -> None:
        super().__init__(name=config.name, model=model or config.default_model, dims=dims)
        self.config = config
        self.batch_size = max(1, batch_size)
        #: Gemini's embeddings are Matryoshka-trained, so asking for 768 of
        #: the native 3072 dimensions keeps almost all of the quality at a
        #: quarter of the storage and a quarter of the cosine arithmetic —
        #: which is what keeps brute-force recall viable in pure Python.
        self.request_dimensions = request_dimensions
        self._env = env
        self._client: Any | None = None

    def is_configured(self) -> bool:
        return self.config.api_key(self._env) is not None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ProviderNotConfigured(
                    f"{self.name}: the 'openai' package is required (pip install openai)"
                ) from exc
            key = self.config.api_key(self._env)
            if not key:
                raise ProviderNotConfigured(
                    f"{self.name}: set {self.config.api_key_env} to enable embeddings"
                )
            self._client = OpenAI(
                api_key=key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=0,
            )
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from .providers import _translate_error

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            kwargs: dict[str, Any] = {"model": self.model, "input": list(batch)}
            if self.request_dimensions and self.dims:
                kwargs["dimensions"] = self.dims
            try:
                response = self.client.embeddings.create(**kwargs)
            except Exception as exc:
                raise _translate_error(self.name, exc) from exc
            # The API is documented to preserve order and it usually also
            # returns an explicit index — but Google's shim sends `index:
            # None`, so fall back to arrival order rather than sorting on a
            # key that may not be an int.
            rows = sorted(
                enumerate(response.data),
                key=lambda pair: pair[1].index if isinstance(getattr(pair[1], "index", None), int) else pair[0],
            )
            rows = [row for _, row in rows]
            for row in rows:
                vector = list(row.embedding)
                if self.dims and len(vector) != self.dims:
                    # First real response wins: the configured dims was a
                    # guess, the vendor's answer is the truth.
                    self.dims = len(vector)
                out.append(vector)
        return out


#: Native width of the Gemini embedding models. Truncating to
#: :data:`DEFAULT_EMBED_DIMS` is a deliberate quality/cost trade, not a limit.
GOOGLE_NATIVE_DIMS = 3072

#: What memory actually stores. Change it and every stored vector's signature
#: changes with it, so old rows are ignored rather than silently mis-compared;
#: run ``itsbob memory reindex`` to re-embed them at the new width.
DEFAULT_EMBED_DIMS = 768


class GoogleEmbedder(OpenAICompatibleEmbedder):
    def __init__(
        self, model: str | None = None, *, dims: int = DEFAULT_EMBED_DIMS, **kwargs: Any
    ) -> None:
        super().__init__(GOOGLE_EMBEDDING, model=model, dims=dims, **kwargs)


_WORD_RE = re.compile(r"[a-z0-9']+")


class HashingEmbedder(Embedder):
    """Offline fallback: signed feature hashing over words and bigrams.

    Not a learned embedding — it cannot see that "car" and "automobile" are
    related. What it *can* do is survive with no network and no download, put
    near-duplicate text close together, and keep recall working end to end so
    the system degrades in quality rather than falling over. Deterministic, so
    a vector written today still matches one written next month.
    """

    def __init__(self, dims: int = 768) -> None:
        super().__init__(name="hashing", model=f"signed-hash-{dims}", dims=dims)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        words = _WORD_RE.findall(text.lower())
        features: list[str] = list(words)
        features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dims
            # Sign from an independent byte, so collisions cancel instead of
            # accumulating into one hot bucket.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to ``[-1, 1]``. Returns 0 on a length mismatch."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


@dataclass
class _EmbedStat:
    calls: int = 0
    texts: int = 0
    failures: int = 0
    latency_ms: float = 0.0
    last_error: str | None = None


class EmbeddingRouter(Embedder):
    """Failover chain over embedders, with an in-process de-duplication cache.

    Mirrors :class:`~itsbob.llm.router.LLMRouter`'s contract but stays much
    simpler: embeddings are idempotent and cheap to retry, so there is no
    circuit breaker or rate limiter here — just "try each in order, remember
    which one answered".

    :attr:`signature` reflects the embedder that *actually* answered last, so
    a chain that silently degrades from Google to hashing tags its rows
    differently and recall stops comparing across the boundary.
    """

    def __init__(self, embedders: Sequence[Embedder], *, cache_size: int = 4096) -> None:
        if not embedders:
            raise ValueError("EmbeddingRouter needs at least one embedder")
        self.embedders = list(embedders)
        self.active: Embedder = self.embedders[0]
        self.cache_size = max(0, cache_size)
        self._cache: dict[tuple[str, str], list[float]] = {}
        self.stats: dict[str, _EmbedStat] = {}
        #: Why the chain last had to fall past an embedder. Deliberately NOT
        #: cleared on a later success: silently degrading from a real
        #: embedding model to :class:`HashingEmbedder` changes recall quality,
        #: and a diagnostic that erases its own evidence is how that goes
        #: unnoticed for weeks. ``itsbob doctor`` surfaces it.
        self.last_error: str | None = None
        self.degraded: bool = False
        super().__init__(
            name="chain", model=self.active.model, dims=self.active.dims
        )

    @property
    def signature(self) -> str:
        return self.active.signature

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for e in self.embedders:
            stat = self.stats.get(e.signature, _EmbedStat())
            configured = getattr(e, "is_configured", lambda: True)()
            rows.append(
                {
                    "name": e.name,
                    "model": e.model,
                    "dims": e.dims,
                    "configured": bool(configured),
                    "active": e is self.active,
                    "calls": stat.calls,
                    "texts": stat.texts,
                    "failures": stat.failures,
                    "avg_latency_ms": round(stat.latency_ms / stat.calls, 1) if stat.calls else 0.0,
                    "last_error": stat.last_error,
                }
            )
        return rows

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        # Cache is keyed on (signature, text) so a chain that failed over
        # never serves a vector produced by a different model.
        signature = self.active.signature
        pending = [t for t in dict.fromkeys(texts) if (signature, t) not in self._cache]
        if pending:
            self._embed_uncached(pending)
            signature = self.active.signature

        out: list[list[float]] = []
        for text in texts:
            vector = self._cache.get((signature, text))
            if vector is None:
                # Failover mid-batch re-keyed the cache; embed the stragglers
                # under the embedder that actually answered.
                self._embed_uncached([text])
                signature = self.active.signature
                vector = self._cache.get((signature, text), [0.0] * self.active.dims)
            out.append(vector)
        return out

    def _embed_uncached(self, texts: Sequence[str]) -> None:
        errors: list[str] = []
        for embedder in self.embedders:
            stat = self.stats.setdefault(embedder.signature, _EmbedStat())
            configured = getattr(embedder, "is_configured", lambda: True)()
            if not configured:
                stat.last_error = "not configured"
                errors.append(f"{embedder.name}: not configured")
                continue
            started = time.perf_counter()
            try:
                vectors = embedder.embed(list(texts))
            except Exception as exc:  # noqa: BLE001 - the next embedder gets a turn
                stat.failures += 1
                stat.last_error = f"{type(exc).__name__}: {exc}"[:200]
                errors.append(f"{embedder.name}: {stat.last_error}")
                continue
            stat.calls += 1
            stat.texts += len(texts)
            stat.latency_ms += (time.perf_counter() - started) * 1000
            self.active = embedder
            self.model = embedder.model
            self.dims = embedder.dims
            if errors:
                # Answered, but only after something above it failed. Keep the
                # reason: this is exactly the case that otherwise looks fine.
                self.degraded = True
                self.last_error = "; ".join(errors)
            self._store(embedder.signature, texts, vectors)
            return

        self.degraded = True
        self.last_error = "; ".join(errors)
        raise EmbeddingError(f"all embedders failed ({self.last_error})")

    def _store(self, signature: str, texts: Sequence[str], vectors: Sequence[list[float]]) -> None:
        for text, vector in zip(texts, vectors):
            if self.cache_size and len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[(signature, text)] = list(vector)


def default_embedder(
    env: Mapping[str, str] | None = None, *, allow_api: bool = True
) -> EmbeddingRouter:
    """The standard chain: Google's embedding API, then offline hashing.

    ``ITSBOB_EMBED_MODEL`` pins the API model. ``ITSBOB_EMBED_OFFLINE=true``
    (or ``allow_api=False``) skips the API entirely — useful when you want
    memory writes to stay on the laptop.
    """
    env = os.environ if env is None else env
    offline = (env.get("ITSBOB_EMBED_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"})
    try:
        dims = int(env.get("ITSBOB_EMBED_DIMS", "").strip() or DEFAULT_EMBED_DIMS)
    except ValueError:
        dims = DEFAULT_EMBED_DIMS

    embedders: list[Embedder] = []
    if allow_api and not offline:
        model = env.get("ITSBOB_EMBED_MODEL", "").strip() or None
        google = GoogleEmbedder(model=model, dims=dims, env=env)
        if google.is_configured():
            embedders.append(google)
            for fallback in GOOGLE_EMBEDDING.fallback_models:
                if fallback != google.model:
                    embedders.append(GoogleEmbedder(model=fallback, dims=dims, env=env))
    # Same width as the API embedder so a failover doesn't also change the
    # vector length — the signature still differs, which is what matters, but
    # keeping the width stable makes the two directly diffable in diagnostics.
    embedders.append(HashingEmbedder(dims=dims))
    return EmbeddingRouter(embedders)
