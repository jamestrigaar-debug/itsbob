"""Semantic caching — "the big saver".

Keyed on the Gatekeeper's 5-word state fingerprint, not the raw state: two
differently-worded but tactically identical situations should hash to (or
near) the same cache entry. Entries expire after ``ttl_seconds`` (the spec's
"happened 5 minutes ago" window) so a cached tactic doesn't go stale mid-match.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["SemanticCache", "normalize_fingerprint"]

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_fingerprint(fingerprint: str) -> str:
    """Lowercase, sort the words, drop punctuation — order/casing shouldn't matter."""
    words = sorted(_WORD_RE.findall(fingerprint.lower()))
    return " ".join(words)


@dataclass
class _Entry:
    value: Any
    expires_at: float


class SemanticCache:
    """Small in-process TTL cache, hit-rate tracked so the GUI can show it.

    In-process is a deliberate choice: this pipeline runs as one long-lived
    process (a game companion, not a fleet of workers), so there's no need for
    Redis or similar — a dict with timestamps covers the "same tactical
    situation happened 5 minutes ago" case exactly.
    """

    def __init__(
        self, *, ttl_seconds: float = 300.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(fingerprint: str) -> str:
        normalized = normalize_fingerprint(fingerprint)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get(self, fingerprint: str) -> Any | None:
        key = self.key_for(fingerprint)
        entry = self._entries.get(key)
        if entry is None or entry.expires_at < self._clock():
            self.misses += 1
            self._entries.pop(key, None)
            return None
        self.hits += 1
        return entry.value

    def put(self, fingerprint: str, value: Any) -> None:
        key = self.key_for(fingerprint)
        self._entries[key] = _Entry(value=value, expires_at=self._clock() + self.ttl_seconds)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
        }
