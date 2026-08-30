"""Step 2 of the handshake: the Gatekeeper classifier prompt.

"You are the 'Gatekeeper'. Analyze the provided game state. Output ONLY one
of these tags: [SCRIPT], [LOCAL_SUM], [CLOUD_B], or [CLOUD_A]. Also output a
5-word compressed 'state fingerprint' for caching."

The Gatekeeper never generates free text for the user — its entire job is
one tag plus one fingerprint. It prefers the local Back Brain
(:class:`~itsbob.llm.local.OllamaProvider`); if that is unreachable or
answers garbage, it falls back to a fast rule-based classifier so the
pipeline degrades gracefully rather than stalling on a missing model
download.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from ..llm.base import LLMRequest, Provider, ProviderUnavailable, system, user
from .ingestion import GameState
from .scripts import ScriptRegistry
from .tiers import GATEKEEPER_TAGS, GateDecision, Tier

__all__ = ["Gatekeeper"]

_SYSTEM_PROMPT = (
    "You are the 'Gatekeeper'. Analyze the provided game state. Output ONLY "
    "one of these tags: [SCRIPT], [LOCAL_SUM], [CLOUD_B], or [CLOUD_A]. Also "
    "output a 5-word compressed 'state fingerprint' for caching. "
    'Reply as strict JSON: {"tag": "<TAG>", "fingerprint": "<five words>"}'
)

_TAG_RE = re.compile(r"\[?(SCRIPT|LOCAL_SUM|CLOUD_B|CLOUD_A)\]?")

# Heuristic fallback thresholds, tuned to the spec's own examples.
_HIGH_RISK_WORDS = {
    "negotiat", "contract", "renegotiat", "release clause", "fallout", "dispute",
}
_WORLD_KNOWLEDGE_WORDS = {
    "tactic", "formation", "opponent", "transfer", "valuation", "press", "wing",
}


@dataclass
class Gatekeeper:
    """Classifies a :class:`GameState` into a :class:`~itsbob.router.tiers.Tier`."""

    registry: ScriptRegistry
    local_provider: Provider | None = None
    local_model: str | None = None
    max_local_latency_ms: float = 800.0

    def classify(self, state: GameState) -> GateDecision:
        # Tier D check first — a deterministic hit is always cheapest and
        # fastest, so the model (local or otherwise) never even gets asked.
        script_name = self.registry.first_triggered(state)
        if script_name is not None:
            return GateDecision(
                tier=Tier.D,
                fingerprint=_fallback_fingerprint(state, script_name),
                source="script-trigger",
                reasoning=f"deterministic trigger matched: {script_name}",
                latency_ms=0.0,
                metadata={"script": script_name},
            )

        if self.local_provider is not None:
            try:
                return self._classify_with_model(state)
            except Exception as exc:  # noqa: BLE001 - any failure degrades gracefully
                heuristic = self._classify_heuristically(state)
                heuristic.reasoning = f"local model failed ({type(exc).__name__}); {heuristic.reasoning}"
                return heuristic

        return self._classify_heuristically(state)

    # -- local model path ----------------------------------------------------

    def _classify_with_model(self, state: GameState) -> GateDecision:
        request = LLMRequest(
            messages=[system(_SYSTEM_PROMPT), user(state.render())],
            max_tokens=60,
            temperature=0.0,
            json_mode=True,
        )
        started = time.perf_counter()
        response = self.local_provider.complete_with_fallback(  # type: ignore[union-attr]
            request, preferred_model=self.local_model
        )
        latency_ms = (time.perf_counter() - started) * 1000

        tag, fingerprint = _parse_gatekeeper_reply(response.text)
        if tag is None:
            raise ValueError(f"gatekeeper reply had no recognizable tag: {response.text!r}")

        return GateDecision(
            tier=GATEKEEPER_TAGS[tag],
            fingerprint=fingerprint or _fallback_fingerprint(state, tag),
            source="gatekeeper",
            reasoning=f"local model tagged [{tag}]",
            latency_ms=latency_ms,
            metadata={"raw_tag": tag, "model": response.model},
        )

    # -- rule-based fallback --------------------------------------------------

    def _classify_heuristically(self, state: GameState) -> GateDecision:
        """Pure-Python stand-in for the local model's job — same tag space.

        Mirrors the taxonomy's own criteria: reasoning depth (does it need
        world knowledge / negotiation), data volume (how much text), and
        action risk (does it look high-stakes). Nowhere near as accurate as
        a tuned classifier, but it keeps the pipeline running end to end with
        nothing installed.
        """
        started = time.perf_counter()
        text = state.render().lower()
        char_count = len(text)

        if any(w in text for w in _HIGH_RISK_WORDS):
            tag = "CLOUD_A"
        elif any(w in text for w in _WORLD_KNOWLEDGE_WORDS):
            tag = "CLOUD_B"
        elif char_count < 500:
            tag = "LOCAL_SUM"
        else:
            tag = "CLOUD_B"

        latency_ms = (time.perf_counter() - started) * 1000
        return GateDecision(
            tier=GATEKEEPER_TAGS[tag],
            fingerprint=_fallback_fingerprint(state, tag),
            source="heuristic",
            reasoning=f"rule-based classifier (no local model available): {char_count} chars -> [{tag}]",
            latency_ms=latency_ms,
            metadata={"raw_tag": tag, "char_count": char_count},
        )


def _parse_gatekeeper_reply(text: str) -> tuple[str | None, str | None]:
    from ..llm.router import extract_json

    parsed = extract_json(text)
    if parsed:
        tag_raw = str(parsed.get("tag", ""))
        match = _TAG_RE.search(tag_raw) or _TAG_RE.search(text)
        tag = match.group(1) if match else None
        fingerprint = str(parsed.get("fingerprint", "")).strip() or None
        return tag, fingerprint

    match = _TAG_RE.search(text)
    return (match.group(1) if match else None), None


def _fallback_fingerprint(state: GameState, tag: str) -> str:
    """Cheap 5-ish-word fingerprint when the model didn't supply one."""
    words = [f"{k}:{v}" for k, v in list(state.facts.items())[:4]]
    words.append(tag.lower())
    return " ".join(str(w) for w in words[:5]) or tag.lower()
