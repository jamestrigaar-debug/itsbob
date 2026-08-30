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

def _build_system_prompt(script_names: tuple[str, ...]) -> str:
    # The base spec's Gatekeeper prompt asks for a tag and a fingerprint only
    # — but a [SCRIPT] tag with no script name attached is undispatchable
    # (the pipeline has nothing to execute), so this also asks for the
    # script itself when tagging SCRIPT, constrained to names the registry
    # actually knows — the same "only ever *name* a pre-registered macro"
    # contract Tier B/A cloud prompts already use, applied to Tier D too.
    names = ", ".join(script_names) or "(none registered)"
    return (
        "You are the 'Gatekeeper'. Analyze the provided game state. Output ONLY "
        "one of these tags: [SCRIPT], [LOCAL_SUM], [CLOUD_B], or [CLOUD_A]. Also "
        "output a 5-word compressed 'state fingerprint' for caching. If — and "
        f"only if — the tag is [SCRIPT], also name exactly one script from this "
        f"list that applies: {names}. Never invent a script name not in that list. "
        'Reply as strict JSON: {"tag": "<TAG>", "fingerprint": "<five words>", '
        '"script": "<SCRIPT_NAME or omit/null if tag is not SCRIPT>"}'
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
            messages=[system(_build_system_prompt(tuple(self.registry.names()))), user(state.render())],
            max_tokens=80,
            temperature=0.0,
            json_mode=True,
        )
        started = time.perf_counter()
        response = self.local_provider.complete_with_fallback(  # type: ignore[union-attr]
            request, preferred_model=self.local_model
        )
        latency_ms = (time.perf_counter() - started) * 1000

        tag, fingerprint, script = _parse_gatekeeper_reply(response.text)
        if tag is None:
            raise ValueError(f"gatekeeper reply had no recognizable tag: {response.text!r}")

        metadata: dict[str, Any] = {"raw_tag": tag, "model": response.model}
        reasoning = f"local model tagged [{tag}]"
        if tag == "SCRIPT":
            if script and self.registry.has(script):
                metadata["script"] = script
                reasoning += f" -> {script}"
            else:
                reasoning += (
                    f" but named no valid script ({script!r})" if script
                    else " but named no script"
                )
        if latency_ms > self.max_local_latency_ms:
            # A soft budget, surfaced rather than enforced: aborting a slow
            # but *successful* local call would just throw away a working
            # answer over a target latency (<800ms) that assumes a
            # quantized model on capable hardware — plain CPU inference on
            # a modest laptop can legitimately take several seconds. This
            # makes that visible in the trace instead of pretending it
            # isn't happening.
            reasoning += f" (took {latency_ms:.0f}ms, over the {self.max_local_latency_ms:.0f}ms target)"

        return GateDecision(
            tier=GATEKEEPER_TAGS[tag],
            fingerprint=fingerprint or _fallback_fingerprint(state, tag),
            source="gatekeeper",
            reasoning=reasoning,
            latency_ms=latency_ms,
            metadata=metadata,
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


def _parse_gatekeeper_reply(text: str) -> tuple[str | None, str | None, str | None]:
    from ..llm.router import extract_json

    parsed = extract_json(text)
    if parsed:
        tag_raw = str(parsed.get("tag", ""))
        match = _TAG_RE.search(tag_raw) or _TAG_RE.search(text)
        tag = match.group(1) if match else None
        fingerprint = str(parsed.get("fingerprint", "")).strip() or None
        script = str(parsed.get("script", "")).strip() or None
        return tag, fingerprint, script

    match = _TAG_RE.search(text)
    return (match.group(1) if match else None), None, None


def _fallback_fingerprint(state: GameState, tag: str) -> str:
    """Cheap 5-ish-word fingerprint when the model didn't supply one."""
    words = [f"{k}:{v}" for k, v in list(state.facts.items())[:4]]
    words.append(tag.lower())
    return " ".join(str(w) for w in words[:5]) or tag.lower()
