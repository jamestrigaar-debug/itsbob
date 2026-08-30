"""Step 1 of the pipeline: Ingestion & Compression.

Truncates raw scraped state to the most recent N events so it fits the local
model's tiny context window, and renders it into the compact text block both
the Gatekeeper and the cloud prompts are built from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["GameState", "compress"]

#: "truncates this to the most recent 20 events" — the spec's number.
DEFAULT_EVENT_WINDOW = 20


@dataclass
class GameState:
    """One ingested, compressed snapshot ready to classify or route.

    ``facts`` is the flat scalar state (score, minute, formation, morale, ...);
    ``events`` is the event log, already truncated to the window.
    """

    facts: dict[str, Any] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)

    def render(self) -> str:
        """Compact single block, cheap on tokens for both local and cloud prompts."""
        lines = [f"{k}={v}" for k, v in self.facts.items()]
        block = "; ".join(lines)
        if self.events:
            tail = " | ".join(_event_text(e) for e in self.events)
            block = f"{block}\nrecent events: {tail}" if block else f"recent events: {tail}"
        return block or "(empty state)"

    def as_dict(self) -> dict[str, Any]:
        return {"facts": self.facts, "events": self.events}


def compress(
    raw: Mapping[str, Any] | str,
    *,
    event_window: int = DEFAULT_EVENT_WINDOW,
) -> GameState:
    """Parse + truncate raw scraper output into a :class:`GameState`.

    Accepts either a dict already shaped like ``{"facts": ..., "events": [...]}``,
    a flat dict of scalars (treated entirely as facts, no events), or a raw
    JSON string of either. Truncation always keeps the *most recent* events —
    the tail of the list, not the head.
    """
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}

    if "facts" in raw or "events" in raw:
        facts = dict(raw.get("facts") or {})
        events: Sequence[Any] = raw.get("events") or []
    else:
        facts = {k: v for k, v in raw.items() if k != "events"}
        events = raw.get("events", []) if isinstance(raw.get("events"), list) else []

    truncated = list(events)[-event_window:] if event_window > 0 else list(events)
    return GameState(facts=facts, events=truncated)


def _event_text(event: Any) -> str:
    if isinstance(event, str):
        return event
    if isinstance(event, Mapping):
        return ", ".join(f"{k}={v}" for k, v in event.items())
    return str(event)
