"""Step 1: ingestion and compression.

Whatever arrives — a typed message, a scraped state dict, a scheduled task's
payload — becomes a :class:`Snapshot`: some free text, some flat facts, and a
bounded event log. Everything downstream (the classifier, the cache
fingerprint, the prompts) reads that one shape, so adding an input source is a
converter rather than a new code path.

The event window exists because the classifier runs on the cheapest model
available, whose context is small and whose accuracy falls off a cliff when
you fill it. Keeping the most recent N events is a better trade than keeping a
uniform sample of all of them: what just happened is what the decision is
about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["Snapshot", "GameState", "compress", "DEFAULT_EVENT_WINDOW"]

#: How many trailing events survive compression.
DEFAULT_EVENT_WINDOW = 20


@dataclass
class Snapshot:
    """One ingested, compressed input ready to classify or route.

    ``text`` is what a person actually said, when there was one. ``facts`` is
    flat scalar state; ``events`` is a log, already truncated to the window.
    """

    text: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)

    def render(self, *, max_chars: int = 4000) -> str:
        """Compact block, cheap on tokens for both local and cloud prompts."""
        blocks: list[str] = []
        if self.text.strip():
            blocks.append(self.text.strip())
        if self.facts:
            blocks.append("; ".join(f"{k}={v}" for k, v in self.facts.items()))
        if self.events:
            blocks.append("recent: " + " | ".join(_event_text(e) for e in self.events))
        rendered = "\n".join(blocks) or "(empty)"
        if len(rendered) > max_chars:
            half = max_chars // 2
            rendered = f"{rendered[:half]}\n…\n{rendered[-half:]}"
        return rendered

    @property
    def is_empty(self) -> bool:
        return not (self.text.strip() or self.facts or self.events)

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "facts": self.facts, "events": self.events}


#: The router's original name for this type, kept so existing callers still
#: import cleanly. It was never game-specific in structure, only in naming.
GameState = Snapshot


def compress(
    raw: Mapping[str, Any] | str | None,
    *,
    event_window: int = DEFAULT_EVENT_WINDOW,
) -> Snapshot:
    """Parse and truncate any supported input into a :class:`Snapshot`.

    Accepts a plain string (becomes ``text``), a dict shaped like
    ``{"text"/"message", "facts", "events"}``, a flat dict of scalars (all
    facts), or a JSON string of either. A string that happens to parse as JSON
    is treated as structure; one that does not is treated as what someone
    typed, which is the common case and must never raise.
    """
    if raw is None:
        return Snapshot()

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return Snapshot()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return Snapshot(text=stripped)
        if not isinstance(parsed, Mapping):
            return Snapshot(text=stripped)
        raw = parsed

    text = str(raw.get("text") or raw.get("message") or "")
    if "facts" in raw or "events" in raw or text:
        facts = dict(raw.get("facts") or {})
        events: Sequence[Any] = raw.get("events") or []
    else:
        facts = {k: v for k, v in raw.items() if k != "events"}
        events = raw.get("events", []) if isinstance(raw.get("events"), list) else []

    truncated = list(events)[-event_window:] if event_window > 0 else list(events)
    return Snapshot(text=text, facts=facts, events=truncated)


def _event_text(event: Any) -> str:
    if isinstance(event, str):
        return event
    if isinstance(event, Mapping):
        return ", ".join(f"{k}={v}" for k, v in event.items())
    return str(event)
