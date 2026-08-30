"""Append-only record of everything the agent did.

An agent that can run commands on your laptop needs an answer to "what did it
actually do?" that does not depend on a chat scrollback you may have closed.
One JSON object per line, flushed on write, never rewritten: cheap to append,
trivially greppable, and safe to tail while the daemon is running.

Denied calls are recorded too. What the agent *tried* to do and was stopped
from doing is the more interesting half of the log.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .base import ToolCall, ToolResult

__all__ = ["AuditLog"]

#: Keys whose values are replaced in the log. The agent handles credentials
#: (an http_request header, an API key in an env override) and a log is exactly
#: the wrong place for them to come to rest.
_REDACT_KEYS = ("authorization", "api_key", "apikey", "token", "secret", "password", "cookie")


@dataclass
class AuditLog:
    """JSONL sink for tool activity."""

    path: Path | None = None
    #: Kept in memory as well, so the GUI and ``itsbob audit`` can show recent
    #: activity without re-reading the file.
    keep: int = 500
    max_value_chars: int = 2000

    def __post_init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path is not None:
            self.path = Path(self.path).expanduser()
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        call: ToolCall,
        result: ToolResult | None,
        *,
        denied: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "pid": os.getpid(),
            "tool": call.name,
            "params": _redact(call.params, self.max_value_chars),
            "reason": call.reason,
            "denied": denied,
            "ok": None if denied else bool(result and result.ok),
            "dry_run": bool(result and result.dry_run),
            "duration_ms": round(result.duration_ms, 1) if result else 0.0,
            "output": _truncate(result.output if result else "", 400),
            "error": (result.error if result else None) or denied,
        }
        with self._lock:
            self.entries.append(entry)
            if len(self.entries) > self.keep:
                del self.entries[: len(self.entries) - self.keep]
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, default=str) + "\n")
        return entry

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.entries[-limit:])

    def read(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Replay the file, oldest first. Skips lines that aren't valid JSON."""
        if self.path is None or not self.path.exists():
            return iter(())
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]

        def _iter() -> Iterator[dict[str, Any]]:
            for line in lines:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

        return _iter()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            entries = list(self.entries)
        return {
            "path": str(self.path) if self.path else None,
            "in_memory": len(entries),
            "ok": sum(1 for e in entries if e["ok"] is True),
            "failed": sum(1 for e in entries if e["ok"] is False),
            "denied": sum(1 for e in entries if e["denied"]),
            "dry_run": sum(1 for e in entries if e["dry_run"]),
        }


def _redact(params: dict[str, Any], limit: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if any(marker in key.lower() for marker in _REDACT_KEYS):
            out[key] = "[redacted]"
        elif isinstance(value, dict):
            out[key] = _redact(value, limit)
        elif isinstance(value, str):
            out[key] = _truncate(value, limit)
        else:
            out[key] = value
    return out


def _truncate(text: str, limit: int) -> str:
    if not isinstance(text, str) or len(text) <= limit:
        return text
    return f"{text[:limit]}… [+{len(text) - limit} chars]"
