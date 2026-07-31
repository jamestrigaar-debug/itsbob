"""A tiny synchronous event bus.

The simulation emits; everything else — logging, the CLI's narration, a future
UI — subscribes. Keeps the tick loop free of presentation concerns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["Event", "EventBus", "WILDCARD"]

WILDCARD = "*"


@dataclass(frozen=True)
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


Listener = Callable[[Event], None]


class EventBus:
    """Subscribe by exact name or with ``"*"`` for everything."""

    def __init__(self, *, keep_history: int = 500) -> None:
        self._listeners: dict[str, list[Listener]] = {}
        self.history: list[Event] = []
        self._keep_history = keep_history

    def subscribe(self, name: str, listener: Listener) -> Callable[[], None]:
        """Register ``listener``; returns a function that unsubscribes it."""
        self._listeners.setdefault(name, []).append(listener)

        def unsubscribe() -> None:
            handlers = self._listeners.get(name, [])
            if listener in handlers:
                handlers.remove(listener)

        return unsubscribe

    def on(self, name: str) -> Callable[[Listener], Listener]:
        """Decorator form of :meth:`subscribe`."""

        def decorator(listener: Listener) -> Listener:
            self.subscribe(name, listener)
            return listener

        return decorator

    def emit(self, name: str, **payload: Any) -> Event:
        event = Event(name=name, payload=payload)
        if self._keep_history:
            self.history.append(event)
            if len(self.history) > self._keep_history:
                del self.history[: len(self.history) - self._keep_history]
        for listener in (*self._listeners.get(name, ()), *self._listeners.get(WILDCARD, ())):
            # A broken listener must not take the simulation down with it.
            try:
                listener(event)
            except Exception:  # pragma: no cover - defensive
                continue
        return event

    def events(self, name: str) -> list[Event]:
        return [e for e in self.history if e.name == name]

    def clear(self) -> None:
        self.history.clear()
