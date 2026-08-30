"""The browser's view of one running agent: an event stream and an approval gate.

Two problems the previous interface had, both structural rather than cosmetic:

**A turn was a blocking POST.** You sent a message and watched a spinner for
ten seconds while the agent read files and ran commands, then everything
appeared at once. The interesting part — what it decided to do and what came
back — was invisible exactly while it was happening.

**Tools that needed approval could never run.** The interface passed no
confirmation handler, so the policy correctly failed closed and every
``guarded``-mode command was refused. Safe, and useless: the browser was the
one place a person actually *is*, and it was the one place that could not say
yes.

Both are solved by running the turn on a worker thread and streaming events to
the page. When the agent reaches a tool that needs consent, the callback parks
on an :class:`threading.Event` and the page renders an approve/deny card; the
answer releases it. A request nobody answers times out and is **denied**, which
keeps the fail-closed property: a card scrolled off screen, or a tab closed
mid-turn, must not become an approval.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

__all__ = ["Session", "PendingApproval"]

#: How long a tool call waits for a human before giving up and being refused.
#: Long enough to read the card and think; short enough that a closed tab does
#: not hold a worker thread until the process exits.
APPROVAL_TIMEOUT = 180.0

#: Cap on queued events per listener. A browser that stops reading (a
#: backgrounded tab, a dead connection) must not grow the queue without bound.
MAX_QUEUED_EVENTS = 500


@dataclass
class PendingApproval:
    """One tool call waiting on a person."""

    id: str
    tool: str
    params: dict[str, Any]
    risk: str
    reason: str
    created_at: float = field(default_factory=time.time)
    decided: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    #: Set when the person chose "always allow" for this tool this session.
    remember: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "params": self.params,
            "risk": self.risk,
            "reason": self.reason,
            "created_at": self.created_at,
            "timeout": APPROVAL_TIMEOUT,
        }


class Session:
    """One agent, its listeners, and whatever it is currently waiting on."""

    def __init__(self, agent_factory: Callable[[Callable[..., bool]], Any]) -> None:
        self._agent_factory = agent_factory
        self._agent: Any = None
        self._build_lock = threading.Lock()
        #: One turn at a time. The agent keeps conversation state, so two
        #: concurrent turns would interleave into each other's history.
        self._turn_lock = threading.Lock()
        self._listeners: list[queue.Queue] = []
        self._listeners_lock = threading.Lock()
        self.pending: dict[str, PendingApproval] = {}
        self.auto_allow: set[str] = set()
        self.busy = False
        self.last_error: str | None = None

    # -- the agent ---------------------------------------------------------

    @property
    def agent(self) -> Any:
        """Built on first use — a browser tab opening should not be what
        discovers a missing key."""
        with self._build_lock:
            if self._agent is None:
                self._agent = self._agent_factory(self.confirm)
            return self._agent

    def reset_conversation(self) -> None:
        from ..agent.context import Conversation

        self.agent.conversation = Conversation()

    # -- events ------------------------------------------------------------

    def listen(self) -> Iterator[str]:
        """Server-sent events for one browser connection."""
        stream: queue.Queue = queue.Queue(maxsize=MAX_QUEUED_EVENTS)
        with self._listeners_lock:
            self._listeners.append(stream)
        try:
            yield _sse({"kind": "hello", "busy": self.busy})
            while True:
                try:
                    event = stream.get(timeout=20.0)
                except queue.Empty:
                    # A comment frame keeps proxies and browsers from timing the
                    # connection out during a long quiet period.
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    return
                yield _sse(event)
        finally:
            with self._listeners_lock:
                if stream in self._listeners:
                    self._listeners.remove(stream)

    def emit(self, kind: str, **data: Any) -> None:
        event = {"kind": kind, "at": time.time(), **data}
        with self._listeners_lock:
            listeners = list(self._listeners)
        for stream in listeners:
            try:
                stream.put_nowait(event)
            except queue.Full:
                # This listener is not keeping up. Dropping its oldest event is
                # better than blocking the agent thread on a dead browser.
                try:
                    stream.get_nowait()
                    stream.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass

    # -- approvals ---------------------------------------------------------

    def confirm(self, tool: Any, params: dict[str, Any], call: Any) -> bool:
        """The policy's confirmation hook, answered by the browser."""
        if tool.name in self.auto_allow:
            self.emit("approval_auto", tool=tool.name)
            return True

        pending = PendingApproval(
            id=uuid.uuid4().hex[:12],
            tool=tool.name,
            params=dict(params),
            risk=getattr(tool.risk, "value", str(tool.risk)),
            reason=getattr(call, "reason", "") or "",
        )
        self.pending[pending.id] = pending
        self.emit("approval_request", **pending.as_dict())
        try:
            answered = pending.decided.wait(timeout=APPROVAL_TIMEOUT)
            if not answered:
                # Nobody answered. Refusing is the only safe reading of silence.
                self.emit("approval_timeout", id=pending.id, tool=pending.tool)
                return False
            if pending.approved and pending.remember:
                self.auto_allow.add(pending.tool)
            return pending.approved
        finally:
            self.pending.pop(pending.id, None)

    def decide(self, approval_id: str, *, approved: bool, remember: bool = False) -> bool:
        pending = self.pending.get(approval_id)
        if pending is None:
            return False
        pending.approved = approved
        pending.remember = remember
        pending.decided.set()
        self.emit("approval_decided", id=approval_id, tool=pending.tool, approved=approved)
        return True

    def cancel_pending(self) -> int:
        """Deny everything waiting. Used when a turn is abandoned."""
        count = 0
        for approval_id in list(self.pending):
            if self.decide(approval_id, approved=False):
                count += 1
        return count

    # -- turns -------------------------------------------------------------

    def start_turn(self, message: str, *, context: Any = None) -> bool:
        """Run one turn on a worker thread. False if one is already running."""
        if not self._turn_lock.acquire(blocking=False):
            return False

        def run() -> None:
            self.busy = True
            self.last_error = None
            self.emit("turn_start", message=message)
            try:
                turn = self.agent.chat(
                    message, context=context, on_event=self._forward
                )
                self.emit("turn_end", turn=turn.as_dict(), reply=turn.final)
            except Exception as exc:  # noqa: BLE001 - surface it, never 500 the stream
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.emit("turn_error", error=self.last_error)
            finally:
                self.busy = False
                self.cancel_pending()
                self._turn_lock.release()

        threading.Thread(target=run, name="itsbob-gui-turn", daemon=True).start()
        return True

    def _forward(self, event: Any) -> None:
        self.emit(event.kind, **event.data)

    def close(self) -> None:
        with self._listeners_lock:
            for stream in self._listeners:
                try:
                    stream.put_nowait(None)
                except queue.Full:
                    pass


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"
