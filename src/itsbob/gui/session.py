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

**A message sent while it was working was rejected.** Turns run one at a time —
they have to, since the agent carries conversation state and two at once would
interleave into each other's history — but refusing the message put that
constraint on the person rather than on the software. Messages are now queued
and run in order, so you can keep typing while it works and it catches up.

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
from collections import deque
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

#: Cap on unsent messages. Deep enough that nobody types into it by accident,
#: shallow enough that a stuck agent does not accumulate an evening of work to
#: replay when it recovers.
MAX_QUEUED_MESSAGES = 20


@dataclass
class QueuedMessage:
    """One thing waiting to be run, and where it came from."""

    text: str
    source: str = "user"  #: "user" | "task"
    context: Any = None
    label: str = ""
    #: Called with the finished Turn (or None on failure). Used by autonomous
    #: mode to record the run and decide whether to interrupt.
    on_done: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source": self.source, "label": self.label}


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
        self.current_source: str | None = None
        self.last_error: str | None = None
        #: Messages waiting their turn, oldest first.
        self._queue: deque[QueuedMessage] = deque()
        self._queue_lock = threading.Lock()
        self.current: str | None = None

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

    def submit(
        self,
        message: str,
        *,
        context: Any = None,
        source: str = "user",
        label: str = "",
        on_done: Any = None,
    ) -> dict[str, Any]:
        """Accept a message. Runs it now if idle, queues it if not.

        Always accepts — refusing while busy made the one-turn-at-a-time
        constraint the person's problem rather than the software's.

        A typed message goes ahead of anything scheduled that is still waiting.
        Autonomous work is by definition not urgent, and making a person wait
        behind a nightly backup summary to ask a question is the wrong way
        round. It does not preempt whatever is already *running*: interrupting
        a turn mid-tool-call would leave the work half done.
        """
        item = QueuedMessage(
            text=message, source=source, context=context, label=label, on_done=on_done
        )
        with self._queue_lock:
            if len(self._queue) >= MAX_QUEUED_MESSAGES:
                return {
                    "accepted": False,
                    "queued": len(self._queue),
                    "error": (
                        f"{len(self._queue)} messages are already waiting. "
                        "Let it catch up, or clear the queue."
                    ),
                }
            if source == "user":
                # After the last queued user message, before the first task.
                position = len(self._queue)
                for index, queued in enumerate(self._queue):
                    if queued.source != "user":
                        position = index
                        break
                self._queue.insert(position, item)
            else:
                self._queue.append(item)
            depth = len(self._queue)

        started_now = self._ensure_worker()
        if not started_now:
            self.emit("queued", message=message, source=source, label=label, queued=depth)
        return {"accepted": True, "queued": depth, "started_now": started_now}

    def _ensure_worker(self) -> bool:
        """Start the drain thread if one is not already running. True if started."""
        if not self._turn_lock.acquire(blocking=False):
            return False
        threading.Thread(target=self._drain, name="itsbob-gui-turns", daemon=True).start()
        return True

    def _drain(self) -> None:
        """Run queued messages in order until there are none left.

        One thread for the whole queue rather than one per message: the lock is
        held for the entire drain, so a message submitted mid-run joins the
        queue instead of racing to start a second turn.
        """
        try:
            while True:
                with self._queue_lock:
                    if not self._queue:
                        return
                    item = self._queue.popleft()
                    remaining = len(self._queue)
                self._run_one(item, remaining)
        finally:
            self.busy = False
            self.current = None
            self.current_source = None
            self._turn_lock.release()
            # Anything submitted between the queue emptying and the lock being
            # released would otherwise sit there until the next message.
            with self._queue_lock:
                orphaned = bool(self._queue)
            if orphaned:
                self._ensure_worker()

    def _run_one(self, item: QueuedMessage, remaining: int) -> None:
        self.busy = True
        self.current = item.label or item.text
        self.current_source = item.source
        self.last_error = None
        turn = None
        self.emit(
            "turn_start", message=item.text, source=item.source,
            label=item.label, queued=remaining,
        )
        try:
            turn = self.agent.chat(item.text, context=item.context, on_event=self._forward)
            self.emit(
                "turn_end", turn=turn.as_dict(), reply=turn.final,
                source=item.source, label=item.label, queued=remaining,
            )
        except Exception as exc:  # noqa: BLE001 - surface it, never 500 the stream
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.emit("turn_error", error=self.last_error, source=item.source, queued=remaining)
        finally:
            # Denied here rather than at drain time: an approval left waiting by
            # a failed turn must not outlive it into the next one.
            self.cancel_pending()
            if item.on_done is not None:
                try:
                    item.on_done(turn, self.last_error)
                except Exception:  # noqa: BLE001 - a callback must not break the queue
                    pass

    def start_turn(self, message: str, *, context: Any = None) -> bool:
        """Backwards-compatible shim. Prefer :meth:`submit`."""
        return bool(self.submit(message, context=context)["accepted"])

    def queued_messages(self) -> list[dict[str, Any]]:
        with self._queue_lock:
            return [item.as_dict() for item in self._queue]

    def clear_queue(self) -> int:
        """Drop everything waiting. Does not touch the turn already running."""
        with self._queue_lock:
            dropped = len(self._queue)
            self._queue.clear()
        if dropped:
            self.emit("queue_cleared", dropped=dropped)
        return dropped

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
