"""Discord as the workspace: where Bob can speak first.

Everything else in this system is answer-shaped. Someone types, a turn runs, a
reply comes back. The daemon can already *decide* something is worth saying, but
the places it can say it — a desktop toast, a line in a log — are not places you
have a conversation. Discord is, and it is already open.

So this module makes a channel into a two-way workspace:

* **Outbound**, as a :class:`~itsbob.daemon.notify.Sink`: anything the notice
  gate passes gets posted to the channel. Task results, alerts, the morning
  briefing, and — the part that was actually asked for — messages Bob starts
  himself, with nobody having said anything first.
* **Inbound**, as a poller: messages typed in that channel by a human become
  ordinary turns, queued alongside anything typed in the browser.

Deliberately built on the REST API with ``urllib`` rather than the gateway with
``discord.py``. The gateway means a websocket, an async runtime and a large
dependency, to gain sub-second delivery for a channel a person checks a few
times an hour. Polling every few seconds costs one cheap request and no new
dependency, and it is the *same* mechanism whether it runs inside the GUI, the
daemon, or on its own.

Two Discord rules are handled here rather than left to the caller, because
getting them wrong is silent: messages over 2000 characters are rejected (so
they are split), and rate limits come back as HTTP 429 with ``retry_after``
(so they are waited out and retried, with a bounded number of attempts).

Bot messages are never treated as input. Without that check the bot answers
itself, and each answer is another message to answer.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "DiscordClient",
    "DiscordSink",
    "DiscordBridge",
    "discord_tools",
    "is_configured",
    "MAX_MESSAGE_CHARS",
]

API = "https://discord.com/api/v10"
#: Discord's own limit. A longer message is rejected outright, not truncated.
MAX_MESSAGE_CHARS = 2000
TIMEOUT = 20.0
#: How many times a rate-limited or failed request is retried before giving up.
MAX_ATTEMPTS = 4


def is_configured(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(
        str(env.get("DISCORD_BOT_TOKEN", "")).strip()
        and str(env.get("DISCORD_CHANNEL_ID", "")).strip()
    )


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Break ``text`` into postable chunks, preferring paragraph then line breaks.

    Splitting on a blank line where possible keeps a briefing readable across
    two posts; splitting mid-word, which a naive slice does, does not.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


@dataclass
class DiscordClient:
    """The two REST calls this needs, with retries and rate-limit handling."""

    token: str
    channel_id: str
    timeout: float = TIMEOUT
    #: Injectable so tests never touch the network.
    opener: Any = None
    #: Set when the last call failed, for the status panel.
    last_error: str | None = None
    sent: int = 0
    received: int = 0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DiscordClient | None":
        env = os.environ if env is None else env
        token = str(env.get("DISCORD_BOT_TOKEN", "")).strip()
        channel = str(env.get("DISCORD_CHANNEL_ID", "")).strip()
        if not (token and channel):
            return None
        return cls(token=token, channel_id=channel)

    # -- transport ---------------------------------------------------------

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{API}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bot {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "itsbob (https://github.com/jamestrigaar-debug/itsbob, 1.0)",
            },
        )
        opener = self.opener or urllib.request.urlopen
        for attempt in range(MAX_ATTEMPTS):
            try:
                with opener(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8", "replace")
                self.last_error = None
                return json.loads(raw) if raw.strip() else None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code == 429:
                    # Discord tells us exactly how long to wait. Honour it
                    # rather than guessing — guessing gets the bot banned.
                    wait = 1.0
                    try:
                        wait = float(json.loads(detail).get("retry_after", 1.0))
                    except (ValueError, TypeError, AttributeError):
                        pass
                    time.sleep(min(30.0, max(0.5, wait)))
                    continue
                if 500 <= exc.code < 600 and attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2.0**attempt)
                    continue
                self.last_error = f"HTTP {exc.code}: {detail}"
                raise DiscordError(self.last_error) from exc
            except Exception as exc:  # noqa: BLE001 - network, retried then reported
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2.0**attempt)
                    continue
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise DiscordError(self.last_error) from exc
        self.last_error = "gave up after repeated rate limits"
        raise DiscordError(self.last_error)

    # -- the two operations ------------------------------------------------

    def send(self, content: str) -> list[str]:
        """Post ``content``, split if needed. Returns the message ids created."""
        ids: list[str] = []
        for chunk in split_message(content):
            payload = self._call(
                "POST", f"/channels/{self.channel_id}/messages", {"content": chunk}
            )
            self.sent += 1
            if isinstance(payload, dict) and payload.get("id"):
                ids.append(str(payload["id"]))
        return ids

    def fetch(self, *, after: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Messages newer than ``after``, oldest first.

        Discord returns newest-first; they are reversed here so a burst of
        messages is answered in the order they were typed.
        """
        query: dict[str, Any] = {"limit": max(1, min(100, limit))}
        if after:
            query["after"] = after
        payload = self._call(
            "GET", f"/channels/{self.channel_id}/messages?{urllib.parse.urlencode(query)}"
        )
        rows = list(payload or [])
        rows.reverse()
        self.received += len(rows)
        return rows

    def latest_id(self) -> str | None:
        """The newest message id right now, used to start without a backlog."""
        rows = self.fetch(limit=1)
        return str(rows[-1]["id"]) if rows else None

    def describe(self) -> dict[str, Any]:
        return {
            "channel": self.channel_id,
            "sent": self.sent,
            "received": self.received,
            "last_error": self.last_error,
        }


class DiscordError(RuntimeError):
    """A Discord call that could not be completed."""


@dataclass
class DiscordSink:
    """A :class:`~itsbob.daemon.notify.Sink` that posts to the channel.

    Returns False rather than raising when Discord is unreachable, matching
    every other sink: one dead channel must not stop a desktop notification or
    the file log from getting the same message.
    """

    client: DiscordClient
    #: Prefix urgent notices so they are visible in a busy channel.
    mention_on_urgent: str = ""

    def send(self, notification: Any) -> bool:
        title = getattr(notification, "title", "") or "itsbob"
        body = getattr(notification, "body", "") or ""
        urgency = getattr(notification, "urgency", "normal")
        prefix = (
            f"{self.mention_on_urgent} " if urgency == "high" and self.mention_on_urgent else ""
        )
        try:
            self.client.send(f"{prefix}**{title}**\n{body}".strip())
        except DiscordError:
            return False
        return True


@dataclass
class DiscordBridge:
    """Polls the channel and turns what people type there into turns.

    Runs on its own thread. ``submit`` is the session's queue, so a Discord
    message and a browser message contend for exactly one agent, in the order
    they arrived — the same single serialization point everything else uses.
    """

    client: DiscordClient
    submit: Callable[..., Any]
    #: Seconds between polls. Discord's rate limit is generous; this is chosen
    #: for how quickly a person expects a reply, not for what the API allows.
    interval: float = 5.0
    #: Ignore anything already in the channel when starting, so a restart does
    #: not answer a week of backlog at once.
    skip_backlog: bool = True
    #: Only these user ids may drive the assistant. Empty means anyone who can
    #: post in the channel can — which is the right default for a private
    #: channel and the wrong one for a public server, hence the switch.
    allowed_users: frozenset[str] = frozenset()
    after: str | None = None
    running: bool = False
    polls: int = 0
    handled: int = 0
    errors: int = 0
    last_error: str | None = None
    _thread: Any = field(default=None, repr=False)
    _stop: Any = field(default_factory=threading.Event, repr=False)

    def start(self) -> bool:
        if self.running:
            return False
        if self.skip_backlog and self.after is None:
            try:
                self.after = self.client.latest_id()
            except DiscordError as exc:
                self.last_error = str(exc)
                return False
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, name="itsbob-discord", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self.running = False
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - a bridge must not die on one bad poll
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            self._stop.wait(self.interval)
        self.running = False

    def poll_once(self) -> int:
        """One poll. Returns how many messages were submitted as turns."""
        self.polls += 1
        rows = self.client.fetch(after=self.after)
        submitted = 0
        if rows:
            # The cursor moves once, to the newest id in the batch — including
            # past anything skipped, or a bot message would be re-fetched
            # forever. Set per row it could move backwards if a batch ever
            # arrived out of order.
            self.after = str(rows[-1].get("id") or self.after)
        for row in rows:
            if not self._is_input(row):
                continue
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            author = str((row.get("author") or {}).get("username") or "someone")
            self.handled += 1
            submitted += 1
            self.submit(
                content,
                source="discord",
                label=f"discord ({author}): {content[:50]}",
                on_done=self._reply,
            )
        return submitted

    def _is_input(self, row: Mapping[str, Any]) -> bool:
        """Whether a message should become a turn.

        The bot's own posts are the important exclusion: without it, every
        answer becomes a new question and the loop never ends.
        """
        author = row.get("author") or {}
        if author.get("bot"):
            return False
        if self.allowed_users and str(author.get("id")) not in self.allowed_users:
            return False
        return not str(row.get("content") or "").startswith("//")  # a convention for notes

    def _reply(self, turn: Any, error: str | None) -> None:
        text = getattr(turn, "final", "") if turn is not None else ""
        if error:
            text = f"⚠️ That went wrong: {error}"
        if not text:
            return
        try:
            self.client.send(text)
        except DiscordError as exc:
            self.errors += 1
            self.last_error = str(exc)

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "interval": self.interval,
            "polls": self.polls,
            "handled": self.handled,
            "errors": self.errors,
            "last_error": self.last_error or self.client.last_error,
            **self.client.describe(),
        }


# -- the tool --------------------------------------------------------------


def discord_tools(client: DiscordClient | None = None) -> list[Any]:
    """``discord_post``, so Bob can start a conversation rather than only finish one.

    Gated as a network write and *not* auto-allowed: posting is visible to other
    people and cannot be taken back, which is exactly the shape of action the
    confirmation gate exists for.
    """
    from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

    def run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        active = client or DiscordClient.from_env(ctx.env)
        if active is None:
            raise ToolError(
                "Discord is not configured — set DISCORD_BOT_TOKEN and "
                "DISCORD_CHANNEL_ID in .env and restart"
            )
        message = str(params.get("message", "")).strip()
        if not message:
            raise ToolError("message is empty")
        try:
            ids = active.send(message)
        except DiscordError as exc:
            raise ToolError(str(exc)) from exc
        return ToolResult(
            ok=True,
            output=f"posted to Discord ({len(ids)} message{'s' if len(ids) != 1 else ''})",
            data={"ids": ids},
        )

    return [
        Tool(
            name="discord_post",
            description=(
                "Post a message to the Discord channel. This is how you reach the "
                "user when they have not asked you anything — a finding, a "
                "reminder, something you noticed. Markdown works; long messages "
                "are split automatically."
            ),
            run=run,
            risk=Risk.NETWORK,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        )
    ]
