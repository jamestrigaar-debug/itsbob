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
from typing import Any, Callable, Mapping, Sequence

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
    _me: dict[str, Any] | None = field(default=None, repr=False)

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
                self.last_error = _explain(exc.code, detail, self.channel_id)
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

    def me(self) -> dict[str, Any]:
        """The bot's own account, cached. Needed to recognise being tagged.

        Cached because it cannot change while the process runs, and this is on
        the path of every poll.
        """
        if self._me is None:
            self._me = dict(self._call("GET", "/users/@me") or {})
        return self._me

    @property
    def user_id(self) -> str:
        try:
            return str(self.me().get("id") or "")
        except DiscordError:
            return ""

    def reply_to(self, message_id: str, content: str) -> list[str]:
        """Post as a threaded reply, so an answer sits under its question.

        In a channel with any traffic at all, a bare message is an answer with
        no visible question. ``fail_if_not_exists: False`` means a deleted
        original degrades to an ordinary post rather than losing the reply.
        """
        ids: list[str] = []
        for index, chunk in enumerate(split_message(content)):
            body: dict[str, Any] = {"content": chunk}
            if index == 0 and message_id:
                body["message_reference"] = {
                    "message_id": str(message_id),
                    "fail_if_not_exists": False,
                }
            payload = self._call("POST", f"/channels/{self.channel_id}/messages", body)
            self.sent += 1
            if isinstance(payload, dict) and payload.get("id"):
                ids.append(str(payload["id"]))
        return ids

    def check(self) -> tuple[bool, str]:
        """Can this bot actually see the channel? One cheap read, for `doctor`.

        A token and a channel id in `.env` prove nothing — the first real sign
        of trouble was otherwise a failed `discord_post` mid-conversation.
        """
        try:
            self.fetch(limit=1)
        except DiscordError as exc:
            return False, str(exc)
        return True, f"channel {self.channel_id} is reachable"

    def intent_check(self, *, limit: int = 25) -> tuple[bool | None, str]:
        """Can it read what people *type*? A different permission from reading
        the channel, and the one that fails silently.

        Returns ``None`` when there is nothing recent to judge from — no
        opinion is the honest answer there, and claiming either one would be
        worse than saying so.
        """
        try:
            rows = self.fetch(limit=limit)
        except DiscordError as exc:
            return False, str(exc)

        from_people = [r for r in rows if not (r.get("author") or {}).get("bot")]
        if not from_people:
            return None, "no recent messages from a person to judge from"

        withheld = [r for r in from_people if content_withheld(r)]
        if not withheld:
            return True, f"read the text of {len(from_people)} recent message(s)"
        return False, (
            f"{len(withheld)} of {len(from_people)} recent message(s) arrived with "
            f"their text stripped out — {INTENT_FIX}"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "channel": self.channel_id,
            "sent": self.sent,
            "received": self.received,
            "last_error": self.last_error,
        }


class DiscordError(RuntimeError):
    """A Discord call that could not be completed."""


#: Discord's failures are all "404" or "403" with a two-word body, and every
#: one of them has a different fix that is not guessable from the code. Real
#: transcript: `Unknown Channel (HTTP 404)` — which is what you get from a
#: *server* id pasted where a *channel* id goes, from a bot that was never
#: invited, and from a channel it cannot see, three unrelated problems with one
#: message. Saying which checks to make is the difference between a two-minute
#: fix and giving up on the integration.
#: What Discord says when a bot without the Message Content intent is told to
#: read a channel. Worth spelling out in one place: it is the single most
#: common reason a working bot appears to ignore people at random.
INTENT_FIX = (
    "turn on MESSAGE CONTENT INTENT under Bot in the Discord Developer Portal "
    "(discord.com/developers/applications), then restart itsbob"
)


def content_withheld(row: Mapping[str, Any]) -> bool:
    """Whether Discord stripped this message's text before handing it over.

    Discord will not accept a post with no text, no attachment, no embed, no
    sticker and no poll — there is nothing to post. So a message that comes
    back with none of those did not arrive that way: it arrived with content
    this bot is not permitted to read, which is what the Message Content
    intent gates.

    That makes it a *definite* test rather than a guess, and it is decidable
    from one message. The distinction matters, because the intent is not all
    or nothing: a bot without it still receives full content for messages that
    tag it, and blanks for everything else. So the channel works when you @ it
    and appears to ignore you when you do not — and any check that waits for
    *every* message to be blank will sit there never firing, because the
    tagged ones are never blank.
    """
    if (row.get("author") or {}).get("bot"):
        return False
    if str(row.get("content") or "").strip():
        return False
    return not any(
        row.get(key) for key in ("attachments", "embeds", "sticker_items", "poll")
    )


def _explain(code: int, detail: str, channel_id: str) -> str:
    body = detail.lower()
    if code == 404 and "unknown channel" in body:
        return (
            f"Discord does not recognise channel {channel_id} (HTTP 404, Unknown Channel). "
            "One of three things, in the order worth checking:\n"
            "  1. DISCORD_CHANNEL_ID is a server id, not a channel id. Right-click the "
            "*channel* in the sidebar (not the server) → Copy Channel ID. You need "
            "Developer Mode on: Settings → Advanced → Developer Mode.\n"
            "  2. The bot was never invited to that server. Generate an invite in the "
            "Developer Portal → OAuth2 → URL Generator, scopes `bot`, permissions "
            "'View Channel' and 'Send Messages', then open the URL.\n"
            "  3. The bot is in the server but cannot see that channel — check the "
            "channel's permissions for the bot's role."
        )
    if code == 401:
        return (
            "Discord rejected the bot token (HTTP 401). It has been reset or was "
            "mistyped — Developer Portal → your application → Bot → Reset Token, then "
            "put the new value in DISCORD_BOT_TOKEN. Note it is the *bot* token, not "
            "the application id or the client secret."
        )
    if code == 403:
        return (
            f"The bot may not post in channel {channel_id} (HTTP 403, Forbidden). It is "
            "in the server but its role lacks 'Send Messages' there — check the "
            "channel's permission overrides."
        )
    if code == 400 and "content" in body:
        return f"Discord rejected the message body (HTTP 400): {detail}"
    return f"HTTP {code}: {detail}"


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
    #: Answer only when tagged. Off by default, because the common setup is a
    #: channel that exists for talking to itsbob, where making people @ it every
    #: time is pure ceremony. Turn it on (ITSBOB_DISCORD_MENTION_ONLY=1) for a
    #: shared channel where it should stay quiet until spoken to. A mention is
    #: always answered either way — that is what tagging *means*.
    mention_only: bool = False
    after: str | None = None
    running: bool = False
    polls: int = 0
    handled: int = 0
    mentions: int = 0
    errors: int = 0
    last_error: str | None = None
    #: Set when Discord returned messages whose content was blank — the
    #: signature of a bot without the Message Content intent.
    content_warning: str | None = None
    #: How many messages were dropped because their text never arrived. Counted
    #: rather than merely warned about: "it missed 6 things" is a report, and
    #: it is what tells you the intent is still off after you thought you fixed it.
    withheld: int = 0
    #: The cross-process claim on this channel. Without it, `itsbob serve` and
    #: the browser's continuous mode both poll and both answer — two turns, two
    #: replies, one of them pure waste. ``None`` disables the check.
    lease: Any = None
    #: True while another process holds the lease. Still polling the clock, not
    #: the channel, so a crash over there is picked up here.
    standby: bool = False
    #: Replies that Discord did not accept yet.  Keeping the text lets the
    #: bridge retry delivery without rerunning an expensive agent turn.
    _pending_replies: dict[str, str] = field(default_factory=dict, repr=False)
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
        if self.lease is not None:
            # Released rather than left to expire, so the other process picks
            # the channel up on its next poll instead of in ninety seconds.
            self.lease.release()

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
        if self.lease is not None and not self.lease.hold():
            # Someone else is answering this channel. Stand by rather than
            # exit: if they stop renewing, the next poll takes over.
            if not self.standby:
                self.last_error = f"standing by — {self.lease.held_by} is answering Discord"
            self.standby = True
            # Follow the holder's progress rather than the clock. Jumping to
            # `latest_id()` here spent an API call per poll doing nothing, and
            # silently discarded anything the holder had not answered yet — so
            # a holder that died mid-turn took those messages with it. The
            # lease already records what was answered, and that is the precise
            # boundary: everything behind it is done, everything after is still
            # owed an answer, and a takeover picks up exactly those.
            self.after = self.lease.last_answered() or self.after
            return 0
        if self.standby:
            self.standby = False
            self.last_error = None
        self._retry_replies()
        rows = self.client.fetch(after=self.after)
        submitted = 0
        if rows:
            # The cursor moves once, to the newest id in the batch — including
            # past anything skipped, or a bot message would be re-fetched
            # forever. Set per row it could move backwards if a batch ever
            # arrived out of order.
            self.after = str(rows[-1].get("id") or self.after)
        self._check_content_intent(rows)
        for row in rows:
            if not self._is_input(row):
                continue
            tagged = self.mentions_us(row)
            if self.mention_only and not tagged:
                continue
            if content_withheld(row):
                # Not an empty message — an unreadable one. Dropping it in
                # silence is what made this look like flakiness rather than a
                # permission, so it is counted and named instead.
                self.withheld += 1
                continue
            content = self._clean(row)
            if not content:
                continue
            message_id_check = str(row.get("id") or "")
            if self.lease is not None and self.lease.already_answered(message_id_check):
                # Answered by whoever held the lease before this one did. The
                # handover window is the only way to get here, and re-answering
                # is exactly what this whole mechanism exists to prevent.
                continue
            author = str((row.get("author") or {}).get("username") or "someone")
            self.handled += 1
            if tagged:
                self.mentions += 1
            submitted += 1
            message_id = message_id_check
            try:
                accepted = self.submit(
                    content,
                    source="discord",
                    label=f"discord ({author}){' @you' if tagged else ''}: {content[:44]}",
                    on_done=lambda turn, error, mid=message_id: self._reply(turn, error, mid),
                )
            except Exception as exc:  # noqa: BLE001 - retain it for a later poll
                accepted = False
                self.last_error = f"could not queue Discord message: {type(exc).__name__}: {exc}"[:200]
            if accepted is False:
                # The cursor has advanced, so an explicit retry queue is what
                # prevents a full inbox from silently dropping this message.
                self._pending_replies[message_id] = "⚠️ The assistant is busy; please retry shortly."
        return submitted

    def mentions_us(self, row: Mapping[str, Any]) -> bool:
        """Whether this message tagged the bot.

        Checked against the `mentions` array Discord supplies rather than by
        scanning the text: a mention is `<@123>` or `<@!123>` depending on age
        and client, and someone typing the bot's display name is not a mention
        at all.
        """
        me = self.client.user_id
        if not me:
            return False
        for mentioned in row.get("mentions") or []:
            if str((mentioned or {}).get("id")) == me:
                return True
        # A reply to one of the bot's own messages is a mention in every sense
        # that matters, and Discord does not always populate `mentions` for it.
        referenced = row.get("referenced_message") or {}
        return str((referenced.get("author") or {}).get("id") or "") == me

    def _clean(self, row: Mapping[str, Any]) -> str:
        """The message with the bot's own tag removed.

        "@itsbob what is the score" should reach the agent as "what is the
        score" — the tag is addressing, not content, and leaving it in invites
        the model to wonder who `<@1543…>` is.
        """
        content = str(row.get("content") or "")
        me = self.client.user_id
        if me:
            for form in (f"<@{me}>", f"<@!{me}>"):
                content = content.replace(form, " ")
        return " ".join(content.split())

    def _check_content_intent(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Notice a bot that cannot read what people type.

        Discord withholds message content from apps without the Message
        Content intent — except in messages that tag the bot. So the symptom
        is not silence, which somebody would investigate; it is a channel that
        answers when tagged and ignores you when not, with no error anywhere,
        which reads as flakiness. It is a checkbox in the Developer Portal,
        and worth naming rather than leaving to be guessed at.
        """
        if self.content_warning:
            return
        withheld = [row for row in rows if content_withheld(row)]
        if not withheld:
            return
        self.content_warning = (
            f"Discord handed over {len(withheld)} message(s) with the text removed. "
            f"That is the Message Content intent being off: {INTENT_FIX}. Until then "
            "itsbob can only read messages that tag it, which is why it answers "
            "sometimes and not others."
        )

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

    def _reply(self, turn: Any, error: str | None, message_id: str = "") -> None:
        text = getattr(turn, "final", "") if turn is not None else ""
        if error:
            text = f"⚠️ That went wrong: {error}"
        if not text:
            text = "⚠️ The assistant did not produce a response. Please try again."
        try:
            self.client.reply_to(message_id, text)
        except DiscordError as exc:
            self.errors += 1
            self.last_error = str(exc)
            self._pending_replies[message_id] = text
            return
        if self.lease is not None:
            # A message counts as answered only after Discord accepted its reply.
            self.lease.mark_answered(message_id)
        self._pending_replies.pop(message_id, None)

    def _retry_replies(self) -> None:
        """Retry a failed Discord post before accepting new questions."""
        for message_id, text in list(self._pending_replies.items()):
            try:
                self.client.reply_to(message_id, text)
            except DiscordError as exc:
                self.errors += 1
                self.last_error = str(exc)
                continue
            if self.lease is not None:
                self.lease.mark_answered(message_id)
            self._pending_replies.pop(message_id, None)

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "interval": self.interval,
            "polls": self.polls,
            "handled": self.handled,
            "mentions": self.mentions,
            "mention_only": self.mention_only,
            "errors": self.errors,
            "last_error": self.last_error or self.client.last_error,
            "content_warning": self.content_warning,
            "withheld": self.withheld,
            "standby": self.standby,
            "lease": self.lease.describe() if self.lease is not None else None,
            **self.client.describe(),
        }

    @classmethod
    def from_env(
        cls,
        submit: Callable[..., Any],
        env: Mapping[str, str] | None = None,
        *,
        home: Any = None,
        role: str = "unknown",
    ) -> "DiscordBridge | None":
        env = os.environ if env is None else env
        client = DiscordClient.from_env(env)
        if client is None:
            return None
        lease = None
        if home is not None:
            from pathlib import Path

            from .lease import DiscordLease

            lease = DiscordLease(path=Path(home) / "discord.lease", role=role)
        return cls(
            client=client,
            submit=submit,
            lease=lease,
            mention_only=str(env.get("ITSBOB_DISCORD_MENTION_ONLY", "")).strip().lower()
            in ("1", "true", "yes", "on"),
            allowed_users=frozenset(
                u.strip() for u in str(env.get("ITSBOB_DISCORD_USERS", "")).split(",") if u.strip()
            ),
        )


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
        policy = getattr(ctx, "policy", None)
        if policy is not None:
            host_reason = policy.check_url(API)
            if host_reason:
                raise ToolError(host_reason)
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
