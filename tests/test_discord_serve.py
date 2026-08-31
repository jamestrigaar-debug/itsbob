"""Discord read and write while `itsbob serve` is running, end to end.

A stand-in Discord speaking the two REST endpoints the bridge uses, driven
through the real :class:`~itsbob.daemon.service.Daemon`. Everything else in the
suite mocks the client; this exercises the path a person actually uses — type in
the channel, the daemon notices, a turn runs, the answer lands back in the
channel as a reply.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from itsbob.integrations.discord import DiscordBridge, DiscordClient


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: A003
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/users/@me"):
            self._json({"id": "BOT1", "username": "itsbob", "bot": True})
            return
        with self.server.lock:
            after = ""
            if "after=" in self.path:
                after = self.path.split("after=")[1].split("&")[0]
            rows = self.server.inbox
            if after:
                index = next(
                    (i for i, r in enumerate(rows) if r["id"] == after), -1
                )
                rows = rows[index + 1 :]
            # Discord answers newest-first.
            self._json(list(reversed(rows[-50:])))

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        with self.server.lock:
            self.server.posted.append(body)
            message = {
                "id": f"BOTMSG{len(self.server.posted)}",
                "content": body.get("content", ""),
                "author": {"id": "BOT1", "username": "itsbob", "bot": True},
                "mentions": [],
            }
            self.server.inbox.append(message)
        self._json(message)


@pytest.fixture
def discord(monkeypatch):
    """A stand-in Discord on loopback, with itsbob pointed at it."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.inbox, server.posted, server.lock = [], [], threading.Lock()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        "itsbob.integrations.discord.API", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_CHANNEL_ID", "CHAN1")

    def human(text, *, mentions_bot=False, author="james", author_id="U1"):
        with server.lock:
            server.inbox.append(
                {
                    "id": f"MSG{len(server.inbox) + 1}",
                    "content": (f"<@BOT1> {text}" if mentions_bot else text),
                    "author": {"id": author_id, "username": author},
                    "mentions": [{"id": "BOT1", "username": "itsbob"}] if mentions_bot else [],
                }
            )

    server.human = human
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _bridge(discord, submitted, **kwargs):
    bridge = DiscordBridge.from_env(
        lambda text, **kw: submitted.append((text, kw)), env={
            "DISCORD_BOT_TOKEN": "t", "DISCORD_CHANNEL_ID": "CHAN1", **kwargs,
        }
    )
    bridge.skip_backlog = False
    return bridge


def test_the_bot_knows_its_own_identity(discord):
    client = DiscordClient(token="t", channel_id="CHAN1")
    assert client.user_id == "BOT1"
    assert client.me()["username"] == "itsbob"


def test_a_plain_message_is_answered_and_a_tag_is_stripped(discord):
    submitted = []
    bridge = _bridge(discord, submitted)
    discord.human("what is the weather")
    discord.human("what is the score", mentions_bot=True)
    assert bridge.poll_once() == 2

    assert submitted[0][0] == "what is the weather"
    # The tag is addressing, not content: it must not reach the model.
    assert submitted[1][0] == "what is the score"
    assert "<@BOT1>" not in submitted[1][0]
    assert "@you" in submitted[1][1]["label"]
    assert bridge.mentions == 1


def test_being_tagged_always_gets_a_reply_even_in_mention_only_mode(discord):
    """That is what tagging means, and it is the point of the mode."""
    submitted = []
    bridge = _bridge(discord, submitted, ITSBOB_DISCORD_MENTION_ONLY="1")
    assert bridge.mention_only is True

    discord.human("just chatting to someone else in here")
    discord.human("are you there", mentions_bot=True)
    bridge.poll_once()

    assert [text for text, _ in submitted] == ["are you there"]
    assert bridge.mentions == 1


def test_a_reply_to_the_bot_counts_as_tagging_it(discord):
    """Discord does not always populate `mentions` for a reply, and hitting
    reply on someone's message is unambiguously talking to them."""
    submitted = []
    bridge = _bridge(discord, submitted, ITSBOB_DISCORD_MENTION_ONLY="1")
    with discord.lock:
        discord.inbox.append({
            "id": "MSG9", "content": "and what about tomorrow",
            "author": {"id": "U1", "username": "james"}, "mentions": [],
            "referenced_message": {"id": "B1", "author": {"id": "BOT1", "bot": True}},
        })
    bridge.poll_once()
    assert [text for text, _ in submitted] == ["and what about tomorrow"]


def test_the_answer_is_posted_back_as_a_threaded_reply(discord):
    submitted = []
    bridge = _bridge(discord, submitted)
    discord.human("what is the score", mentions_bot=True)
    bridge.poll_once()

    class Turn:
        final = "Two-nil."

    submitted[0][1]["on_done"](Turn(), None)
    with discord.lock:
        posted = list(discord.posted)
    assert posted[0]["content"] == "Two-nil."
    # Threaded, so an answer sits under its question in a busy channel.
    assert posted[0]["message_reference"]["message_id"] == "MSG1"
    assert posted[0]["message_reference"]["fail_if_not_exists"] is False


def test_the_bot_never_answers_its_own_reply(discord):
    """Its own post lands back in the channel on the next poll."""
    submitted = []
    bridge = _bridge(discord, submitted)
    discord.human("hello", mentions_bot=True)
    bridge.poll_once()

    class Turn:
        final = "Hello back."

    submitted[0][1]["on_done"](Turn(), None)
    assert bridge.poll_once() == 0  # its own message is not a question
    assert len(submitted) == 1


def test_a_missing_message_content_intent_is_named(discord):
    """The symptom is a bot that only ever answers when @-ed, with no error."""
    submitted = []
    bridge = _bridge(discord, submitted)
    with discord.lock:
        for i in range(3):
            discord.inbox.append({
                "id": f"BLANK{i}", "content": "",
                "author": {"id": "U1", "username": "james"}, "mentions": [],
            })
    bridge.poll_once()
    assert bridge.content_warning
    assert "MESSAGE CONTENT INTENT" in bridge.content_warning
    assert bridge.status()["content_warning"]


def test_only_named_users_may_drive_it_when_that_is_set(discord):
    submitted = []
    bridge = _bridge(discord, submitted, ITSBOB_DISCORD_USERS="U1")
    discord.human("from the owner", author_id="U1")
    discord.human("from a stranger", author_id="U999")
    bridge.poll_once()
    assert [t for t, _ in submitted] == ["from the owner"]


# -- through the real daemon ----------------------------------------------


def test_serving_reads_and_writes_discord_end_to_end(discord, tmp_path):
    """Type in the channel; `itsbob serve` answers there. No mocks in between."""
    from itsbob.daemon.service import Daemon
    from itsbob.daemon.tasks import TaskStore

    class Agent:
        class toolbox:  # noqa: N801 - a stand-in for the real shape
            class policy:
                class mode:
                    value = "guarded"

        conversation = None
        brain = None

        def chat(self, message, **kwargs):
            class Turn:
                final = f"You said: {message}"
                error = None
                tools_used = ()

            return Turn()

    daemon = Daemon(
        agent=Agent(), tasks=TaskStore(tmp_path / "t.sqlite"), home=tmp_path,
        sink=None, gate=None, handle_signals=False, remember_runs=False,
    )
    daemon.discord = DiscordBridge.from_env(daemon.submit_message)
    daemon.discord.skip_backlog = False
    assert daemon.discord is not None

    discord.human("hello over discord", mentions_bot=True)
    daemon.discord.poll_once()      # inbound: the channel into the daemon's queue
    assert daemon.drain_inbox() == 1  # the daemon runs it as an ordinary turn

    with discord.lock:
        posted = [p["content"] for p in discord.posted]
    assert posted == ["You said: hello over discord"]


def test_the_daemon_starts_and_stops_the_bridge_with_itself(discord, tmp_path):
    from itsbob.daemon.service import Daemon
    from itsbob.daemon.tasks import TaskStore

    class Agent:
        brain = None

    daemon = Daemon(
        agent=Agent(), tasks=TaskStore(tmp_path / "t.sqlite"), home=tmp_path,
        sink=None, gate=None, handle_signals=False,
    )
    daemon.discord = DiscordBridge.from_env(daemon.submit_message)
    daemon._start_discord()
    assert daemon.discord is not None and daemon.discord.running
    daemon.discord.stop()
    time.sleep(0.05)
    assert not daemon.discord.running
