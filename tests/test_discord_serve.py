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


# -- one channel, one answer ----------------------------------------------


def test_two_bridges_on_one_channel_answer_exactly_once(discord, tmp_path):
    """The reported bug: `itsbob serve` and the browser's continuous mode each
    poll, each run a turn, and each post. Two replies to every message, not even
    the same reply — separate turns with separate state, so one recommended a
    different film from the other. The second is a whole turn's tokens spent on
    something nobody asked for."""

    served, browsed = [], []
    daemon = DiscordBridge.from_env(
        lambda t, **k: served.append((t, k)), home=tmp_path, role="daemon")
    gui = DiscordBridge.from_env(
        lambda t, **k: browsed.append((t, k)), home=tmp_path, role="browser")
    daemon.skip_backlog = gui.skip_backlog = False

    discord.human("what is my favourite manga", mentions_bot=True)
    assert daemon.poll_once() == 1     # whoever gets there first answers
    assert gui.poll_once() == 0        # the other stands by
    assert gui.standby is True
    assert "standing by" in (gui.last_error or "")
    assert len(served) == 1 and browsed == []

    # And the standby one still knows it is alive and why.
    assert gui.status()["standby"] is True
    assert gui.status()["lease"]["holder"] == "daemon"

    # Only one reply reaches the channel.
    class Turn:
        final = "Berserk."

    served[0][1]["on_done"](Turn(), None)
    with discord.lock:
        assert [p["content"] for p in discord.posted] == ["Berserk."]


def test_the_standby_bridge_takes_over_when_the_holder_dies(discord, tmp_path):
    """Standing by must not mean Discord goes quiet if the holder crashes."""
    served, browsed = [], []
    daemon = DiscordBridge.from_env(
        lambda t, **k: served.append((t, k)), home=tmp_path, role="daemon")
    gui = DiscordBridge.from_env(
        lambda t, **k: browsed.append(t), home=tmp_path, role="browser")
    daemon.skip_backlog = gui.skip_backlog = False
    daemon.lease.ttl = gui.lease.ttl = 0.2

    discord.human("first question", mentions_bot=True)
    daemon.poll_once()
    gui.poll_once()
    assert served[0][0] == "first question" and browsed == []

    class Turn:
        final = "First answer."

    served[0][1]["on_done"](Turn(), None)

    time.sleep(0.3)  # the daemon stops renewing
    discord.human("second question", mentions_bot=True)
    assert gui.poll_once() == 1
    assert browsed == ["second question"]
    assert gui.standby is False


def test_a_clean_stop_hands_the_channel_over_at_once(discord, tmp_path):
    served, browsed = [], []
    daemon = DiscordBridge.from_env(
        lambda t, **k: served.append((t, k)), home=tmp_path, role="daemon")
    gui = DiscordBridge.from_env(
        lambda t, **k: browsed.append(t), home=tmp_path, role="browser")
    daemon.skip_backlog = gui.skip_backlog = False

    discord.human("one", mentions_bot=True)
    daemon.poll_once()
    gui.poll_once()
    assert gui.standby

    class Turn:
        final = "First answer."

    served[0][1]["on_done"](Turn(), None)

    daemon.stop()  # releases rather than waiting ninety seconds to expire
    discord.human("two", mentions_bot=True)
    assert gui.poll_once() == 1
    assert browsed == ["two"]


def test_a_handover_does_not_re_answer_what_was_already_answered(discord, tmp_path):
    """The one window where both could act, and the one thing it must not do."""
    served, browsed = [], []
    daemon = DiscordBridge.from_env(
        lambda t, **k: served.append((t, k)), home=tmp_path, role="daemon")
    gui = DiscordBridge.from_env(
        lambda t, **k: browsed.append(t), home=tmp_path, role="browser")
    daemon.skip_backlog = gui.skip_backlog = False

    discord.human("only answer me once", mentions_bot=True)
    daemon.poll_once()
    assert served[0][0] == "only answer me once"

    class Turn:
        final = "Once."

    served[0][1]["on_done"](Turn(), None)

    # The browser takes over with its cursor still behind, so it sees the same
    # message again. The answered-id trail is what stops it running a turn.
    daemon.lease.release()
    gui.after = None
    assert gui.poll_once() == 0
    assert browsed == []


def test_a_single_bridge_with_no_home_still_works(discord):
    """The lease is an optimisation for a shared machine, not a requirement."""
    submitted = []
    bridge = DiscordBridge.from_env(lambda t, **k: submitted.append(t))
    bridge.skip_backlog = False
    assert bridge.lease is None
    discord.human("hello", mentions_bot=True)
    assert bridge.poll_once() == 1
    assert submitted == ["hello"]


def test_an_unwritable_home_answers_rather_than_going_quiet(tmp_path):
    """One process answering twice is a smaller failure than none answering."""
    from itsbob.integrations.lease import DiscordLease

    lease = DiscordLease(path=tmp_path / "nope" / "x" / "discord.lease", role="daemon")
    lease.path = tmp_path / "\0bad" / "discord.lease"  # unwritable by construction
    assert lease.hold() is True


# -- the intent, which is what "seen sometimes" actually means ---------------


def _gagged(discord, text, *, mentions_bot=False, author_id="U1"):
    """A message as a bot without the Message Content intent receives it.

    Discord still sends the envelope — id, author, mentions — and removes the
    text, unless the message tags the bot, in which case it arrives intact.
    That split is the whole bug: the channel works when you @ it.
    """
    with discord.lock:
        discord.inbox.append(
            {
                "id": f"MSG{len(discord.inbox) + 1}",
                "content": (f"<@BOT1> {text}" if mentions_bot else ""),
                "author": {"id": author_id, "username": "james"},
                "mentions": [{"id": "BOT1"}] if mentions_bot else [],
            }
        )


def test_a_message_with_nothing_in_it_at_all_cannot_be_real(discord):
    from itsbob.integrations.discord import content_withheld

    # Discord refuses to post one of these, so receiving one means the text
    # was taken out on the way. Anything that could have carried the message
    # instead is a real, empty-texted message and must not be flagged.
    assert content_withheld({"content": "", "author": {"id": "U1"}}) is True
    assert content_withheld({"content": "   ", "author": {"id": "U1"}}) is True
    for carrier in ("attachments", "embeds", "sticker_items", "poll"):
        assert content_withheld(
            {"content": "", "author": {"id": "U1"}, carrier: [{"x": 1}]}
        ) is False
    assert content_withheld({"content": "hello", "author": {"id": "U1"}}) is False
    # The bot's own posts are not evidence of anything.
    assert content_withheld({"content": "", "author": {"id": "B", "bot": True}}) is False


def test_the_intent_warning_fires_when_only_some_messages_are_readable(discord):
    """The case the old check could not see.

    It required *every* message from a person to be blank. But a bot without
    the intent reads the tagged ones perfectly, so in any real channel some
    always have content — and the warning that would have explained the whole
    problem sat there never firing.
    """
    submitted = []
    bridge = _bridge(discord, submitted)
    _gagged(discord, "what is the weather")               # unreadable
    _gagged(discord, "and the score", mentions_bot=True)  # readable, because tagged

    assert bridge.poll_once() == 1
    assert [text for text, _ in submitted] == ["and the score"]

    assert bridge.content_warning is not None
    assert "MESSAGE CONTENT INTENT" in bridge.content_warning
    assert "developers" in bridge.content_warning
    # And it says what the person actually observed, so they recognise it.
    assert "sometimes and not others" in bridge.content_warning


def test_unreadable_messages_are_counted_not_silently_dropped(discord):
    submitted = []
    bridge = _bridge(discord, submitted)
    _gagged(discord, "one")
    _gagged(discord, "two")
    _gagged(discord, "three", mentions_bot=True)
    bridge.poll_once()

    assert bridge.withheld == 2
    assert bridge.status()["withheld"] == 2
    # Dropping them in silence is what made a permission problem look like
    # flakiness; the count is what shows it is still happening.
    assert len(submitted) == 1


def test_a_single_unreadable_message_is_enough_to_warn(discord):
    submitted = []
    bridge = _bridge(discord, submitted)
    _gagged(discord, "hello")
    bridge.poll_once()
    assert bridge.content_warning is not None


def test_the_check_separates_reaching_the_channel_from_reading_it(discord):
    """`doctor` reported the first and called it the second."""
    client = DiscordClient(token="t", channel_id="CHAN1")

    ok, detail = client.check()
    assert ok is True
    # It must no longer claim it can read what people type — that is the
    # sentence that made this look configured while it was broken.
    assert "can read it" not in detail

    # Nothing to go on yet: no opinion, rather than a wrong one.
    readable, why = client.intent_check()
    assert readable is None and "no recent messages" in why

    _gagged(discord, "what is the weather")
    readable, why = client.intent_check()
    assert readable is False
    assert "1 of 1" in why and "MESSAGE CONTENT INTENT" in why


def test_the_check_passes_once_the_intent_is_on(discord):
    client = DiscordClient(token="t", channel_id="CHAN1")
    discord.human("what is the weather")
    discord.human("and the score", mentions_bot=True)

    readable, why = client.intent_check()
    assert readable is True
    assert "2 recent message" in why


# -- handover, the other way a message goes missing -------------------------


def test_standby_follows_the_holder_instead_of_the_clock(discord, tmp_path):
    """A takeover must not skip what the previous holder never answered.

    Standby used to jump its cursor to the newest message on every poll. That
    cost an API call per poll from a process doing nothing, and — worse — it
    stepped over everything the holder had not got to yet. A holder that died
    mid-turn took those messages with it, and nobody ever answered them.
    """
    from itsbob.integrations.lease import DiscordLease

    submitted_a, submitted_b = [], []
    holder = _bridge(discord, submitted_a)
    holder.lease = DiscordLease(path=tmp_path / "lease", role="daemon")
    standby = _bridge(discord, submitted_b)
    standby.lease = DiscordLease(path=tmp_path / "lease", role="browser")

    discord.human("first question")
    assert holder.poll_once() == 1          # the holder takes the lease
    assert standby.poll_once() == 0         # and the other stands by
    assert standby.standby is True

    class Turn:
        final = "First answer."

    submitted_a[0][1]["on_done"](Turn(), None)

    # Three more arrive. The holder dies before it polls again, so it never
    # sees them — which is exactly when the handover has to be lossless.
    discord.human("second question")
    discord.human("third question")
    discord.human("fourth question")
    assert standby.poll_once() == 0         # still in standby, lease still fresh
    holder.lease.release()

    assert standby.poll_once() == 3
    assert [text for text, _ in submitted_b] == [
        "second question", "third question", "fourth question",
    ]


def test_a_takeover_does_not_re_answer_what_the_holder_already_did(discord, tmp_path):
    from itsbob.integrations.lease import DiscordLease

    submitted_a, submitted_b = [], []
    holder = _bridge(discord, submitted_a)
    holder.lease = DiscordLease(path=tmp_path / "lease", role="daemon")
    standby = _bridge(discord, submitted_b)
    standby.lease = DiscordLease(path=tmp_path / "lease", role="browser")

    discord.human("answered by the first one")
    holder.poll_once()
    standby.poll_once()

    class Turn:
        final = "Done."

    submitted_a[0][1]["on_done"](Turn(), None)
    holder.lease.release()

    assert standby.poll_once() == 0
    assert submitted_b == []


def test_the_lease_remembers_how_far_the_holder_got(tmp_path):
    from itsbob.integrations.lease import DiscordLease

    lease = DiscordLease(path=tmp_path / "lease", role="daemon")
    assert lease.last_answered() is None
    lease.mark_answered("MSG1")
    lease.mark_answered("MSG2")
    assert lease.last_answered() == "MSG2"
    # It is a high-water mark, so re-marking something older does not move it.
    lease.mark_answered("MSG1")
    assert lease.last_answered() == "MSG2"


def test_standing_by_costs_no_discord_calls(discord, tmp_path):
    from itsbob.integrations.lease import DiscordLease

    holder = _bridge(discord, [])
    holder.lease = DiscordLease(path=tmp_path / "lease", role="daemon")
    standby = _bridge(discord, [])
    standby.lease = DiscordLease(path=tmp_path / "lease", role="browser")

    discord.human("hello")
    holder.poll_once()

    before = standby.client.received
    for _ in range(5):
        standby.poll_once()
    assert standby.client.received == before, "a standby process is polling Discord"
