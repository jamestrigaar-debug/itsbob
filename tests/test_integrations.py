"""Weather, news, search, vision, Discord and the messages window. All offline.

Every network boundary here is injectable, which is deliberate rather than
convenient: a test suite that needs a Discord token and a news API key is a
suite nobody runs.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from itsbob.integrations.apis import BUILTIN_SPECS, register_builtins
from itsbob.integrations.briefing import (
    GEOPOLITICS,
    Place,
    fetch_news,
    fetch_weather,
)
from itsbob.integrations.discord import (
    MAX_MESSAGE_CHARS,
    DiscordBridge,
    DiscordClient,
    DiscordSink,
    split_message,
)
from itsbob.tools.http import ApiCatalog, ApiSpec


# -- built-in API specs ----------------------------------------------------


def test_a_key_in_the_environment_is_the_whole_setup():
    catalog = ApiCatalog()
    added = register_builtins(catalog, {"FOOTBALL_DATA_KEY": "x", "GNEWS_API_KEY": "y"})
    assert sorted(added) == ["football", "gnews"]
    football = catalog.get("football")
    assert football.base_url == "https://api.football-data.org/v4"
    assert football.auth == "header" and football.header_name == "X-Auth-Token"
    url, headers = football.build(
        "competitions/PL/matches", params={"dateFrom": "2026-01-01"},
        env={"FOOTBALL_DATA_KEY": "secret"},
    )
    assert headers["X-Auth-Token"] == "secret"
    assert url.endswith("/competitions/PL/matches?dateFrom=2026-01-01")


def test_a_key_that_is_not_set_adds_nothing():
    catalog = ApiCatalog()
    assert register_builtins(catalog, {}) == []
    assert catalog.names() == []


def test_a_hand_written_entry_always_wins():
    """Someone who wrote out a spec meant it; a default must not overwrite it."""
    mine = ApiSpec(name="weather", base_url="http://my-own-proxy/v1", key_env="OPENWEATHER_API_KEY")
    catalog = ApiCatalog({"weather": mine})
    register_builtins(catalog, {"OPENWEATHER_API_KEY": "x"})
    assert catalog.get("weather").base_url == "http://my-own-proxy/v1"


def test_every_builtin_names_the_variable_it_needs():
    for spec in BUILTIN_SPECS:
        assert spec.key_env and spec.description
        assert spec.base_url.startswith("https://")


# -- weather ---------------------------------------------------------------


def test_the_weather_is_condensed_to_a_few_lines():
    """OpenWeather's forecast is 40 rows of 16 fields; four lines is the answer."""
    payloads = {
        "weather": {
            "weather": [{"description": "light rain"}],
            "main": {"temp": 11.4, "feels_like": 9.2},
            "wind": {"speed": 6.0},
        },
        "forecast": {
            "list": [
                {
                    "dt_txt": f"{time.strftime('%Y-%m-%d')} {hour:02d}:00:00",
                    "weather": [{"description": "cloud"}],
                    "main": {"temp": 10 + hour / 10},
                    "rain": {"3h": 0.4},
                }
                for hour in range(0, 24, 3)
            ]
            + [{"dt_txt": "2099-01-01 00:00:00", "weather": [{"description": "no"}]}],
        },
    }

    def fetch(url, headers=None):
        return payloads["forecast" if "/forecast" in url else "weather"]

    weather = fetch_weather(key="k", place=Place("Hull, UK", 53.7, -0.3), fetch=fetch)
    assert weather.place == "Hull, UK"
    assert "Light rain" in weather.summary
    assert round(weather.wind_mph) == 13  # 6 m/s
    assert 1 <= len(weather.outlook) <= 4  # today only, every second entry
    assert "2099" not in weather.render()


# -- news ------------------------------------------------------------------


def _newsapi(titles):
    return {
        "articles": [
            {
                "title": t,
                "source": {"name": "Wire"},
                "url": f"https://example.test/{i}",
                "publishedAt": f"2026-08-30T1{i}:00:00Z",
                "description": "x" * 400,
                "content": "y" * 4000,
                "urlToImage": "https://example.test/huge.jpg",
            }
            for i, t in enumerate(titles)
        ]
    }


def test_news_merges_sources_and_drops_the_same_story_twice():
    def fetch(url, headers=None):
        if "newsapi" in url:
            return _newsapi(["Ceasefire agreed in border talks", "Election result confirmed"])
        return {
            "articles": [
                {
                    "title": "Ceasefire agreed in border talks",  # same story, other wire
                    "source": {"name": "GNews"},
                    "url": "https://other.test/1",
                    "publishedAt": "2026-08-30T09:00:00Z",
                    "description": "z",
                }
            ]
        }

    headlines, problems = fetch_news(
        newsapi_key="a", gnews_key="b", limit=10, fetch=fetch
    )
    assert problems == []
    assert len(headlines) == 2
    # And the 4KB `content` blob and image URL never make it into the output.
    rendered = "\n".join(h.render() for h in headlines)
    assert "yyyy" not in rendered and "huge.jpg" not in rendered
    assert len(headlines[0].summary) <= 221


def test_one_dead_news_source_does_not_cost_the_other_its_results():
    from itsbob.tools.base import ToolError

    def fetch(url, headers=None):
        if "newsapi" in url:
            raise ToolError("HTTP 429 from newsapi.org")
        return {"articles": [{"title": "Summit opens", "source": {"name": "GNews"}}]}

    headlines, problems = fetch_news(newsapi_key="a", gnews_key="b", fetch=fetch)
    assert [h.title for h in headlines] == ["Summit opens"]
    assert problems and "newsapi" in problems[0]


def test_no_key_at_all_is_reported_rather_than_returning_nothing():
    headlines, problems = fetch_news(fetch=lambda *a, **k: {})
    assert headlines == []
    assert "NEWSAPI_KEY" in problems[0]


def test_the_default_beat_is_geopolitics_and_large_events():
    assert "geopolitics" in GEOPOLITICS and "earthquake" in GEOPOLITICS


# -- discord ---------------------------------------------------------------


def test_a_long_message_is_split_on_a_paragraph_not_mid_word():
    text = "\n\n".join("paragraph " + "word " * 100 for _ in range(6))
    chunks = split_message(text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_MESSAGE_CHARS for c in chunks)
    assert not any(c.endswith("wor") for c in chunks)
    # Nothing is lost in the split.
    assert sum(len(c.split()) for c in chunks) == len(text.split())


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client(responses):
    """A client whose opener replays canned responses and records requests."""
    sent = []

    def opener(request, timeout=None):
        sent.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "body": json.loads(request.data) if request.data else None,
                "auth": request.headers.get("Authorization"),
            }
        )
        return FakeResponse(responses.pop(0) if responses else {})

    client = DiscordClient(token="tok", channel_id="99", opener=opener)
    return client, sent


def test_sending_posts_to_the_channel_with_the_bot_token():
    client, sent = _client([{"id": "1"}])
    assert client.send("hello") == ["1"]
    assert sent[0]["method"] == "POST"
    assert sent[0]["url"].endswith("/channels/99/messages")
    assert sent[0]["auth"] == "Bot tok"
    assert sent[0]["body"] == {"content": "hello"}


def test_fetching_returns_oldest_first():
    """Discord answers newest-first; a burst must be answered in typing order."""
    client, _ = _client([[{"id": "3"}, {"id": "2"}, {"id": "1"}]])
    assert [r["id"] for r in client.fetch()] == ["1", "2", "3"]


def test_the_bridge_never_answers_its_own_messages():
    """Without this the bot replies to itself, forever."""
    # Newest first, as Discord itself returns them.
    rows = [
        {"id": "3", "content": "// a note to self", "author": {"username": "james"}},
        {"id": "2", "content": "hi james", "author": {"username": "bob", "bot": True}},
        {"id": "1", "content": "hello bob", "author": {"username": "james", "id": "7"}},
    ]
    client, _ = _client([rows])
    submitted = []
    bridge = DiscordBridge(
        client=client,
        submit=lambda text, **kw: submitted.append((text, kw)),
        skip_backlog=False,
    )
    assert bridge.poll_once() == 1
    assert submitted[0][0] == "hello bob"
    assert submitted[0][1]["source"] == "discord"
    assert bridge.after == "3"  # the cursor still advances past what it skipped


def test_the_bridge_can_be_limited_to_named_users():
    rows = [
        {"id": "2", "content": "and this", "author": {"username": "james", "id": "7"}},
        {"id": "1", "content": "run this", "author": {"username": "stranger", "id": "999"}},
    ]
    client, _ = _client([rows])
    submitted = []
    bridge = DiscordBridge(
        client=client,
        submit=lambda text, **kw: submitted.append(text),
        skip_backlog=False,
        allowed_users=frozenset({"7"}),
    )
    bridge.poll_once()
    assert submitted == ["and this"]


def test_a_rate_limit_is_waited_out_and_retried(monkeypatch):
    import urllib.error

    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    calls = {"n": 0}

    def opener(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {},
                __import__("io").BytesIO(json.dumps({"retry_after": 2.5}).encode()),
            )
        return FakeResponse({"id": "ok"})

    client = DiscordClient(token="t", channel_id="1", opener=opener)
    assert client.send("hi") == ["ok"]
    assert slept == [2.5]


def test_a_dead_channel_does_not_break_the_other_sinks():
    """One unreachable channel must not stop the desktop toast or the log."""
    def opener(request, timeout=None):
        raise OSError("no route to host")

    client = DiscordClient(token="t", channel_id="1", opener=opener)
    from itsbob.daemon.notify import Notification

    with pytest.MonkeyPatch.context() as m:
        m.setattr(time, "sleep", lambda *_: None)
        assert DiscordSink(client=client).send(Notification("t", "b")) is False


# -- the messages window ---------------------------------------------------


def test_messages_are_read_with_ids_and_read_state(tmp_path):
    from itsbob.daemon.notify import FileSink, Notification
    from itsbob.gui.messages import MessageLog

    sink = FileSink(path=tmp_path / "notifications.jsonl")
    for i in range(3):
        sink.send(Notification(title=f"n{i}", body="b", task="t"))

    log = MessageLog(tmp_path / "notifications.jsonl")
    rows = log.recent()
    assert [r["title"] for r in rows] == ["n0", "n1", "n2"]  # oldest first
    assert log.unread_count() == 3

    log.mark_read([rows[0]["id"]])
    assert log.unread_count() == 2
    assert log.recent(unread_only=True)[0]["title"] == "n1"

    # And read state survives a new reader over the same files.
    assert MessageLog(tmp_path / "notifications.jsonl").unread_count() == 2


def test_a_message_written_before_ids_existed_still_gets_a_stable_one(tmp_path):
    path = tmp_path / "notifications.jsonl"
    path.write_text(
        json.dumps({"title": "old", "body": "b", "at": 1700000000.0}) + "\n", encoding="utf-8"
    )
    from itsbob.gui.messages import MessageLog

    first = MessageLog(path).recent()[0]["id"]
    assert first and MessageLog(path).recent()[0]["id"] == first
    MessageLog(path).mark_read([first])
    assert MessageLog(path).unread_count() == 0


def test_following_yields_new_messages_and_idles_in_between(tmp_path):
    from itsbob.daemon.notify import FileSink, Notification
    from itsbob.gui.messages import MessageLog

    path = tmp_path / "notifications.jsonl"
    sink = FileSink(path=path)
    sink.send(Notification(title="first", body=""))
    log = MessageLog(path)

    stop = threading.Event()
    seen: list = []

    def reader():
        for item in log.follow(after=log.latest_id(), interval=0.02, stop=stop):
            seen.append(item)
            if len([s for s in seen if s is not None]) >= 2:
                stop.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    time.sleep(0.1)
    sink.send(Notification(title="second", body=""))
    time.sleep(0.1)
    sink.send(Notification(title="third", body=""))
    thread.join(timeout=3)
    stop.set()

    titles = [s["title"] for s in seen if s is not None]
    assert titles == ["second", "third"]
    assert None in seen  # idle ticks, which is what keeps the SSE alive


# -- search and vision -----------------------------------------------------


def test_search_prefers_a_command_line_client_when_one_is_installed(monkeypatch):
    import subprocess

    from itsbob.tools import websearch

    monkeypatch.setattr(websearch.shutil, "which", lambda b: "/usr/bin/ddgr" if b == "ddgr" else None)
    monkeypatch.setattr(
        websearch.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0,
            stdout=json.dumps([{"title": "T", "url": "https://x.test", "abstract": "A"}]),
            stderr="",
        ),
    )
    results, backend = websearch.search("anything")
    assert backend == "ddgr"
    assert results[0].url == "https://x.test"
    assert websearch.available_backend() == "ddgr"


def test_search_falls_back_to_html_when_no_client_is_installed(monkeypatch):
    from itsbob.tools import websearch

    monkeypatch.setattr(websearch.shutil, "which", lambda b: None)
    monkeypatch.setattr(
        websearch,
        "_from_html",
        lambda q, n: [websearch.SearchResult(title="H", url="https://y.test")],
    )
    results, backend = websearch.search("anything")
    assert backend == "duckduckgo-html" and results[0].title == "H"


def test_duckduckgo_markup_is_unwrapped_into_real_urls():
    from itsbob.tools.websearch import _from_html

    body = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.test%2Fpage">'
        "Real &amp; Page</a>"
        '<a class="result__snippet" href="#">the <b>snippet</b></a>'
    )

    class Fake:
        def read(self):
            return body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import itsbob.tools.websearch as ws

    original = ws.urllib.request.urlopen
    ws.urllib.request.urlopen = lambda *a, **k: Fake()
    try:
        results = _from_html("q", 5)
    finally:
        ws.urllib.request.urlopen = original
    assert results[0].url == "https://real.test/page"
    assert results[0].title == "Real & Page"
    assert results[0].snippet == "the snippet"


def test_an_image_is_downscaled_before_it_is_uploaded(tmp_path):
    pillow = pytest.importorskip("PIL.Image")
    from itsbob.tools.vision import MAX_EDGE, _prepare

    big = tmp_path / "big.png"
    pillow.new("RGB", (3000, 2000), "red").save(big)
    data, mime = _prepare(big)
    assert mime == "image/jpeg"
    with pillow.open(__import__("io").BytesIO(data)) as shrunk:
        assert max(shrunk.size) <= MAX_EDGE
    assert len(data) < big.stat().st_size


def test_image_tools_stay_inside_the_workspace(tmp_path):
    from itsbob.tools import build_toolbox

    box = build_toolbox(workspace=tmp_path / "ws", mode="trusted", env={})
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"not really an image")
    result = box.call("image_info", path=str(outside))
    assert not result.ok and "outside the workspace" in (result.error or "")


# -- screenshots -----------------------------------------------------------


def test_a_headless_machine_is_told_why_it_cannot_screenshot(tmp_path):
    from itsbob.scripts.screenshot import capture
    from itsbob.tools.base import ToolError

    with pytest.raises(ToolError, match="no display"):
        capture(tmp_path / "s.png", env={})


def test_an_active_window_shot_falls_back_to_the_full_screen(tmp_path, monkeypatch):
    """Most Wayland compositors cannot do window capture. A full screen beats an error."""
    from itsbob.scripts import screenshot

    monkeypatch.setattr(
        screenshot.shutil, "which", lambda b: "/usr/bin/grim" if b == "grim" else None
    )

    def fake_run(command):
        Path = type(tmp_path)
        Path(command[-1]).write_bytes(b"PNG-ish")
        return True, ""

    monkeypatch.setattr(screenshot, "_run", fake_run)
    shot = screenshot.capture(tmp_path / "s.png", window=True, env={"DISPLAY": ":0"})
    assert shot.window is False
    assert "not available on this desktop" in shot.note
    assert "describe_image" in shot.render()


def test_a_new_script_dropped_in_joins_the_pool(tmp_path):
    """Adding a script must not mean editing a list."""
    from itsbob.scripts import describe_scripts, discover, script_tools

    directory = tmp_path / "scripts"
    directory.mkdir()
    (directory / "coffee.py").write_text(
        "from itsbob.tools.base import Risk, Tool, ToolResult\n"
        'SUMMARY = "Make coffee."\n'
        "def tools():\n"
        "    return [Tool(name='make_coffee', description='d', "
        "run=lambda p, c: ToolResult(ok=True, output='brewing'), risk=Risk.READ)]\n",
        encoding="utf-8",
    )
    env = {"ITSBOB_SCRIPTS_DIR": str(directory)}
    assert "coffee" in [n for n, _, _ in discover(env)]
    assert "make_coffee" in [t.name for t in script_tools(env)]
    row = next(r for r in describe_scripts(env) if r["name"] == "coffee")
    assert row["summary"] == "Make coffee." and row["source"] == "user"


def test_a_broken_script_costs_only_its_own_capability(tmp_path):
    from itsbob.scripts import load_errors, script_tools

    directory = tmp_path / "scripts"
    directory.mkdir()
    (directory / "broken.py").write_text("import nonexistent_module_xyz\n", encoding="utf-8")
    env = {"ITSBOB_SCRIPTS_DIR": str(directory)}
    names = [t.name for t in script_tools(env)]
    assert "system_status" in names  # everything else still there
    assert "broken" in load_errors
