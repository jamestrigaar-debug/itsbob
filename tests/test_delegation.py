"""Delegating hard questions somewhere cheap, and reading web pages.

The premise is a cost trade: reasoning done free somewhere else, with only a
small local call to shape the result. That trade is only worth making if the
handoff never lets a login wall, a half-rendered page or a chat preamble reach
the agent looking like an answer — so most of this file is about the failure
paths, which is where the value is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from itsbob.integrations.delegate import Delegate, Envelope, unwrap, wrap
from itsbob.scripts.web_scraper import readable_text, scrape
from itsbob.tools.base import ToolError


# -- the envelope ----------------------------------------------------------


def test_the_envelope_asks_for_one_block_and_no_conversation():
    prompt = wrap("Why is the sky blue?", context="some background")
    assert "Why is the sky blue?" in prompt
    assert "some background" in prompt
    assert "```json" in prompt
    # A free chat interface will otherwise ask a clarifying question nobody can
    # answer, and wait.
    assert "cannot reply" in prompt or "one-shot" in prompt
    for field in ("answer", "key_points", "caveats", "confidence"):
        assert f'"{field}"' in prompt


def test_the_block_is_found_however_it_arrives():
    payload = {"answer": "Rayleigh scattering.", "confidence": "high"}
    fenced = f"Here is my thinking...\n```json\n{json.dumps(payload)}\n```"
    assert unwrap(fenced) == payload
    assert unwrap(f"```\n{json.dumps(payload)}\n```") == payload
    assert unwrap(f"Sure!\n{json.dumps(payload)}") == payload


def test_the_last_block_wins():
    """A reply that reasons aloud may show a draft before the real one."""
    reply = (
        '```json\n{"answer": "a first draft"}\n```\n'
        'On reflection:\n```json\n{"answer": "the real answer"}\n```'
    )
    assert unwrap(reply)["answer"] == "the real answer"


def test_prose_is_not_a_parse_failure():
    """It means the far end answered in prose, which is what shaping is for."""
    assert unwrap("Just a paragraph of prose.") is None
    assert unwrap("") is None
    assert unwrap("```json\nnot actually json\n```") is None


# -- the failsafe chain ----------------------------------------------------


def _shaper(result):
    return lambda system, text: result


def test_a_structured_reply_needs_no_local_call():
    """The whole point: free reasoning, and only pay when shaping is needed."""
    payload = {"answer": "Use WAL.", "key_points": ["fewer writer stalls"],
               "confidence": "high"}
    called = []
    delegate = Delegate(
        transport=lambda prompt: f"```json\n{json.dumps(payload)}\n```",
        formatter=lambda s, t: called.append(t) or {},
    )
    result = delegate.ask("Should I use WAL?")
    assert result.ok and result.structured_at_source
    assert called == []            # no local call was needed
    assert delegate.shaped_locally == 0
    assert "Use WAL." in result.render()
    assert "fewer writer stalls" in result.render()


def test_prose_is_shaped_locally_rather_than_discarded():
    delegate = Delegate(
        transport=lambda prompt: "WAL mode lets readers and a writer work at once. " * 3,
        formatter=_shaper({"answer": "WAL lets readers and a writer work at once.",
                           "confidence": "medium"}),
    )
    result = delegate.ask("Should I use WAL?")
    assert result.ok and not result.structured_at_source
    assert delegate.shaped_locally == 1
    assert "shaped locally" in result.render()


def test_a_broken_formatter_keeps_the_prose():
    """The answer is still an answer even if the shaping step falls over."""
    def explode(system, text):
        raise RuntimeError("the local model is down")

    delegate = Delegate(
        transport=lambda prompt: "A long and genuinely useful prose answer. " * 3,
        formatter=explode,
    )
    result = delegate.ask("anything")
    assert result.ok
    assert "genuinely useful prose answer" in result.answer


def test_a_dead_transport_is_reported_not_faked():
    """The one outcome that must never happen is a confident empty answer."""
    def refuse(prompt):
        raise RuntimeError("no browser on this machine")

    delegate = Delegate(transport=refuse, name="deepseek")
    result = delegate.ask("anything")
    assert not result.ok
    assert "no browser" in result.error
    assert "delegation failed" in result.render()
    assert delegate.failures == 1


@pytest.mark.parametrize(
    "reply, because",
    [
        ("", "nothing"),
        ("   ", "nothing"),
        ("ok", "too short"),
        ("Please log in to continue.", "login"),
        ("Rate limit exceeded, try later.", "rate limit"),
    ],
)
def test_a_non_answer_is_caught_before_it_becomes_an_answer(reply, because):
    """A login wall is a page, not a reply. It must not reach the agent."""
    result = Delegate(transport=lambda p: reply).ask("a hard question")
    assert not result.ok, f"{because}: {reply!r} was accepted as an answer"
    assert result.error


def test_a_long_reply_mentioning_a_rate_limit_is_still_an_answer():
    """The phrase appears in real answers about APIs; only a short reply is a wall."""
    reply = "Rate limit handling matters here. " * 40
    assert Delegate(transport=lambda p: reply).ask("how do I handle 429s?").ok


def test_the_envelope_is_customisable_without_touching_the_parser():
    envelope = Envelope(fields=(("answer", "the answer"), ("sources", "urls")))
    prompt = wrap("q", envelope=envelope)
    assert '"sources"' in prompt and '"caveats"' not in prompt


# -- reading a page --------------------------------------------------------


def test_markup_is_stripped_to_the_part_worth_paying_for():
    """An observation gets clipped; what survives should be the article."""
    markup = """<html><head><title>An Article</title>
      <style>body{color:red}</style></head><body>
      <nav>Home · About · Contact · Subscribe</nav>
      <script>var tracking = 1; analytics.push('x');</script>
      <h1>The Headline</h1>
      <p>First paragraph with &amp; an entity.</p><p>Second paragraph.</p>
      <footer>© 2026 · Privacy · Terms</footer></body></html>"""
    text = readable_text(markup)
    assert "The Headline" in text and "First paragraph with & an entity." in text
    assert "tracking" not in text and "color:red" not in text
    assert "Privacy" not in text and "Subscribe" not in text


def test_a_page_reports_how_much_it_kept(monkeypatch):
    from itsbob.scripts import web_scraper

    body = "<html><title>Long</title><body>" + "<p>paragraph text</p>" * 400 + "</body></html>"
    monkeypatch.setattr(web_scraper, "_plain_fetch", lambda url: (body, "Long"))
    page = scrape("https://example.test/a", max_chars=500)
    assert page.method == "http" and page.truncated
    assert len(page.text) == 500
    assert "truncated at" in page.render()
    assert str(page.chars) in page.render() or f"{page.chars:,}" in page.render()


def test_a_javascript_shell_escalates_rather_than_returning_nothing(monkeypatch):
    """A plain fetch of an app returns almost no text — detectable, not guessed."""
    from itsbob.scripts import web_scraper

    monkeypatch.setattr(
        web_scraper, "_plain_fetch",
        lambda url: ('<html><body><div id="root"></div></body></html>', "App"))
    monkeypatch.setattr(web_scraper, "available", lambda env: {
        "playwright": {"ready": True, "why": "installed"},
        "xdotool": {"ready": False, "why": "no display"}})

    class FakeSession:
        def fetch_text(self, url, scrolls=0):
            return "The content that only exists after JavaScript runs. " * 20

    monkeypatch.setattr(web_scraper, "PlaywrightSession", lambda *a, **k: FakeSession())
    page = scrape("https://app.test/")
    assert page.method == "headless browser"
    assert "only exists after JavaScript" in page.text


def test_every_mechanism_failing_says_what_was_tried(monkeypatch):
    from itsbob.scripts import web_scraper

    def dead(url):
        raise ToolError("HTTP 403 from example.test")

    monkeypatch.setattr(web_scraper, "_plain_fetch", dead)
    monkeypatch.setattr(web_scraper, "available", lambda env: {
        "playwright": {"ready": False, "why": "not installed"},
        "xdotool": {"ready": False, "why": "no display attached"}})
    with pytest.raises(ToolError) as caught:
        scrape("https://example.test/")
    message = str(caught.value)
    assert "403" in message and "not installed" in message and "no display" in message


def test_only_http_urls_are_read():
    for url in ("file:///etc/passwd", "ftp://x.test/a", "javascript:alert(1)"):
        with pytest.raises(ToolError, match="only http"):
            scrape(url)


# -- the tools -------------------------------------------------------------


def test_the_deepseek_tool_is_off_until_switched_on(tmp_path):
    from itsbob.tools import build_toolbox

    box = build_toolbox(workspace=tmp_path / "ws", mode="trusted", env={})
    result = box.call("ask_deepseek", question="what is the meaning of it all")
    assert not result.ok
    assert "ITSBOB_DEEPSEEK=1" in result.error
    # Off means off: no browser was launched to discover that.
    assert "opt-in" in result.error


def test_the_deepseek_tool_uses_the_injected_shaper(tmp_path, monkeypatch):
    from itsbob.scripts import deepseek
    from itsbob.tools import build_toolbox

    monkeypatch.setattr(deepseek, "ask_deepseek",
                        lambda prompt, env=None: "Prose, at some length, about the thing. " * 4)
    box = build_toolbox(
        workspace=tmp_path / "ws", mode="trusted",
        env={"ITSBOB_DEEPSEEK": "1"},
        extras={"shape_json": lambda s, t: {"answer": "Shaped by the cheap model.",
                                            "confidence": "medium"}},
    )
    result = box.call("ask_deepseek", question="a genuinely hard question")
    assert result.ok
    assert "Shaped by the cheap model." in result.output
    assert result.data["structured_at_source"] is False


def test_a_failed_delegation_points_back_at_the_tier_ladder(tmp_path, monkeypatch):
    """Falling back to the paid path is the correct outcome, and it says so."""
    from itsbob.scripts import deepseek
    from itsbob.tools import build_toolbox

    def refuse(prompt, env=None):
        raise RuntimeError("no browser available")

    monkeypatch.setattr(deepseek, "ask_deepseek", refuse)
    box = build_toolbox(workspace=tmp_path / "ws", mode="trusted",
                        env={"ITSBOB_DEEPSEEK": "1"})
    result = box.call("ask_deepseek", question="a hard question worth asking")
    assert not result.ok
    assert "no browser available" in result.error
    assert "tier ladder" in result.error


def test_both_scripts_are_discovered_with_their_tools():
    from itsbob.scripts import describe_scripts

    rows = {r["name"]: r for r in describe_scripts({})}
    assert "read_page" in [t["name"] for t in rows["web_scraper"]["tools"]]
    assert "ask_deepseek" in [t["name"] for t in rows["deepseek"]["tools"]]
    assert rows["web_scraper"]["summary"].startswith("Read a web page")


def test_the_browser_layer_reports_what_this_machine_can_do():
    from itsbob.integrations.browser import available, profile_dir

    ready = available({})
    assert set(ready) == {"playwright", "xdotool", "preferred", "profile", "chromium"}
    for name in ("playwright", "xdotool"):
        assert ready[name]["why"], f"{name} gives no reason either way"
    # Its own profile, never the one a person browses with.
    assert "browser-profile" in str(profile_dir({"ITSBOB_HOME": "/tmp/h"}))


# -- having a browser, as opposed to having the package that drives one ------


def _fake_download(root: Path, folder: str, relative: str) -> Path:
    """A stand-in for a browser Playwright downloaded."""
    binary = root / folder / relative
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def test_a_configured_chromium_wins_and_a_missing_one_says_so(tmp_path):
    from itsbob.integrations.browser import available, chromium

    real = tmp_path / "my-chromium"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    assert chromium({"ITSBOB_CHROMIUM": str(real)}) == {
        "path": str(real),
        "source": "configured",
    }

    # Pointed somewhere wrong is its own answer: reporting "no browser" would
    # send somebody installing one they already have.
    missing = chromium({"ITSBOB_CHROMIUM": str(tmp_path / "nope")})
    assert missing == {"path": "", "source": "configured-missing"}
    why = available({"ITSBOB_CHROMIUM": str(tmp_path / "nope")})["playwright"]["why"]
    assert "ITSBOB_CHROMIUM" in why and "not there" in why


def test_a_full_chromium_beats_a_headless_shell_even_from_an_older_build(tmp_path):
    from itsbob.integrations.browser import chromium

    # The shell sorts later by name and by version, and still must not win: it
    # cannot open a visible window, which is half of what the browser is for.
    _fake_download(tmp_path, "chromium_headless_shell-2000", "chrome-linux/headless_shell")
    full = _fake_download(tmp_path, "chromium-1194", "chrome-linux/chrome")

    found = chromium({"PLAYWRIGHT_BROWSERS_PATH": str(tmp_path)})
    assert found == {"path": str(full), "source": "playwright"}


def test_the_newest_build_wins_among_equals(tmp_path):
    from itsbob.integrations.browser import chromium

    _fake_download(tmp_path, "chromium-1100", "chrome-linux/chrome")
    newest = _fake_download(tmp_path, "chromium-1194", "chrome-linux/chrome")

    assert chromium({"PLAYWRIGHT_BROWSERS_PATH": str(tmp_path)})["path"] == str(newest)


def test_a_system_chromium_is_used_when_playwright_never_downloaded_one(tmp_path, monkeypatch):
    from itsbob.integrations import browser

    installed = tmp_path / "usr-bin-chromium"
    installed.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        browser.shutil,
        "which",
        lambda name: str(installed) if name == "chromium" else None,
    )

    found = browser.chromium({"PLAYWRIGHT_BROWSERS_PATH": str(tmp_path / "empty")})
    assert found == {"path": str(installed), "source": "system"}


def test_the_package_alone_is_not_a_browser(tmp_path, monkeypatch):
    """Installing playwright downloads no browser, and driving nothing fails."""
    from itsbob.integrations import browser

    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    ready = browser.available({"PLAYWRIGHT_BROWSERS_PATH": str(tmp_path / "empty")})

    assert ready["playwright"]["ready"] is False
    assert ready["chromium"] == {"path": "", "source": ""}
    # Whatever the reason, it has to name the next command to run.
    assert "install" in ready["playwright"]["why"]


def test_playwright_is_left_to_find_its_own_download(tmp_path, monkeypatch):
    """Its own build needs no executable_path; anything else does."""
    from itsbob.integrations import browser

    launched: dict[str, object] = {}

    class _Chromium:
        def launch_persistent_context(self, **options):
            launched.update(options)
            return object()

    class _Playwright:
        chromium = _Chromium()

    downloaded = _fake_download(tmp_path, "chromium-1194", "chrome-linux/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
    monkeypatch.delenv("ITSBOB_CHROMIUM", raising=False)
    session = browser.PlaywrightSession(profile=tmp_path / "profile")
    session._launch(_Playwright())
    assert "executable_path" not in launched
    assert downloaded.exists()

    # A system browser is invisible to Playwright unless it is handed over.
    launched.clear()
    system = tmp_path / "system-chromium"
    system.write_text("#!/bin/sh\n")
    monkeypatch.setenv("ITSBOB_CHROMIUM", str(system))
    browser.PlaywrightSession(profile=tmp_path / "profile")._launch(_Playwright())
    assert launched["executable_path"] == str(system)

