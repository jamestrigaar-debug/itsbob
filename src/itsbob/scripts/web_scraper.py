"""Reading a web page, by the cheapest method that works.

Three mechanisms, tried in order of cost, because most pages do not need the
expensive one:

1. **A plain HTTP fetch**, with the markup stripped locally. No browser, no
   dependency, a fraction of a second. Works for articles, docs, most of the
   readable web.
2. **A headless browser**, when step 1 comes back with a shell — a page that
   builds itself in JavaScript returns almost no text to a plain fetch, and that
   is detectable rather than something to guess at.
3. **The browser already on screen**, for anything behind a login. It is
   intrusive and it is last.

The output is shaped rather than dumped, for the same reason API payloads are:
a page is mostly navigation, cookie banners and footer links, and an
observation is clipped to a few thousand characters. Handing over raw HTML
means the clip lands in the middle of a `<script>` tag and the actual article
never reaches the model. Stripping first means what survives the clip is the
part worth reading.

`itsbob` already has `web_search` for finding pages and `http_request` for APIs.
This is for reading one page properly, which is the thing neither does.
"""

from __future__ import annotations

import html
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..integrations.browser import BrowserError, PlaywrightSession, XdotoolSession, available
from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["scrape", "readable_text", "Page", "tools"]

SUMMARY = "Read a web page as text, by the cheapest method that works."

TIMEOUT = 25.0
MAX_FETCH_BYTES = 4_000_000
#: Default cap on what reaches the model. Generous enough for a long article,
#: bounded because an observation is clipped anyway and a silent clip is worse
#: than an explicit one.
DEFAULT_MAX_CHARS = 12_000
#: Below this, a plain fetch has probably returned a JavaScript shell.
JS_SHELL_THRESHOLD = 500

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|svg|template|iframe|nav|footer|header|form)\b.*?</\1>",
    re.S | re.I,
)
_COMMENTS = re.compile(r"<!--.*?-->", re.S)
_BREAKS = re.compile(r"</(p|div|li|tr|h[1-6]|section|article|br)\s*>", re.I)
_TAGS = re.compile(r"<[^>]+>")
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACE_RUN = re.compile(r"[ \t]{2,}")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


def readable_text(markup: str) -> str:
    """Markup into something worth spending tokens on.

    A deliberately small transformation, not a readability engine: drop the
    blocks that are never prose, turn block ends into newlines so paragraphs
    survive, strip what is left, and collapse the whitespace. Pulling in a real
    extractor would be better on hard pages and is a dependency for a fallback
    path that mostly reads articles.
    """
    text = _COMMENTS.sub(" ", markup or "")
    text = _DROP_BLOCKS.sub(" ", text)
    text = _BREAKS.sub("\n", text)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = _SPACE_RUN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_RUN.sub("\n\n", text).strip()


@dataclass
class Page:
    """One page, read."""

    url: str
    text: str = ""
    title: str = ""
    method: str = ""
    chars: int = 0
    truncated: bool = False
    #: Every mechanism that was tried and did not work, in order.
    attempts: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"{self.title or self.url} — {self.chars:,} characters via {self.method}"
        body = self.text
        if self.truncated:
            body += (
                f"\n\n… [truncated at {len(self.text):,} characters of {self.chars:,}. "
                "Ask for a specific section if you need more.]"
            )
        return f"{head}\n\n{body}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "method": self.method,
            "chars": self.chars,
            "truncated": self.truncated,
            "attempts": self.attempts,
        }


def _plain_fetch(url: str) -> tuple[str, str]:
    """Raw markup and the page title, over plain HTTP."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read(MAX_FETCH_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        raise ToolError(f"HTTP {exc.code} from {urllib.parse.urlparse(url).netloc}") from exc
    except Exception as exc:  # noqa: BLE001 - one outcome, several causes
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    markup = raw.decode(charset, "replace")
    found = _TITLE.search(markup)
    return markup, html.unescape(found.group(1)).strip() if found else ""


def scrape(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    scrolls: int = 0,
    force: str = "",
    env: Mapping[str, str] | None = None,
) -> Page:
    """Read ``url``, escalating only as far as it has to."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ToolError(f"only http/https URLs can be read, got {url!r}")

    page = Page(url=url)
    ready = available(env)

    if force in ("", "http"):
        try:
            markup, title = _plain_fetch(url)
            text = readable_text(markup)
            if len(text) >= JS_SHELL_THRESHOLD or force == "http":
                return _finish(page, text, title, "http", max_chars)
            page.attempts.append(
                f"http: only {len(text)} characters of text — the page builds "
                "itself in JavaScript"
            )
        except ToolError as exc:
            page.attempts.append(f"http: {exc}")

    if force in ("", "browser") and ready["playwright"]["ready"]:
        try:
            text = PlaywrightSession().fetch_text(url, scrolls=scrolls)
            if text.strip():
                return _finish(page, text, page.title, "headless browser", max_chars)
            page.attempts.append("headless browser: the page rendered no text")
        except Exception as exc:  # noqa: BLE001
            page.attempts.append(f"headless browser: {type(exc).__name__}: {exc}"[:200])
    elif force in ("", "browser"):
        page.attempts.append(f"headless browser: {ready['playwright']['why']}")

    if force in ("", "desktop") and ready["xdotool"]["ready"]:
        try:
            text = XdotoolSession().fetch_text(url, scrolls=scrolls)
            if text.strip():
                return _finish(page, text, page.title, "desktop browser", max_chars)
            page.attempts.append("desktop browser: nothing came back on the clipboard")
        except BrowserError as exc:
            page.attempts.append(f"desktop browser: {exc}"[:200])
    elif force in ("", "desktop"):
        page.attempts.append(f"desktop browser: {ready['xdotool']['why']}")

    raise ToolError("could not read that page — " + "; ".join(page.attempts))


def _finish(page: Page, text: str, title: str, method: str, max_chars: int) -> Page:
    page.chars = len(text)
    page.truncated = len(text) > max_chars
    page.text = text[:max_chars]
    page.title = title or page.title
    page.method = method
    return page


def _scrape(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    page = scrape(
        str(params.get("url", "")),
        max_chars=max(500, min(60_000, int(params.get("max_chars", DEFAULT_MAX_CHARS)))),
        scrolls=max(0, min(30, int(params.get("scrolls", 0)))),
        force=str(params.get("method") or ""),
        env=ctx.env if ctx.env is not None else os.environ,
    )
    return ToolResult(ok=True, output=page.render(), data=page.as_dict())


def tools() -> list[Tool]:
    return [
        Tool(
            name="read_page",
            description=(
                "Read a web page as clean text. Tries a plain fetch first, then a "
                "headless browser if the page needs JavaScript, then the browser on "
                "screen for anything behind a login. Use after `web_search` when a "
                "snippet is not enough."
            ),
            run=_scrape,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {
                        "type": "integer",
                        "description": "How much to return. Default 12000; the rest is named, not silently dropped.",
                    },
                    "scrolls": {
                        "type": "integer",
                        "description": "Scroll this many times before reading — for feeds that load as you go.",
                    },
                    "method": {
                        "type": "string",
                        "description": "Force one of http | browser | desktop. Normally leave unset.",
                    },
                },
                "required": ["url"],
            },
        )
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Read a web page as text.")
    parser.add_argument("url")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--scrolls", type=int, default=0)
    parser.add_argument("--method", default="", choices=["", "http", "browser", "desktop"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        page = scrape(args.url, max_chars=args.max_chars, scrolls=args.scrolls,
                      force=args.method)
    except ToolError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(page.as_dict(), indent=2) if args.json else page.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
