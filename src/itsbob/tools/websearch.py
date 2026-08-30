"""Web search, without an API key or an account.

Three routes, tried in order, all of them free:

``ddgr --json`` / ``googler --json``
    Command-line search clients. If either is installed, it is the best option:
    structured JSON, no key, no scraping on our part.

DuckDuckGo's HTML endpoint
    A last resort when neither binary is present, parsed with a small regex
    rather than a HTML library so nothing new has to be installed. It is the
    most fragile route and says so in its output, but "search is unavailable
    until you install something" is a worse answer than a slightly ragged one.

This is a separate tool rather than an instruction to run ``ddgr`` through
``run_shell`` on purpose. Search is a read-only network fetch; shell access is
the broadest capability in the system. Routing an everyday action through the
most dangerous door means either approving that door permanently — which is
what ``ITSBOB_AUTO_ALLOW=run_shell`` amounts to — or answering a confirmation
prompt every time somebody wants to look something up. A dedicated tool is
gated as :attr:`~itsbob.tools.base.Risk.NETWORK`, and ``run_shell`` can stay
locked.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["web_search_tools", "search", "SearchResult", "available_backend"]

TIMEOUT = 25.0
USER_AGENT = "Mozilla/5.0 (compatible; itsbob/1.0)"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def render(self) -> str:
        line = f"- {self.title}\n  {self.url}"
        if self.snippet:
            line += f"\n  {self.snippet}"
        return line

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def available_backend() -> str:
    """Which route search will take, for ``itsbob doctor`` and the status panel."""
    for binary in ("ddgr", "googler"):
        if shutil.which(binary):
            return binary
    return "duckduckgo-html"


def _from_cli(binary: str, query: str, limit: int) -> list[SearchResult]:
    # -n/--num is honoured by both; --noprompt keeps ddgr non-interactive when
    # it is run without a tty, which is exactly how it is run here.
    command = [binary, "--json", "--noprompt", "-n", str(limit), query]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed binary, argument list, no shell
            command, capture_output=True, text=True, timeout=TIMEOUT, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"{binary}: {exc}") from exc
    if completed.returncode != 0 and not completed.stdout.strip():
        raise ToolError(f"{binary} failed: {(completed.stderr or '').strip()[:200]}")
    try:
        rows = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ToolError(f"{binary} did not return JSON: {completed.stdout[:120]!r}") from exc
    return [
        SearchResult(
            title=str(row.get("title") or "").strip(),
            url=str(row.get("url") or "").strip(),
            snippet=" ".join(str(row.get("abstract") or "").split())[:300],
        )
        for row in rows
        if isinstance(row, dict) and row.get("url")
    ][:limit]


#: DuckDuckGo's no-JS result markup. Matched rather than parsed because pulling
#: in a HTML library for a fallback path is a poor trade.
_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?:.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>)?',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _from_html(query: str, limit: int) -> list[SearchResult]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - every failure here is the same answer
        raise ToolError(f"duckduckgo: {type(exc).__name__}: {exc}") from exc

    results: list[SearchResult] = []
    for match in _RESULT_RE.finditer(body):
        href = html.unescape(match.group("url"))
        # DDG wraps outbound links; unwrap so the model gets the real URL.
        if "uddg=" in href:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = (parsed.get("uddg") or [href])[0]
        results.append(
            SearchResult(
                title=_text(match.group("title")),
                url=href,
                snippet=_text(match.group("snippet") or "")[:300],
            )
        )
        if len(results) >= limit:
            break
    return results


def _text(markup: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub("", markup)).split())


def search(query: str, *, limit: int = 6) -> tuple[list[SearchResult], str]:
    """Search the web. Returns ``(results, which backend answered)``."""
    query = query.strip()
    if not query:
        raise ToolError("query is empty")
    limit = max(1, min(20, limit))

    problems: list[str] = []
    for binary in ("ddgr", "googler"):
        if not shutil.which(binary):
            continue
        try:
            results = _from_cli(binary, query, limit)
        except ToolError as exc:
            problems.append(str(exc))
            continue
        if results:
            return results, binary
    try:
        results = _from_html(query, limit)
    except ToolError as exc:
        problems.append(str(exc))
        results = []
    if results:
        return results, "duckduckgo-html"
    raise ToolError("; ".join(problems) or "no results")


def _run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    results, backend = search(
        str(params.get("query", "")), limit=int(params.get("limit", 6))
    )
    return ToolResult(
        ok=True,
        output="\n".join(r.render() for r in results) + f"\n\n(via {backend})",
        data={"results": [r.as_dict() for r in results], "backend": backend},
    )


def web_search_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web and get back titles, links and snippets. Free, no "
                "key needed. Use it for anything current, then read the page you "
                "want with `http_request` if the snippet is not enough."
            ),
            run=_run,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "1-20. Default 6."},
                },
                "required": ["query"],
            },
        )
    ]
