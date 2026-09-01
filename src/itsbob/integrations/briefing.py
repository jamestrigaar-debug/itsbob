"""Weather, news, and the one condensed report that combines them.

Three tools, one shape: fetch the raw payload, throw away everything that is not
worth a token, and hand back something a model can read in one glance.

The stripping is the point. NewsAPI returns roughly 1.5 KB per article — image
URLs, truncated `content` blobs, source ids, author bylines — and twenty of
those is thirty kilobytes of prompt for what amounts to twenty headlines. The
same is true of OpenWeather's forecast, which is forty three-hourly entries with
sixteen fields each when what anyone wants is "cold, wet after four". So each
tool here returns a handful of lines, and the full payload stays behind
``call_api`` for the rare case something else is needed.

The location defaults to the one this assistant is set up for and is overridable
by environment variable — a postcode is not something to hard-code past the
point of a sensible default.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["briefing_tools", "fetch_weather", "fetch_news", "Place"]

TIMEOUT = 20.0

#: Hull, UK (HU6). Overridable with ITSBOB_WEATHER_LAT / _LON / _PLACE.
DEFAULT_PLACE = ("Hull, UK", 53.7767, -0.3274)


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Place":
        env = os.environ if env is None else env
        name, lat, lon = DEFAULT_PLACE
        return cls(
            name=env.get("ITSBOB_WEATHER_PLACE", "").strip() or name,
            lat=_float(env.get("ITSBOB_WEATHER_LAT"), lat),
            lon=_float(env.get("ITSBOB_WEATHER_LON"), lon),
        )


def _float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _get_json(url: str, *, headers: Mapping[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:200]
        raise ToolError(f"HTTP {exc.code} from {urllib.parse.urlparse(url).netloc}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"could not reach {urllib.parse.urlparse(url).netloc}: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise ToolError(f"{type(exc).__name__} fetching {urllib.parse.urlparse(url).netloc}") from exc


# -- weather ---------------------------------------------------------------


@dataclass
class Weather:
    place: str
    summary: str = ""
    temp_c: float | None = None
    feels_c: float | None = None
    wind_mph: float | None = None
    #: One line per part of the day, already condensed.
    outlook: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"{self.place}: {self.summary}"]
        if self.temp_c is not None:
            feels = f" (feels {self.feels_c:.0f})" if self.feels_c is not None else ""
            lines[0] += f", {self.temp_c:.0f}°C{feels}"
        if self.wind_mph is not None:
            lines[0] += f", wind {self.wind_mph:.0f} mph"
        lines.extend(self.outlook)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "place": self.place,
            "summary": self.summary,
            "temp_c": self.temp_c,
            "feels_c": self.feels_c,
            "wind_mph": self.wind_mph,
            "outlook": list(self.outlook),
        }


def fetch_weather(
    *, key: str, place: Place, forecast: bool = True, fetch: Any = _get_json
) -> Weather:
    """Current conditions, plus today's shape in four lines."""
    base = "https://api.openweathermap.org/data/2.5"
    query = urllib.parse.urlencode(
        {"lat": place.lat, "lon": place.lon, "units": "metric", "appid": key}
    )
    now = fetch(f"{base}/weather?{query}")
    weather = Weather(
        place=place.name,
        summary=str((now.get("weather") or [{}])[0].get("description", "")).capitalize(),
        temp_c=_float((now.get("main") or {}).get("temp"), None) if now.get("main") else None,
        feels_c=_float((now.get("main") or {}).get("feels_like"), None) if now.get("main") else None,
        wind_mph=(
            _float((now.get("wind") or {}).get("speed"), 0.0) * 2.23694
            if now.get("wind")
            else None
        ),
    )
    if not forecast:
        return weather

    # Only today's entries, and only every second one: eight three-hourly rows
    # say the same thing as four and cost twice as much to read.
    data = fetch(f"{base}/forecast?{query}")
    today = time.strftime("%Y-%m-%d")
    rows = [
        row
        for row in (data.get("list") or [])
        if str(row.get("dt_txt", "")).startswith(today)
    ][:8:2]
    for row in rows:
        when = str(row.get("dt_txt", ""))[11:16]
        desc = str((row.get("weather") or [{}])[0].get("description", ""))
        temp = (row.get("main") or {}).get("temp")
        rain = (row.get("rain") or {}).get("3h")
        line = f"  {when} — {desc}"
        if temp is not None:
            line += f", {float(temp):.0f}°C"
        if rain:
            line += f", {float(rain):.1f}mm rain"
        weather.outlook.append(line)
    return weather


# -- news ------------------------------------------------------------------

#: The default beat. The request was for "geo-political, large events" rather
#: than a general feed, and a query is the only lever either API gives for that.
GEOPOLITICS = (
    "geopolitics OR election OR sanctions OR ceasefire OR treaty OR summit OR "
    "invasion OR parliament OR referendum OR strike OR earthquake"
)


@dataclass
class Headline:
    title: str
    source: str = ""
    url: str = ""
    published: str = ""
    summary: str = ""

    def render(self) -> str:
        stamp = f" ({self.published[:10]})" if self.published else ""
        line = f"- {self.title} — {self.source}{stamp}"
        if self.summary:
            line += f"\n    {self.summary}"
        return line

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published": self.published,
            "summary": self.summary,
        }


def _dedupe(headlines: Sequence[Headline], limit: int) -> list[Headline]:
    """One row per story. Two wire services carrying the same event is one story."""
    seen: set[str] = set()
    out: list[Headline] = []
    for item in headlines:
        # First six significant words: enough to catch a re-headline, short
        # enough that a genuinely different story is not swallowed.
        key = " ".join(
            w for w in item.title.lower().split() if len(w) > 3
        )[:60]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def fetch_news(
    *,
    newsapi_key: str = "",
    gnews_key: str = "",
    query: str = GEOPOLITICS,
    limit: int = 10,
    fetch: Any = _get_json,
) -> tuple[list[Headline], list[str]]:
    """Headlines from whichever sources are configured, merged and deduplicated.

    Returns ``(headlines, problems)``. A source that fails is a problem, not an
    exception: one dead API should not cost the other one's results, and the
    daily report is more useful incomplete than absent.
    """
    headlines: list[Headline] = []
    problems: list[str] = []

    if newsapi_key:
        try:
            query_string = urllib.parse.urlencode(
                {
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": min(50, limit * 3),
                }
            )
            data = fetch(
                f"https://newsapi.org/v2/everything?{query_string}",
                headers={"X-Api-Key": newsapi_key},
            )
            for article in data.get("articles") or []:
                headlines.append(
                    Headline(
                        title=str(article.get("title") or "").strip(),
                        source=str((article.get("source") or {}).get("name") or ""),
                        url=str(article.get("url") or ""),
                        published=str(article.get("publishedAt") or ""),
                        summary=_clip(str(article.get("description") or ""), 220),
                    )
                )
        except ToolError as exc:
            problems.append(f"newsapi: {exc}")

    if gnews_key:
        try:
            query_string = urllib.parse.urlencode(
                {"q": query, "lang": "en", "max": min(25, limit * 2), "apikey": gnews_key}
            )
            data = fetch(f"https://gnews.io/api/v4/search?{query_string}")
            for article in data.get("articles") or []:
                headlines.append(
                    Headline(
                        title=str(article.get("title") or "").strip(),
                        source=str((article.get("source") or {}).get("name") or ""),
                        url=str(article.get("url") or ""),
                        published=str(article.get("publishedAt") or ""),
                        summary=_clip(str(article.get("description") or ""), 220),
                    )
                )
        except ToolError as exc:
            problems.append(f"gnews: {exc}")

    if not headlines and not problems:
        problems.append("no news API key is set (NEWSAPI_KEY or GNEWS_API_KEY)")
    headlines.sort(key=lambda h: h.published, reverse=True)
    return _dedupe([h for h in headlines if h.title], limit), problems


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit].rstrip()}…"


# -- tools -----------------------------------------------------------------


def _key(ctx: ToolContext, name: str) -> str:
    env = ctx.env if ctx.env is not None else os.environ
    return str(env.get(name, "")).strip()


def _allow_hosts(ctx: ToolContext, *urls: str) -> None:
    """Check the fixed endpoints hidden behind the convenience tools."""
    policy = getattr(ctx, "policy", None)
    if policy is None:
        return
    for url in urls:
        host_reason = policy.check_url(url)
        if host_reason:
            raise ToolError(host_reason)


def _weather_tool(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = _key(ctx, "OPENWEATHER_API_KEY")
    if not key:
        raise ToolError("OPENWEATHER_API_KEY is not set — add it to .env and restart")
    _allow_hosts(ctx, "https://api.openweathermap.org/")
    place = Place.from_env(ctx.env)
    weather = fetch_weather(
        key=key, place=place, forecast=bool(params.get("forecast", True))
    )
    return ToolResult(ok=True, output=weather.render(), data=weather.as_dict())


def _news_tool(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    news_key = _key(ctx, "NEWSAPI_KEY")
    gnews_key = _key(ctx, "GNEWS_API_KEY")
    _allow_hosts(
        ctx,
        *(["https://newsapi.org/"] if news_key else []),
        *(["https://gnews.io/"] if gnews_key else []),
    )
    limit = max(1, min(25, int(params.get("limit", 10))))
    headlines, problems = fetch_news(
        newsapi_key=news_key,
        gnews_key=gnews_key,
        query=str(params.get("topic") or "").strip() or GEOPOLITICS,
        limit=limit,
    )
    if not headlines:
        raise ToolError("; ".join(problems) or "no headlines came back")
    body = "\n".join(h.render() for h in headlines)
    if problems:
        body += "\n\n(partial: " + "; ".join(problems) + ")"
    return ToolResult(
        ok=True,
        output=body,
        data={"headlines": [h.as_dict() for h in headlines], "problems": problems},
    )


def _briefing_tool(summarize: Any):
    """The daily report: today's weather, then the day's news, condensed."""

    def run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        parts: list[str] = [f"# Briefing — {time.strftime('%A %d %B %Y')}"]
        problems: list[str] = []
        payload: dict[str, Any] = {}

        key = _key(ctx, "OPENWEATHER_API_KEY")
        if key:
            try:
                _allow_hosts(ctx, "https://api.openweathermap.org/")
                weather = fetch_weather(key=key, place=Place.from_env(ctx.env))
                payload["weather"] = weather.as_dict()
                parts += ["", "## Weather", weather.render()]
            except ToolError as exc:
                problems.append(f"weather: {exc}")
        else:
            problems.append("weather: OPENWEATHER_API_KEY is not set")

        news_key = _key(ctx, "NEWSAPI_KEY")
        gnews_key = _key(ctx, "GNEWS_API_KEY")
        try:
            _allow_hosts(
                ctx,
                *(["https://newsapi.org/"] if news_key else []),
                *(["https://gnews.io/"] if gnews_key else []),
            )
        except ToolError as exc:
            news_problems = [str(exc)]
            headlines = []
        else:
            headlines, news_problems = fetch_news(
                newsapi_key=news_key,
                gnews_key=gnews_key,
            query=str(params.get("topic") or "").strip() or GEOPOLITICS,
            limit=max(1, min(25, int(params.get("limit", 12)))),
            )
        problems.extend(news_problems)
        if headlines:
            payload["headlines"] = [h.as_dict() for h in headlines]
            digest = ""
            if summarize is not None:
                digest = summarize(headlines)
            parts += ["", "## What is happening"]
            parts.append(digest or "\n".join(h.render() for h in headlines))
            if digest:
                # The digest is the report; the raw headlines stay as the
                # citation trail, titles only.
                parts += ["", "### Sources", *(f"- {h.title} ({h.source})" for h in headlines)]

        if problems:
            parts += ["", "## Not available", *(f"- {p}" for p in problems)]
        if len(parts) == 1:
            raise ToolError("; ".join(problems) or "nothing to report")

        payload["problems"] = problems
        return ToolResult(ok=True, output="\n".join(parts), data=payload)

    return run


def briefing_tools(summarize: Any = None) -> list[Tool]:
    """Weather, news and the combined daily report.

    ``summarize`` takes the headline list and returns condensed prose. It is
    injected rather than imported so this module never reaches back into the
    model ladder — and so the report still works, as a plain list, when there is
    no model to condense with.
    """
    return [
        Tool(
            name="weather",
            description=(
                "Current conditions and today's outlook for the configured location "
                "(Hull, UK by default). Already condensed — do not call the raw API "
                "for this."
            ),
            run=_weather_tool,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "forecast": {
                        "type": "boolean",
                        "description": "Include today's hour-by-hour outlook. Default true.",
                    }
                },
            },
        ),
        Tool(
            name="news",
            description=(
                "Recent headlines, merged from every configured news source and "
                "deduplicated. Defaults to geopolitics and large events; pass "
                "`topic` for anything else."
            ),
            run=_news_tool,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Search terms. Optional."},
                    "limit": {"type": "integer", "description": "1-25. Default 10."},
                },
            },
        ),
        Tool(
            name="daily_briefing",
            description=(
                "The day in one page: today's weather for the configured location, "
                "then the day's significant news condensed into prose with sources. "
                "Built for a scheduled morning task."
            ),
            run=_briefing_tool(summarize),
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Override the news beat."},
                    "limit": {"type": "integer", "description": "Headlines to consider. Default 12."},
                },
            },
        ),
    ]
