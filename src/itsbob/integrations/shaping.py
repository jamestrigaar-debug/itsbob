"""Turning API payloads into something a model can actually read in full.

This exists because of a measured failure, not a stylistic preference. Asked for
a matchday's results, itsbob reported two matches and said the rest were lost to
"API response truncation". It was telling the truth: football-data returns
~11,000 characters for ten matches, the loop clips a single observation to about
3,000, and **eight of the ten never reached the model at all**. What it did see
was mostly crest URLs, referee nationalities and an advert for the odds package.

So the payload is shaped before the model sees it: one line per item, every item
present, and only the fields anyone asked for. The same ten matches come out at
around 600 characters — complete *and* five times cheaper than the truncated
version that lost most of them.

Three rules hold everything here together.

**Never drop an item to fit.** Fields are dropped; rows are not. If a list is
genuinely too long, the shaper says exactly how many it is showing and how many
there are, so the model can page rather than guess.

**Say the count first.** "10 matches" at the top is what lets a model — and a
reader — notice that only nine were listed.

**Fall back generically, never silently.** An unrecognised payload with a list
in it still gets one line per item from whatever scalar fields it has. Returning
raw JSON is the last resort, not the default.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

__all__ = ["shape", "shape_generic", "SHAPERS"]

#: Rows beyond this are named but not listed. Deliberately high: the point of
#: this module is that a list arrives whole.
MAX_ROWS = 120


def _get(row: Any, *path: str, default: Any = "") -> Any:
    """Walk a nested dict, returning ``default`` rather than raising."""
    current = row
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _team(row: Any, side: str) -> str:
    """A team name short enough to scan and long enough to be unambiguous.

    ``shortName`` is preferred but not trusted: a feed that abbreviates
    "Manchester City" to "Manchester" makes two different clubs identical, and
    a wrong-but-plausible name is worse than a long one.
    """
    short = str(_get(row, side, "shortName") or "").strip()
    full = str(_get(row, side, "name") or "").strip()
    if full:
        trimmed = full.removesuffix(" FC").removesuffix(" AFC").strip() or full
        # Only take the short form when it is a real abbreviation of the full
        # name rather than a truncation of it.
        if short and len(short) >= 4 and not trimmed.startswith(short + " "):
            return short
        return trimmed
    return short or str(_get(row, side, "tla") or "?")


def _when(value: Any) -> str:
    """``2026-08-30T14:00:00Z`` → ``30 Aug 14:00``. Wrong-looking input passes through."""
    text = str(value or "")
    if len(text) < 16 or text[4] != "-":
        return text
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    try:
        month = months[int(text[5:7]) - 1]
    except (ValueError, IndexError):
        return text
    return f"{text[8:10]} {month} {text[11:16]}"


# -- football-data.org -----------------------------------------------------


def _matches(payload: dict[str, Any]) -> str | None:
    rows = payload.get("matches")
    if not isinstance(rows, list):
        return None
    competition = _get(payload, "competition", "name") or "matches"
    lines = [f"{len(rows)} {competition} match(es). Every one is listed below."]
    for row in rows[:MAX_ROWS]:
        home, away = _team(row, "homeTeam"), _team(row, "awayTeam")
        status = str(row.get("status") or "")
        full = _get(row, "score", "fullTime", default={}) or {}
        home_goals, away_goals = full.get("home"), full.get("away")
        if home_goals is None or away_goals is None:
            # No score yet: this is a fixture, and saying so stops a model
            # reporting 0-0 for a match that has not kicked off.
            lines.append(
                f"- {_when(row.get('utcDate'))} — {home} v {away} ({status.lower() or 'scheduled'})"
            )
            continue
        half = _get(row, "score", "halfTime", default={}) or {}
        detail = ""
        if half.get("home") is not None:
            detail = f" [HT {half['home']}-{half['away']}]"
        matchday = row.get("matchday")
        lines.append(
            f"- {home} {home_goals}-{away_goals} {away}{detail}"
            f"  ({_when(row.get('utcDate'))}"
            + (f", matchday {matchday}" if matchday else "")
            + (f", {status.lower()}" if status and status != "FINISHED" else "")
            + ")"
        )
    if len(rows) > MAX_ROWS:
        lines.append(f"… and {len(rows) - MAX_ROWS} more; narrow the date range to see them.")
    # Scorers are not in this endpoint on any tier, and a model asked for them
    # will otherwise go looking. Saying so once saves the search.
    lines.append("(This endpoint carries scores only — it has no goalscorers.)")
    return "\n".join(lines)


def _standings(payload: dict[str, Any]) -> str | None:
    groups = payload.get("standings")
    if not isinstance(groups, list) or not groups:
        return None
    out: list[str] = []
    for group in groups:
        table = group.get("table") if isinstance(group, dict) else None
        if not isinstance(table, list):
            continue
        label = str(group.get("group") or group.get("type") or "table").replace("_", " ").lower()
        out.append(f"{len(table)} teams ({label}). All listed:")
        for row in table[:MAX_ROWS]:
            out.append(
                f"- {row.get('position')}. {_team(row, 'team')} — "
                f"{row.get('points')} pts, played {row.get('playedGames')}, "
                f"W{row.get('won')} D{row.get('draw')} L{row.get('lost')}, "
                f"GF{row.get('goalsFor')} GA{row.get('goalsAgainst')} "
                f"GD{row.get('goalDifference')}"
            )
    return "\n".join(out) or None


def _scorers(payload: dict[str, Any]) -> str | None:
    rows = payload.get("scorers")
    if not isinstance(rows, list):
        return None
    lines = [f"{len(rows)} scorers, all listed:"]
    for row in rows[:MAX_ROWS]:
        lines.append(
            f"- {_get(row, 'player', 'name') or '?'} ({_team(row, 'team')}) — "
            f"{row.get('goals')} goals"
            + (f", {row['assists']} assists" if row.get("assists") is not None else "")
        )
    return "\n".join(lines)


def _competitions(payload: dict[str, Any]) -> str | None:
    rows = payload.get("competitions")
    if not isinstance(rows, list):
        return None
    lines = [f"{len(rows)} competitions, all listed:"]
    for row in rows[:MAX_ROWS]:
        lines.append(
            f"- {row.get('code')}: {row.get('name')} "
            f"({_get(row, 'area', 'name') or '?'})"
        )
    return "\n".join(lines)


# -- news ------------------------------------------------------------------


def _articles(payload: dict[str, Any]) -> str | None:
    rows = payload.get("articles")
    if not isinstance(rows, list):
        return None
    total = payload.get("totalResults")
    header = f"{len(rows)} articles"
    if isinstance(total, int) and total > len(rows):
        header += f" (of {total:,} matching; ask for more pages if you need them)"
    lines = [header + ", all listed:"]
    for row in rows[:MAX_ROWS]:
        source = _get(row, "source", "name") or row.get("source") or "?"
        summary = " ".join(str(row.get("description") or "").split())[:200]
        lines.append(
            f"- {row.get('title')} — {source}"
            f" ({str(row.get('publishedAt') or '')[:10]})"
            + (f"\n    {summary}" if summary else "")
        )
    return "\n".join(lines)


# -- weather ---------------------------------------------------------------


def _forecast(payload: dict[str, Any]) -> str | None:
    rows = payload.get("list")
    if not isinstance(rows, list) or "cod" not in payload:
        return None
    place = _get(payload, "city", "name") or "forecast"
    lines = [f"{len(rows)} forecast points for {place}, all listed:"]
    for row in rows[:MAX_ROWS]:
        weather = (row.get("weather") or [{}])[0]
        main = row.get("main") or {}
        rain = _get(row, "rain", "3h", default=0)
        lines.append(
            f"- {row.get('dt_txt', '')[:16]} — {weather.get('description', '?')}, "
            f"{main.get('temp', '?')}°C (feels {main.get('feels_like', '?')}°C)"
            + (f", {rain}mm rain" if rain else "")
        )
    return "\n".join(lines)


#: ``api name -> (path fragment, shaper)``. The first matching fragment wins,
#: and an empty fragment matches any path.
SHAPERS: dict[str, tuple[tuple[str, Callable[[dict[str, Any]], str | None]], ...]] = {
    "football": (
        ("standings", _standings),
        ("scorers", _scorers),
        ("competitions", _competitions),
        ("matches", _matches),
        ("", _matches),
    ),
    "news": (("", _articles),),
    "gnews": (("", _articles),),
    "weather": (("forecast", _forecast),),
}


def shape(api: str, path: str, payload: Any) -> str | None:
    """A complete, compact rendering of ``payload``, or ``None`` to leave it alone."""
    if not isinstance(payload, dict):
        return None
    for fragment, shaper in SHAPERS.get(api, ()):
        if fragment and fragment not in str(path):
            continue
        try:
            shaped = shaper(payload)
        except Exception:  # noqa: BLE001 - a bad shaper must never lose the data
            return None
        if shaped:
            return shaped
    return shape_generic(payload)


#: Keys that usually hold "the answer" in an API response, most specific first.
_LIST_KEYS = (
    "results", "items", "data", "records", "entries", "rows", "list", "matches",
    "articles", "values", "elements",
)
#: Fields worth showing in a generic rendering, in the order people read them.
_INTERESTING = (
    "name", "title", "id", "code", "status", "date", "time", "value", "count",
    "score", "amount", "type", "state", "description", "summary", "url",
)


def shape_generic(payload: dict[str, Any], *, max_rows: int = MAX_ROWS) -> str | None:
    """One line per item for an API nobody wrote a shaper for.

    Better than raw JSON for the same reason the specific shapers are: the
    failure is not that JSON is ugly, it is that a truncated list silently
    becomes a shorter list.
    """
    key = next(
        (k for k in _LIST_KEYS if isinstance(payload.get(k), list) and payload[k]),
        None,
    )
    if key is None:
        return None
    rows = payload[key]
    if not all(isinstance(r, (dict, str, int, float)) for r in rows):
        return None

    lines = [f"{len(rows)} {key}, all listed:"]
    for row in rows[:max_rows]:
        if not isinstance(row, dict):
            lines.append(f"- {row}")
            continue
        parts = []
        for field in _INTERESTING:
            value = row.get(field)
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                parts.append(f"{field}={str(value)[:90]}")
            if len(parts) >= 5:
                break
        if not parts:
            parts = [
                f"{k}={str(v)[:60]}"
                for k, v in list(row.items())[:5]
                if isinstance(v, (str, int, float, bool))
            ]
        lines.append("- " + ", ".join(parts) if parts else f"- {str(row)[:160]}")
    if len(rows) > max_rows:
        lines.append(f"… and {len(rows) - max_rows} more not shown; ask for a narrower query.")
    return "\n".join(lines)


def summarise_rows(rows: Iterable[Any]) -> int:
    """How many rows a shaped block claims to carry. Used by the loop's check."""
    return sum(1 for _ in rows)
