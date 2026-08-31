"""Built-in API specs, so a key in ``.env`` is the whole setup.

The catalog in :mod:`itsbob.tools.http` is deliberately configuration-driven:
adding an API is a JSON entry, not a code change. That is the right default and
it stays. But it puts the base URL, the auth style, the header name and the
query parameter on the person adding the key — four chances to get something
wrong before anything works, for services whose answers are already known.

So the handful that this assistant is actually set up around ship as specs
here. A spec appears in the catalog only when its key is present in the
environment, and never overwrites one the user configured themselves: an entry
in ``apis.json`` or an ``ITSBOB_API_*`` variable always wins, because someone
who wrote it out meant it.
"""

from __future__ import annotations

import os
from typing import Mapping

from ..tools.http import ApiCatalog, ApiSpec

__all__ = ["BUILTIN_SPECS", "register_builtins", "builtin_status"]


#: Each entry is a spec that is added when ``key_env`` is set. Where a service
#: offers several auth styles, the one chosen here is the one its docs use.
BUILTIN_SPECS: tuple[ApiSpec, ...] = (
    ApiSpec(
        name="weather",
        base_url="https://api.openweathermap.org/data/2.5",
        key_env="OPENWEATHER_API_KEY",
        auth="query",
        query_param="appid",
        description=(
            "OpenWeather: current conditions and 5-day forecast. Prefer the "
            "`weather` tool, which fills in the configured location for you."
        ),
        examples=(
            (
                "path='weather', params={'lat': 53.77, 'lon': -0.33, 'units': 'metric'}",
                "conditions right now",
            ),
            (
                "path='forecast', params={'lat': 53.77, 'lon': -0.33, 'units': 'metric'}",
                "the five-day, three-hourly forecast",
            ),
        ),
    ),
    ApiSpec(
        name="news",
        base_url="https://newsapi.org/v2",
        key_env="NEWSAPI_KEY",
        auth="header",
        header_name="X-Api-Key",
        description=(
            "NewsAPI: headlines and article search. Prefer the `news` tool, which "
            "merges this with GNews and strips the payload down to headlines."
        ),
        examples=(
            (
                "path='everything', params={'q': 'ceasefire', 'sortBy': 'publishedAt', "
                "'pageSize': 10, 'language': 'en'}",
                "articles matching a query, newest first",
            ),
            ("path='top-headlines', params={'country': 'gb', 'pageSize': 10}", "UK headlines"),
        ),
    ),
    ApiSpec(
        name="gnews",
        base_url="https://gnews.io/api/v4",
        key_env="GNEWS_API_KEY",
        auth="query",
        query_param="apikey",
        description=(
            "GNews: a second news source for the `news` tool, and its fallback "
            "when NewsAPI is rate-limited."
        ),
        examples=(
            ("path='search', params={'q': 'election', 'lang': 'en', 'max': 10}", "a query"),
            (
                "path='top-headlines', params={'category': 'world', 'lang': 'en'}",
                "world headlines",
            ),
        ),
    ),
    ApiSpec(
        name="football",
        base_url="https://api.football-data.org/v4",
        key_env="FOOTBALL_DATA_KEY",
        auth="header",
        header_name="X-Auth-Token",
        description=(
            "football-data.org: fixtures, results, standings and scorers. "
            "`path` is always required. Competition codes: PL (Premier League), "
            "ELC (Championship), CL, BL1, SA, PD, FL1. For *results*, pass "
            "status=FINISHED — without it you get scheduled fixtures with null "
            "scores, which look like a bug."
        ),
        examples=(
            (
                "path='competitions/PL/matches', params={'status': 'FINISHED', 'limit': 10}",
                "the most recent finished Premier League matches, with scores",
            ),
            ("path='competitions/PL/matches', params={'matchday': 3}", "one whole matchday"),
            (
                "path='competitions/PL/matches', "
                "params={'dateFrom': '2026-08-28', 'dateTo': '2026-08-31'}",
                "a date range — both dates required, at most 10 days apart",
            ),
            ("path='competitions/PL/standings'", "the current table"),
            ("path='competitions/PL/scorers'", "top scorers"),
        ),
    ),
)


def register_builtins(
    catalog: ApiCatalog, env: Mapping[str, str] | None = None
) -> list[str]:
    """Add every built-in whose key is set and which is not already configured.

    Returns the names actually added, so a caller can say what it wired up.
    """
    env = os.environ if env is None else env
    added: list[str] = []
    for spec in BUILTIN_SPECS:
        if catalog.get(spec.name) is not None:
            continue  # the user's own entry wins
        if not env.get(spec.key_env, "").strip():
            continue
        catalog.register(spec)
        added.append(spec.name)
    return added


def builtin_status(env: Mapping[str, str] | None = None) -> list[dict[str, object]]:
    """What each built-in needs and whether it has it — for ``itsbob doctor``."""
    env = os.environ if env is None else env
    return [
        {
            "name": spec.name,
            "key_env": spec.key_env,
            "configured": bool(env.get(spec.key_env, "").strip()),
            # First sentence, split on ". " rather than "." so
            # "football-data.org: ..." does not lose everything after the dot
            # in its own domain name.
            "description": spec.description.split(". ")[0].rstrip("."),
        }
        for spec in BUILTIN_SPECS
    ]
