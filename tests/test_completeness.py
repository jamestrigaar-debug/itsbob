"""Answers that promise a list and then do not give one.

Reproduced from a real transcript: asked for a matchday, itsbob reported that
"10 matches were played" and named two. Half of that was mechanical — eight of
the ten were clipped out of the tool output before the model ever saw them —
and half was the habit of announcing a count instead of paying it out. Both
halves are tested here.
"""

from __future__ import annotations

import json

import pytest

from itsbob.agent.completeness import inspect, rows_available
from itsbob.integrations.shaping import shape, shape_generic
from itsbob.tools.base import ToolResult


def _match(home, away, h, a, i=0):
    """A football-data v4 match, with the bulk that crowds out the answer."""
    return {
        "id": 500000 + i, "utcDate": f"2026-08-30T1{i % 5}:00:00Z",
        "status": "FINISHED", "matchday": 3, "lastUpdated": "2026-08-30T18:00:00Z",
        "homeTeam": {"id": i, "name": home, "shortName": home.removesuffix(" FC"),
                     "tla": home[:3].upper(), "crest": f"https://crests.football-data.org/{i}.png"},
        "awayTeam": {"id": i + 50, "name": away, "shortName": away.removesuffix(" FC"),
                     "tla": away[:3].upper(), "crest": f"https://crests.football-data.org/{i+50}.png"},
        "score": {"winner": "HOME_TEAM", "duration": "REGULAR",
                  "fullTime": {"home": h, "away": a},
                  "halfTime": {"home": max(0, h - 1), "away": max(0, a - 1)}},
        "odds": {"msg": "Activate Odds-Package in User-Panel"},
        "referees": [{"id": 900 + i, "name": "A Referee", "type": "REFEREE",
                      "nationality": "England"}],
    }


TEAMS = [
    ("Sunderland FC", "Fulham FC", 1, 0), ("Liverpool FC", "Chelsea FC", 2, 2),
    ("Manchester City FC", "Tottenham Hotspur FC", 3, 1), ("Everton FC", "Brentford FC", 0, 0),
    ("Arsenal FC", "Nottingham Forest FC", 2, 0), ("Brighton FC", "Wolverhampton FC", 1, 1),
    ("Newcastle United FC", "AFC Bournemouth", 2, 1), ("West Ham United FC", "Aston Villa FC", 0, 3),
    ("Crystal Palace FC", "Leeds United FC", 1, 1), ("Burnley FC", "Manchester United FC", 0, 2),
]
MATCHDAY = {
    "competition": {"name": "Premier League", "code": "PL"},
    "resultSet": {"count": 10},
    "matches": [_match(*t, i=i) for i, t in enumerate(TEAMS)],
}


# -- the mechanical half: the data has to reach the model at all -----------


def test_the_raw_payload_loses_most_of_the_list_to_truncation():
    """This is the measurement the whole module exists because of."""
    raw = json.dumps(MATCHDAY, indent=2)
    seen = ToolResult(ok=True, output=f"HTTP 200\n{raw}").render(max_chars=3000)
    survived = sum(1 for home, away, *_ in TEAMS if home[:6] in seen and away[:6] in seen)
    assert survived <= 3, "the premise changed — raw JSON now fits, so re-check the shaper"


def test_shaping_carries_every_row_and_costs_a_fraction():
    shaped = shape("football", "competitions/PL/matches", MATCHDAY)
    assert shaped.startswith("10 Premier League match(es)")
    for home, away, h, a in TEAMS:
        assert f"{h}-{a}" in shaped
        assert home.removesuffix(" FC") in shaped
        assert away.removesuffix(" FC").removeprefix("AFC ") in shaped
    # Complete, and an order of magnitude smaller than the JSON that lost most of it.
    assert len(shaped) < len(json.dumps(MATCHDAY)) / 5
    # And it survives the loop's observation budget whole.
    seen = ToolResult(ok=True, output=shaped).render(max_chars=3000)
    assert sum(1 for home, *_ in TEAMS if home.removesuffix(" FC") in seen) == 10


def test_a_fixture_is_not_reported_as_a_nil_nil_draw():
    """A match with no score has not been played; 0-0 would be a wrong answer."""
    upcoming = {
        "competition": {"name": "Premier League"},
        "matches": [{"utcDate": "2026-09-01T19:00:00Z", "status": "SCHEDULED",
                     "homeTeam": {"name": "Aston Villa FC"}, "awayTeam": {"name": "Arsenal FC"},
                     "score": {"fullTime": {"home": None, "away": None}}}],
    }
    shaped = shape("football", "competitions/PL/matches", upcoming)
    assert "0-0" not in shaped
    assert "scheduled" in shaped and "Aston Villa v Arsenal" in shaped


def test_a_short_name_that_is_a_truncation_is_not_trusted():
    """A feed abbreviating "Manchester City" to "Manchester" makes two clubs one."""
    payload = {"matches": [{
        "homeTeam": {"name": "Manchester City FC", "shortName": "Manchester"},
        "awayTeam": {"name": "Manchester United FC", "shortName": "Man United"},
        "score": {"fullTime": {"home": 1, "away": 2}}}]}
    shaped = shape("football", "matches", payload)
    assert "Manchester City 1-2 Man United" in shaped


def test_standings_and_scorers_are_listed_whole():
    table = {"standings": [{"type": "TOTAL", "table": [
        {"position": i + 1, "team": {"name": f"Team {i}"}, "points": 30 - i,
         "playedGames": 12, "won": 9, "draw": 2, "lost": 1,
         "goalsFor": 20, "goalsAgainst": 8, "goalDifference": 12}
        for i in range(20)]}]}
    shaped = shape("football", "competitions/PL/standings", table)
    assert shaped.startswith("20 teams")
    assert all(f"Team {i}" in shaped for i in range(20))

    scorers = {"scorers": [
        {"player": {"name": f"Player {i}"}, "team": {"name": "Someone FC"}, "goals": 9 - i}
        for i in range(8)]}
    shaped = shape("football", "competitions/PL/scorers", scorers)
    assert shaped.startswith("8 scorers")
    assert all(f"Player {i}" in shaped for i in range(8))


def test_news_articles_keep_every_headline_and_drop_the_bulk():
    payload = {"totalResults": 2395, "articles": [
        {"title": f"Headline {i}", "source": {"name": "Wire"},
         "publishedAt": "2026-08-30T20:00:00Z", "description": "d" * 500,
         "content": "c" * 4000, "urlToImage": "https://example.test/huge.jpg"}
        for i in range(12)]}
    shaped = shape("news", "everything", payload)
    assert all(f"Headline {i}" in shaped for i in range(12))
    assert "2,395" in shaped  # says what was left behind
    assert "cccc" not in shaped and "huge.jpg" not in shaped


def test_an_unknown_api_still_gets_one_line_per_item():
    """Raw JSON is the last resort, not the default — truncation loses rows."""
    payload = {"results": [{"name": f"thing {i}", "status": "ok", "value": i} for i in range(30)]}
    shaped = shape_generic(payload)
    assert shaped.startswith("30 results")
    assert all(f"thing {i}" in shaped for i in range(30))


def test_shaping_a_payload_with_no_list_leaves_it_alone():
    assert shape("football", "x", {"message": "Not Found"}) is None
    assert shape_generic({"temperature": 11.4}) is None
    assert shape("football", "x", "not a dict at all") is None


def test_a_very_long_list_names_what_it_is_not_showing():
    """Fields may be dropped to fit. Rows may not — silently."""
    payload = {"items": [{"name": f"row {i}"} for i in range(400)]}
    shaped = shape_generic(payload, max_rows=50)
    assert shaped.startswith("400 items")
    assert "and 350 more not shown" in shaped


# -- the behavioural half: promising a list and not giving one -------------


OBSERVATIONS = [shape("football", "competitions/PL/matches", MATCHDAY)]


def test_a_count_standing_in_for_the_list_is_caught():
    answer = ("I found that 10 matches were played. Chelsea beat Brighton and "
              "Sunderland beat Fulham, among others.")
    found = inspect(answer, OBSERVATIONS)
    assert found.short
    assert found.available == 10 and found.listed == 0
    assert "among others" in found.note()


def test_an_enumerated_answer_is_left_alone():
    answer = "Here are all ten:\n" + "\n".join(
        f"- {h.removesuffix(' FC')} {a}-{b} {w.removesuffix(' FC')}"
        for h, w, a, b in TEAMS
    )
    assert not inspect(answer, OBSERVATIONS).short


def test_a_genuinely_short_answer_is_not_a_shortfall():
    """Brevity is only a fault when something was promised."""
    assert not inspect("The disk is 94% full.", OBSERVATIONS).short
    assert not inspect("Yes.", OBSERVATIONS).short
    assert not inspect("I could not reach the API at all.", OBSERVATIONS).short


def test_nothing_fires_when_no_tool_returned_a_list():
    answer = "Several things happened today, among others."
    assert not inspect(answer, ["HTTP 200\nsome prose with no list in it"]).short
    assert not inspect(answer, []).short


def test_a_list_of_one_or_two_is_never_a_shortfall():
    """Not a list anybody can under-deliver on, and a wrong call costs a model call."""
    two = shape("football", "matches", {"competition": {"name": "PL"},
                                        "matches": [_match(*TEAMS[0]), _match(*TEAMS[1], i=1)]})
    assert rows_available([two]) == 2
    assert not inspect("2 matches were played, among others.", [two]).short


def test_a_partial_list_is_still_a_shortfall():
    """Listing four of ten is the exact failure, not a lesser version of it."""
    answer = "10 matches were played:\n" + "\n".join(
        f"- {h.removesuffix(' FC')} won" for h, *_ in TEAMS[:4]
    )
    found = inspect(answer, OBSERVATIONS)
    assert found.short and found.listed == 4 and found.available == 10


@pytest.mark.parametrize(
    "hedge",
    ["and others", "the rest were also played", "a number of results", "and more",
     "I was unable to compile the full list"],
)
def test_every_hedge_that_stands_in_for_a_list_is_caught(hedge):
    assert inspect(f"There were results. {hedge}.", OBSERVATIONS).short
