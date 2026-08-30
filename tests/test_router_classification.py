"""Ingestion and the rule-based tier classifier."""

from __future__ import annotations


import pytest

from itsbob.router.gatekeeper import Gatekeeper, classify_heuristically
from itsbob.router.ingestion import DEFAULT_EVENT_WINDOW, Snapshot, compress
from itsbob.router.tiers import Tier


def _tier(text: str) -> Tier:
    return classify_heuristically(compress(text)).tier


# -- ingestion -------------------------------------------------------------


def test_plain_text_becomes_the_message():
    assert compress("what time is it?").text == "what time is it?"


def test_a_json_string_is_parsed_as_structure():
    snapshot = compress('{"facts": {"cpu": 91}}')
    assert snapshot.facts == {"cpu": 91} and snapshot.text == ""


def test_text_that_merely_looks_like_json_does_not_raise():
    """The CLI crash that used to happen when someone pasted a README example."""
    for value in ["{not json", "'...'", "{", "[1, 2", "}{"]:
        assert compress(value).text == value


def test_a_json_array_is_treated_as_text_not_structure():
    assert compress("[1, 2, 3]").text == "[1, 2, 3]"


def test_empty_inputs_are_empty_not_errors():
    for value in ("", "   ", None, {}):
        assert compress(value).is_empty


def test_a_flat_dict_is_all_facts():
    assert compress({"stamina": 15, "minute": 60}).facts == {"stamina": 15, "minute": 60}


def test_events_are_truncated_to_the_most_recent():
    snapshot = compress({"events": list(range(50))})
    assert len(snapshot.events) == DEFAULT_EVENT_WINDOW
    assert snapshot.events[-1] == 49  # the tail, not the head


def test_message_is_an_alias_for_text():
    assert compress({"message": "hello"}).text == "hello"


def test_render_is_bounded():
    assert len(compress("x" * 50_000).render(max_chars=500)) < 700


def test_gamestate_alias_still_imports():
    from itsbob.router.ingestion import GameState

    assert GameState is Snapshot


# -- classification --------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["hello there", "thanks!", "summarize this paragraph", "rephrase that more politely"],
)
def test_small_talk_is_cheapest(text):
    assert _tier(text) is Tier.C


@pytest.mark.parametrize(
    "text",
    ["what did I say about the deploy?", "when did I last push to main?",
     "do you remember my wifi password", "remind me to call the bank",
     "what is my home address"],
)
def test_asking_about_a_thing_is_not_doing_it(text):
    """Recall beats the destructive vocabulary — 'deploy' here is a noun."""
    assert _tier(text) is Tier.C


@pytest.mark.parametrize(
    "text",
    ["read config.yaml", "run the test suite", "list the files in src",
     "search the logs for timeouts", "fetch the latest release notes"],
)
def test_tool_work_is_standard(text):
    assert _tier(text) is Tier.A


@pytest.mark.parametrize(
    "text",
    ["delete the old backups", "deploy to production", "email the team the notes",
     "pay the invoice", "force push the branch", "migrate the database"],
)
def test_irreversible_work_is_the_strongest_tier(text):
    assert _tier(text) is Tier.S


@pytest.mark.parametrize(
    "text",
    ["should I rewrite this in Rust?", "compare Postgres versus SQLite for this",
     "what's the best way to structure the module", "is it safe to remove this class",
     "why does the build fail intermittently"],
)
def test_judgement_is_the_strongest_tier(text):
    assert _tier(text) is Tier.S


def test_judgement_outranks_recall():
    """'should I delete X?' is a question, but it is asking for a decision."""
    assert _tier("should I delete the backups?") is Tier.S


@pytest.mark.parametrize(
    "text,not_tier",
    [
        ("merge these two CSV files", Tier.S),      # "merge the" is a prefix of "merge these"
        ("fetch the latest release notes", Tier.S),  # "release" as a noun
        ("post theory notes somewhere", Tier.S),     # "post the" inside "post theory"
        ("what are the release notes", Tier.S),
    ],
)
def test_vocabulary_matches_on_word_boundaries(text, not_tier):
    """Substring matching sent whole categories of ordinary requests to premium."""
    assert _tier(text) is not not_tier


def test_boundary_matching_still_catches_the_real_verb():
    assert _tier("merge the branch into main") is Tier.S
    assert _tier("release the build to production") is Tier.S


def test_length_is_only_the_last_resort():
    assert _tier("mm") is Tier.C
    assert _tier("lorem ipsum dolor " * 15) is Tier.B   # moderate (285 chars)
    assert _tier("lorem ipsum dolor " * 40) is Tier.A   # long


def test_the_classifier_never_raises_on_odd_input():
    for value in ["", "?????", "🙂🙂🙂", "SELECT * FROM x; --", "\x00\x01"]:
        assert classify_heuristically(compress(value)).tier in set(Tier)


# -- the model path --------------------------------------------------------


class _Model:
    def __init__(self, reply):
        self.reply = reply

    def complete_with_fallback(self, request, preferred_model=None):
        from itsbob.llm.base import LLMResponse

        return LLMResponse(text=self.reply, model="m", provider="p")


def test_a_model_tag_is_used_when_it_parses():
    gate = Gatekeeper(local_provider=_Model('{"tag": "COMPLEX", "fingerprint": "a b c d e"}'))
    decision = gate.classify(compress("hello"))
    assert decision.tier is Tier.S and decision.fingerprint == "a b c d e"


def test_tag_synonyms_are_accepted():
    """Small models produce the word that means the thing, not always the token."""
    for reply, expected in (
        ('{"tag": "COMPLEX"}', Tier.S),
        ('{"tag": "STANDARD"}', Tier.A),
        ('{"tag": "SIMPLE"}', Tier.B),
        ('{"tag": "TRIVIAL"}', Tier.C),
        ('{"tag": "CLOUD_A"}', Tier.S),      # legacy name
        ('{"tag": "LOCAL_SUM"}', Tier.C),    # legacy name
    ):
        assert Gatekeeper(local_provider=_Model(reply)).classify(compress("x")).tier is expected


def test_an_unparseable_reply_falls_back_to_the_heuristic():
    gate = Gatekeeper(local_provider=_Model("I think this is probably fine?"))
    decision = gate.classify(compress("delete the backups"))
    assert decision.source == "heuristic" and decision.tier is Tier.S


def test_a_failing_model_falls_back_to_the_heuristic():
    class Boom:
        def complete_with_fallback(self, *a, **k):
            raise RuntimeError("ollama is down")

    assert Gatekeeper(local_provider=Boom()).classify(compress("hi")).source == "heuristic"


def test_an_unknown_routine_name_routes_as_standard_not_as_a_halt():
    gate = Gatekeeper(routines=("BACKUP",), local_provider=_Model('{"tag": "ROUTINE", "routine": "INVENTED"}'))
    decision = gate.classify(compress("x"))
    assert decision.tier is Tier.B
    assert "named no known routine" in decision.reasoning


def test_a_known_routine_is_tier_d():
    gate = Gatekeeper(routines=("BACKUP",), local_provider=_Model('{"tag": "ROUTINE", "routine": "BACKUP"}'))
    decision = gate.classify(compress("x"))
    assert decision.tier is Tier.D and decision.metadata["routine"] == "BACKUP"


def test_a_deterministic_trigger_skips_the_model_entirely():
    class Registry:
        def first_triggered(self, snapshot):
            return "NIGHTLY"

    gate = Gatekeeper(registry=Registry(), local_provider=_Model('{"tag": "COMPLEX"}'))
    decision = gate.classify(compress("anything"))
    assert decision.tier is Tier.D and decision.source == "trigger"


def test_a_broken_trigger_does_not_block_routing():
    class Registry:
        def first_triggered(self, snapshot):
            raise RuntimeError("bad trigger")

    assert Gatekeeper(registry=Registry()).classify(compress("hello")).tier is Tier.C


def test_tier_rank_is_ordered_cheapest_first():
    assert [t.rank for t in (Tier.D, Tier.C, Tier.B, Tier.A, Tier.S, Tier.H)] == [0, 1, 2, 3, 4, 5]


def test_only_model_tiers_are_answered_by_a_model():
    assert [t for t in Tier if t.is_model] == [Tier.C, Tier.B, Tier.A, Tier.S]
    assert not Tier.D.is_model and not Tier.H.is_model
