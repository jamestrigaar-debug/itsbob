"""The command line. Offline: no test here calls a model.

The browser interface is tested in test_gui.py.
"""

from __future__ import annotations

import json

import pytest

from itsbob.cli import main


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ITSBOB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ITSBOB_EMBED_OFFLINE", "true")  # no embedding API calls
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return tmp_path / "home"


# -- the CLI ---------------------------------------------------------------


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage: itsbob" in capsys.readouterr().out


def test_doctor_reports_every_subsystem(home, capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    for section in ("model tiers:", "memory:", "embeddings:", "tools:", "tasks:"):
        assert section in out


def test_memory_add_then_search_round_trips(home, capsys):
    main(["memory", "add", "the fuse box is behind the coats", "--tags", "house"])
    capsys.readouterr()
    assert main(["memory", "search", "fuse box"]) == 0
    assert "behind the coats" in capsys.readouterr().out


def test_memory_search_with_no_hits_is_not_an_error(home, capsys):
    assert main(["memory", "search", "anything at all"]) == 0
    assert "nothing found" in capsys.readouterr().out


def test_memory_stats_flags_the_offline_embedder(home, capsys):
    """The `home` fixture sets ITSBOB_EMBED_OFFLINE, so this is the fallback."""
    main(["memory", "stats"])
    out = capsys.readouterr().out
    assert "offline hashing embedder" in out
    assert "cannot match paraphrases" in out


def test_memory_stats_flags_a_missing_embedder(home, capsys, monkeypatch):
    monkeypatch.setattr("itsbob.cli._open_memory", lambda *a, **k: _NoEmbedder())
    main(["memory", "stats"])
    assert "no embedding model is configured" in capsys.readouterr().out


class _NoEmbedder:
    def stats(self):
        return {
            "database": ":memory:", "records": 0, "fts5": True, "numpy": False,
            "embedder": None, "embedded": 0, "unembedded": 0, "embed_errors": 0,
            "last_embed_error": None, "offline_embedder": False,
            "semantic_recall": False, "degraded": False,
        }


def test_memory_stats_describes_the_store_not_the_command(home, capsys, monkeypatch):
    """`stats` used to open the store without an embedder and then report that
    the store had no embedder — a diagnostic describing itself."""
    monkeypatch.delenv("ITSBOB_EMBED_OFFLINE", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "not-used-no-call-is-made")
    main(["memory", "stats"])
    out = capsys.readouterr().out
    assert "gemini-embedding" in out
    assert "keyword-only" not in out


def test_memory_forget_reports_a_missing_id(home, capsys):
    assert main(["memory", "forget", "nope"]) == 0
    assert "no memory with id" in capsys.readouterr().out


def test_memory_list_shows_what_was_added(home, capsys):
    main(["memory", "add", "a durable fact"])
    capsys.readouterr()
    main(["memory", "list"])
    assert "a durable fact" in capsys.readouterr().out


def test_task_add_and_list(home, capsys):
    assert main(["task", "add", "inbox", "check the inbox", "every 30m"]) == 0
    capsys.readouterr()
    main(["task", "list"])
    out = capsys.readouterr().out
    assert "inbox" in out and "every 30m" in out


def test_task_add_with_a_bad_schedule_fails_cleanly(home, capsys):
    assert main(["task", "add", "x", "y", "whenever I feel like it"]) == 1
    assert "could not parse schedule" in capsys.readouterr().err


def test_task_list_when_empty_says_how_to_add_one(home, capsys):
    main(["task", "list"])
    assert "itsbob task add" in capsys.readouterr().out


def test_task_disable_then_enable(home, capsys):
    main(["task", "add", "t", "do it", "every 30m"])
    capsys.readouterr()
    assert main(["task", "disable", "t"]) == 0
    main(["task", "list"])
    assert "[off]" in capsys.readouterr().out
    assert main(["task", "enable", "t"]) == 0


def test_task_remove(home, capsys):
    main(["task", "add", "t", "do it", "every 30m"])
    capsys.readouterr()
    assert main(["task", "remove", "t"]) == 0
    main(["task", "list"])
    assert "no tasks" in capsys.readouterr().out


def test_an_unknown_task_is_an_error_not_a_traceback(home, capsys):
    assert main(["task", "show", "ghost"]) == 1
    assert "no task named" in capsys.readouterr().err


def test_tools_lists_the_registry_and_the_policy(home, capsys):
    assert main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "run_shell" in out and "mode" in out


def test_audit_with_nothing_recorded(home, capsys):
    assert main(["audit"]) == 0
    assert "no tool activity" in capsys.readouterr().out


def test_an_unexpected_error_is_a_message_not_a_traceback(home, capsys, monkeypatch):
    monkeypatch.setattr("itsbob.cli._open_memory", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk gone")))
    assert main(["memory", "stats"]) == 1
    err = capsys.readouterr().err
    assert "disk gone" in err and "ITSBOB_DEBUG" in err


def test_the_simulation_still_runs(home, capsys):
    assert main(["run", "--ticks", "3", "--policy", "heuristic", "--offline"]) == 0
    assert capsys.readouterr().out.count("energy") == 3
