"""The command line and the browser API. Offline: no test here calls a model."""

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


def test_memory_stats_flags_keyword_only_recall(home, capsys):
    main(["memory", "stats"])
    assert "keyword-only" in capsys.readouterr().out


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


# -- the browser API -------------------------------------------------------


@pytest.fixture
def client(home):
    pytest.importorskip("flask")
    from itsbob.gui.app import create_app

    return create_app(home).test_client()


def test_the_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"itsbob" in response.data


def test_status_describes_every_subsystem(client):
    body = client.get("/api/status").get_json()
    assert set(body) >= {"policy", "tools", "tiers", "memory", "tasks"}
    assert "run_shell" in body["tools"]


def test_an_empty_store_reports_zero_not_missing(client):
    """LongTermMemory defines __len__, so truthiness would hide it entirely."""
    assert client.get("/api/status").get_json()["memory"]["records"] == 0


def test_an_empty_message_is_rejected(client):
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400


def test_tasks_can_be_created_and_removed(client):
    created = client.post(
        "/api/task", json={"name": "t", "prompt": "do it", "schedule": "every 30m"}
    ).get_json()
    assert created["task"]["schedule"] == "every 30m"
    assert client.post("/api/task/remove", json={"id": created["task"]["id"]}).get_json()["ok"]


def test_a_bad_schedule_is_a_400_not_a_500(client):
    assert client.post("/api/task", json={"name": "t", "prompt": "x", "schedule": "soon"}).status_code == 400


def test_memory_search_and_forget(client, home):
    from itsbob.memory.base import MemoryRecord
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(home / "memory.sqlite", embedder=None)
    record = store.add(MemoryRecord(content="the spare key is under the pot"))
    store.close()

    hits = client.get("/api/memory?q=spare key").get_json()["hits"]
    assert hits and "spare key" in hits[0]["content"]
    assert client.post("/api/memory/forget", json={"id": record.id}).get_json()["ok"] is True


def test_reset_clears_the_conversation_not_the_memory(client):
    assert client.post("/api/reset").get_json()["ok"] is True
    assert client.get("/api/status").get_json()["turns"] == 0
