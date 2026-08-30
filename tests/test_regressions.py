"""One test per bug found in the audit sweep, named for the bug.

Each of these failed before the fix. They are grouped here rather than spread
across the suite so that a future change that reintroduces one is obvious from
the test name alone.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from itsbob.config import find_dotenv, load_dotenv
from itsbob.llm.embeddings import HashingEmbedder
from itsbob.logfile import JsonlFile
from itsbob.memory.base import MemoryRecord
from itsbob.memory.long_term import LongTermMemory
from itsbob.router.tiers import Tier
from itsbob.store import Database
from itsbob.tools.base import Risk


# -- 1. concurrent writes silently lost rows -------------------------------


def test_concurrent_writes_do_not_lose_rows(tmp_path):
    """Six threads writing 150 memories landed 34 of them before the fix."""
    store = LongTermMemory(tmp_path / "m.sqlite", embedder=HashingEmbedder(dims=32))
    errors: list[str] = []

    def worker(n: int) -> None:
        try:
            for i in range(25):
                store.add(MemoryRecord(content=f"thread {n} item {i}"))
                store.search("thread", limit=3)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store) == 150


def test_two_handles_on_one_file_share_a_lock(tmp_path):
    """The lock is per database file, not per object — they share the file."""
    path = tmp_path / "m.sqlite"
    a = LongTermMemory(path, embedder=None)
    b = LongTermMemory(path, embedder=None)
    errors: list[str] = []

    def worker(store, n):
        try:
            for i in range(40):
                store.add(MemoryRecord(content=f"x{n}-{i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(a if i % 2 else b, i)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(a) == 240


def test_databases_use_wal_so_readers_do_not_block(tmp_path):
    db = Database(tmp_path / "x.sqlite", schema="CREATE TABLE t (v INTEGER)")
    assert db.scalar("PRAGMA journal_mode") == "wal"
    assert int(db.scalar("PRAGMA busy_timeout")) > 0


def test_a_failed_transaction_rolls_back(tmp_path):
    db = Database(tmp_path / "x.sqlite", schema="CREATE TABLE t (v INTEGER)")
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise RuntimeError("boom")
    assert db.scalar("SELECT COUNT(*) FROM t") == 0


def test_nested_transactions_commit_once(tmp_path):
    db = Database(tmp_path / "x.sqlite", schema="CREATE TABLE t (v INTEGER)")
    with db.transaction() as conn:
        conn.execute("INSERT INTO t VALUES (1)")
        with db.transaction() as inner:
            inner.execute("INSERT INTO t VALUES (2)")
    assert db.scalar("SELECT COUNT(*) FROM t") == 2


# -- 2. .env was only found in the current directory ------------------------


def test_dotenv_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / ".env").write_text("FOUND_FROM_PARENT=yes\n")
    monkeypatch.chdir(tmp_path / "a" / "b")
    assert any(p.name == ".env" for p in find_dotenv())
    env: dict[str, str] = {}
    load_dotenv(env=env)
    assert env["FOUND_FROM_PARENT"] == "yes"


def test_dotenv_is_found_in_the_itsbob_home(tmp_path, monkeypatch):
    """`itsbob serve` is normally started from somewhere else entirely."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("FROM_HOME=yes\n")
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    monkeypatch.setenv("ITSBOB_HOME", str(home))
    monkeypatch.chdir(empty)
    env: dict[str, str] = {}
    load_dotenv(env=env)
    assert env["FROM_HOME"] == "yes"


def test_the_nearest_dotenv_wins(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("KEY=from-home\nONLY_HOME=yes\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("KEY=from-project\n")
    monkeypatch.setenv("ITSBOB_HOME", str(home))
    monkeypatch.chdir(project)
    env: dict[str, str] = {}
    load_dotenv(env=env)
    assert env["KEY"] == "from-project"
    assert env["ONLY_HOME"] == "yes"  # home still fills the gaps


def test_an_unreadable_dotenv_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").mkdir()  # a directory where a file is expected
    assert load_dotenv(env={}) == {}


# -- 3. the offline provider masked real failures ---------------------------


def test_the_offline_provider_is_not_a_fallback_behind_a_real_one():
    from itsbob.agent.brain import build_brain

    with_key = build_brain(env={"GOOGLE_API_KEY": "x"})
    names = [p.name for p in with_key.router_for(Tier.B).providers]
    assert "echo" not in names, "a configured provider that fails must fail, not be papered over"


def test_the_offline_provider_stands_in_when_nothing_is_configured():
    from itsbob.agent.brain import build_brain

    assert [p.name for p in build_brain(env={}).router_for(Tier.B).providers] == ["echo"]


def test_the_offline_provider_answers_in_the_agents_shape():
    """Otherwise a zero-config run loops through escalation for no reason."""
    from itsbob.llm.base import LLMRequest, user
    from itsbob.llm.providers import EchoProvider

    response = EchoProvider().complete(
        LLMRequest(messages=[user('reply with {"thought":..., "final":...}')], json_mode=True)
    )
    payload = json.loads(response.text)
    assert payload["tool"] is None
    assert "GOOGLE_API_KEY" in payload["final"]


# -- 4/5. enum ordering was alphabetical ------------------------------------


def test_tier_orders_by_cost_not_alphabetically():
    assert Tier.A > Tier.B > Tier.C > Tier.D
    assert max([Tier.C, Tier.A, Tier.B]) is Tier.A
    assert [t.value for t in sorted([Tier.A, Tier.D, Tier.S, Tier.B, Tier.C])] == list("DCBAS")


def test_risk_orders_by_severity_in_both_directions():
    assert Risk.READ < Risk.WRITE < Risk.NETWORK < Risk.EXECUTE < Risk.DESTRUCTIVE
    assert Risk.DESTRUCTIVE > Risk.READ
    assert max([Risk.READ, Risk.DESTRUCTIVE, Risk.WRITE]) is Risk.DESTRUCTIVE
    assert [r.value for r in sorted([Risk.DESTRUCTIVE, Risk.READ, Risk.EXECUTE])] == [
        "read", "execute", "destructive"
    ]


@pytest.mark.parametrize("enum,member,text", [(Tier, Tier.B, "B"), (Risk, Risk.READ, "read")])
def test_string_and_json_compatibility_is_preserved(enum, member, text):
    assert member == text
    assert json.dumps({"v": member}) == f'{{"v": "{text}"}}'
    assert {member: 1}[member] == 1


# -- 6. logs grew without bound ---------------------------------------------


def test_jsonl_rotates_and_stays_bounded(tmp_path):
    log = JsonlFile(tmp_path / "a.jsonl", max_bytes=5_000, keep=2)
    for i in range(500):
        log.append({"i": i, "pad": "x" * 200})
    assert log.size() < 25_000
    assert (tmp_path / "a.1.jsonl").exists()
    assert not (tmp_path / "a.3.jsonl").exists()  # beyond keep, deleted


def test_history_reads_continuously_across_a_rotation(tmp_path):
    log = JsonlFile(tmp_path / "a.jsonl", max_bytes=2_000, keep=3)
    for i in range(200):
        log.append({"i": i})
    entries = log.read()
    assert [e["i"] for e in entries] == sorted(e["i"] for e in entries)  # oldest first


def test_a_torn_line_does_not_break_reading(tmp_path):
    path = tmp_path / "a.jsonl"
    log = JsonlFile(path)
    log.append({"ok": 1})
    with path.open("a") as handle:
        handle.write('{"truncated": ')  # a crash mid-write
    log.append({"ok": 2})
    assert [e.get("ok") for e in log.read()] == [1, 2]


def test_the_audit_log_rotates(tmp_path):
    from itsbob.tools.audit import AuditLog
    from itsbob.tools.base import ToolCall, ToolResult

    log = AuditLog(path=tmp_path / "audit.jsonl", max_bytes=5_000, backups=1)
    for _ in range(300):
        log.record(ToolCall("t", {"x": "y" * 100}), ToolResult(ok=True, output="z" * 200, tool="t"))
    assert log.stats()["bytes"] < 20_000


# -- 7. vector recall materialized every float ------------------------------


def test_vectors_are_never_python_float_lists():
    """The invariant in both paths: a list of Python floats costs ~10x."""
    from array import array

    store = LongTermMemory(":memory:", embedder=HashingEmbedder(dims=128))
    store.add_many([MemoryRecord(content=f"item {i}") for i in range(20)])
    _, matrix = store._load_vectors(store.embedder.signature)
    row = matrix[0]
    assert not isinstance(row, list)
    assert isinstance(row, array) or type(row).__module__.startswith("numpy")


def test_the_pure_python_path_stays_compact_and_correct(monkeypatch):
    """Exercised explicitly, since numpy is optional and CI may have it."""
    from array import array

    store = LongTermMemory(":memory:", embedder=HashingEmbedder(dims=128))
    store._np = None  # force the fallback
    store.add_many(
        [MemoryRecord(content=f"memory number {i} about topic {i % 5}") for i in range(40)]
    )
    _, matrix = store._load_vectors(store.embedder.signature)
    assert isinstance(matrix[0], array)
    assert "number 17" in store.search("memory number 17 about topic", limit=1)[0].record.content


def test_recall_is_still_correct_after_the_optimization():
    store = LongTermMemory(":memory:", embedder=HashingEmbedder(dims=256))
    store.add_many(
        [MemoryRecord(content=f"memory number {i} about topic {i % 7}") for i in range(60)]
    )
    assert "number 42" in store.search("memory number 42 about topic", limit=1)[0].record.content


# -- 8/9. a hanging task blocked the daemon ---------------------------------


def _daemon(tmp_path, agent, **kwargs):
    from itsbob.daemon.notify import FileSink
    from itsbob.daemon.service import Daemon
    from itsbob.daemon.tasks import TaskStore
    from itsbob.tools import Mode, build_toolbox

    class _NeverNotify:
        def judge(self, **kw):
            return None

    agent.toolbox = build_toolbox(workspace=tmp_path / "ws", mode=Mode.GUARDED, env={})
    return Daemon(
        agent=agent,
        tasks=TaskStore(":memory:"),
        sink=FileSink(path=tmp_path / "n.jsonl"),
        gate=_NeverNotify(),
        home=tmp_path,
        handle_signals=False,
        **kwargs,
    )


class _SlowAgent:
    max_seconds = 180.0

    def __init__(self):
        self.memory = None
        self.brain = None
        self.conversation = None
        self.seen: list[str] = []

    def chat(self, message, **kwargs):
        from itsbob.agent.context import Turn

        self.seen.append(message)
        if "hang" in message:
            time.sleep(30)
        return Turn(message=message, final="done")


def test_a_hanging_task_is_abandoned_not_waited_on(tmp_path):
    daemon = _daemon(tmp_path, _SlowAgent(), task_timeout=1.0)
    now = time.time()
    daemon.tasks.create("hang", "please hang forever", "every 15m", now=now)
    started = time.perf_counter()
    runs = daemon.tick(now=now)
    assert time.perf_counter() - started < 10
    assert runs[0].status == "failed"
    assert "abandoned" in runs[0].output
    assert daemon.describe()["abandoned_runs"] == 1


def test_a_hanging_task_does_not_poison_the_next_one(tmp_path):
    """A shared single worker made one wedged task time out every task after it."""
    daemon = _daemon(tmp_path, _SlowAgent(), task_timeout=1.0)
    now = time.time()
    daemon.tasks.create("hang", "please hang forever", "every 15m", now=now)
    daemon.tasks.create("fine", "quick", "every 15m", now=now)
    daemon.tasks.create("also", "quick too", "every 15m", now=now)
    assert [r.status for r in daemon.tick(now=now)] == ["failed", "ok", "ok"]


def test_an_agent_without_max_seconds_still_runs(tmp_path):
    """The agent is injectable; a stand-in need not carry every field."""

    class Minimal:
        memory = brain = conversation = None

        def chat(self, message, **kwargs):
            from itsbob.agent.context import Turn

            return Turn(message=message, final="ok")

    daemon = _daemon(tmp_path, Minimal(), task_timeout=5.0)
    now = time.time()
    daemon.tasks.create("t", "do it", "every 15m", now=now)
    assert daemon.tick(now=now)[0].status == "ok"


def test_stop_ends_the_loop(tmp_path):
    daemon = _daemon(tmp_path, _SlowAgent())
    daemon.stop()
    daemon.run_forever()  # returns immediately rather than blocking
    assert daemon.started_at is not None


# -- 10. deny patterns were recompiled per call -----------------------------


def test_extra_deny_patterns_are_compiled_once():
    from itsbob.tools.policy import _compiled_extra

    patterns = (("dangerous", "why"),)
    assert _compiled_extra(patterns) is _compiled_extra(patterns)


def test_a_malformed_extra_pattern_does_not_break_the_gate():
    from itsbob.tools.base import Risk, Tool, ToolResult
    from itsbob.tools.policy import Mode, Policy

    policy = Policy(mode=Mode.TRUSTED, extra_deny=(("[unclosed", "bad regex"), ("nukeit", "no")))
    tool = Tool(name="run_shell", description="", run=lambda p, c: ToolResult(True), risk=Risk.EXECUTE)
    assert policy.evaluate(tool, {"command": "ls"}).allowed is True
    assert policy.evaluate(tool, {"command": "nukeit now"}).allowed is False
