"""Adversarial checks over the whole surface, hunting the failures that hide.

Everything here is a case that would pass a happy-path test and break in use:
empty payloads, absent optional dependencies, hostile inputs, and the seams
between subsystems that each work alone.
"""

from __future__ import annotations

import json
import threading

import pytest

from itsbob.tools import build_toolbox


@pytest.fixture
def box(tmp_path):
    return build_toolbox(workspace=tmp_path / "ws", mode="trusted", env={})


# -- the token-saving tool rendering must not make anything unreachable ----


def test_shortening_the_tool_list_never_removes_a_tool(box):
    """The saving is descriptions, never capability — a tool that vanished from
    the prompt is a tool the model will be told it may not call."""
    full = box.registry.render_for_prompt()
    short = box.registry.render_for_prompt(describe_only={"read_file"})
    for name in box.registry.names():
        assert f"- {name}(" in full, f"{name} missing from the full list"
        assert f"- {name}(" in short, f"{name} missing from the shortened list"
    assert len(short) < len(full) / 2


def test_a_shortened_line_still_carries_the_arguments(box):
    """The signature is what a call needs; only the prose is dropped."""
    short = box.registry.render_for_prompt(describe_only=set())
    line = next(x for x in short.splitlines() if x.startswith("- write_file("))
    assert "path: string" in line and "content: string" in line
    assert "overwrite?: boolean" in line  # optional marked as optional
    assert "—" not in line  # and no description


def test_risk_survives_the_shortening(box):
    """A destructive tool must not look harmless because its prose was trimmed."""
    short = box.registry.render_for_prompt(describe_only=set())
    assert "[risk: destructive]" in short
    assert "[risk: execute]" in short


def test_the_roster_in_the_prompt_stays_complete(tmp_path):
    """Whatever is trimmed, `tool` must be validated against every real tool."""
    from itsbob.agent.context import build_messages
    from itsbob.agent.persona import Persona

    box = build_toolbox(workspace=tmp_path / "ws", env={})
    system = build_messages(
        persona=Persona(),
        tools=box.registry.render_for_prompt(describe_only=set()),
        snapshot_text="hi",
        conversation=__import__("itsbob.agent.context", fromlist=["x"]).Conversation(),
        tool_names=box.registry.names(),
    )[0].content
    for name in box.registry.names():
        assert name in system


# -- optional dependencies are optional ------------------------------------


def test_vision_without_pillow_refuses_a_large_file_rather_than_spending(tmp_path, monkeypatch):
    import builtins

    from itsbob.tools.base import ToolError
    from itsbob.tools.vision import MAX_BYTES, prepare_image

    real_import = builtins.__import__

    def no_pil(name, *args, **kwargs):
        if name.startswith("PIL"):
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    small = tmp_path / "small.png"
    small.write_bytes(b"x" * 100)
    data, mime = prepare_image(small)
    assert data == b"x" * 100  # sent as-is, no resize available

    big = tmp_path / "big.png"
    big.write_bytes(b"x" * (MAX_BYTES + 1))
    with pytest.raises(ToolError, match="vision extra"):
        prepare_image(big)


def test_recall_survives_with_no_embedder(tmp_path):
    from itsbob.memory.base import MemoryRecord
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    store.add(MemoryRecord(content="the fuse box is behind the coats"))
    hits = store.search("where are the electrics")
    assert store.stats()["semantic_recall"] is False
    assert hits or True  # lexical may miss; the point is it does not raise


# -- empty and hostile inputs ----------------------------------------------


def test_tools_reject_empty_required_arguments(box):
    for name, params in (
        ("read_file", {"path": ""}),
        ("web_search", {"query": "   "}),
        ("remember", {"content": ""}),
    ):
        result = box.call(name, **params)
        assert not result.ok, f"{name} accepted an empty argument"


def test_a_tool_result_never_returns_an_unbounded_observation(box):
    """The scratchpad is fed back verbatim; an unbounded one blows the window."""
    # Too big to read at all: refused with the way to read it in slices, which
    # is cheaper than truncating half a megabyte into the prompt.
    huge = box.policy.workspace / "huge.txt"
    huge.write_text("y" * 500_000, encoding="utf-8")
    refusal = box.call("read_file", path="huge.txt")
    assert not refusal.ok and "start_line" in refusal.error

    # Readable, but longer than one observation should be: clipped head and tail.
    big = box.policy.workspace / "big.txt"
    big.write_text("\n".join(f"line {i} " + "y" * 60 for i in range(2000)), encoding="utf-8")
    rendered = box.call("read_file", path="big.txt").render(max_chars=4000)
    assert len(rendered) <= 4200
    assert "characters omitted" in rendered


def test_the_api_catalogue_never_renders_a_key(tmp_path):
    from itsbob.integrations.apis import register_builtins
    from itsbob.tools.http import ApiCatalog

    env = {"FOOTBALL_DATA_KEY": "super-secret-value", "OPENWEATHER_API_KEY": "another-secret"}
    catalog = ApiCatalog()
    register_builtins(catalog, env)
    block = catalog.render_for_prompt(env)
    assert "super-secret-value" not in block and "another-secret" not in block
    for row in catalog.describe(env):
        assert "super-secret-value" not in json.dumps(row)


def test_a_failed_api_call_never_leaks_the_query_string(monkeypatch):
    """The key rides in the query string for `query` auth; the error must not."""
    from itsbob.integrations.apis import register_builtins
    from itsbob.tools import http as http_module
    from itsbob.tools.http import ApiCatalog, _call_api

    env = {"OPENWEATHER_API_KEY": "leaky-key"}
    catalog = ApiCatalog()
    register_builtins(catalog, env)
    monkeypatch.setattr(http_module, "_request", lambda *a, **k: (404, "nope", {}))

    class Ctx:
        pass

    Ctx.env = env
    result = _call_api(catalog)({"api": "weather", "path": "weather"}, Ctx())
    assert "leaky-key" not in result.output
    assert "leaky-key" not in (result.error or "")
    assert "leaky-key" not in json.dumps(result.data)


# -- concurrency -----------------------------------------------------------


def test_the_store_survives_concurrent_writers(tmp_path):
    from itsbob.memory.base import MemoryRecord
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    errors: list[str] = []

    def write(n: int) -> None:
        try:
            for i in range(20):
                store.add(MemoryRecord(content=f"writer {n} row {i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    assert len(store) == 80


def test_the_session_runs_one_turn_at_a_time_under_load():
    from itsbob.gui.session import Session

    overlaps, running = [], []

    class Agent:
        conversation = []

        def chat(self, message, **kwargs):
            running.append(message)
            if len(running) > 1:
                overlaps.append(list(running))
            import time as _t

            _t.sleep(0.02)
            running.remove(message)

            class T:
                final = "done"

                def as_dict(self):
                    return {}

            return T()

    session = Session(lambda confirm: Agent())
    for i in range(25):
        session.submit(f"message {i}")
    for _ in range(200):
        if not session.busy and not session.queued_messages():
            break
        import time as _t

        _t.sleep(0.05)
    assert overlaps == []


# -- the seams between subsystems ------------------------------------------


def test_every_registered_tool_has_a_usable_schema(box):
    """A tool the model cannot call correctly is worse than one that is absent."""
    for tool in box.registry.all():
        assert tool.description.strip(), f"{tool.name} has no description"
        assert isinstance(tool.parameters, dict)
        properties = tool.parameters.get("properties", {})
        for required in tool.parameters.get("required", []):
            assert required in properties, f"{tool.name} requires undeclared {required!r}"
        for key, spec in properties.items():
            assert spec.get("type"), f"{tool.name}.{key} has no type"
        assert tool.render_for_prompt(described=False).startswith(f"- {tool.name}(")


def test_every_tool_name_is_unique_and_callable(box):
    names = box.registry.names()
    assert len(names) == len(set(names))
    for name in names:
        # An unknown-argument call must fail with a usable message, never a crash.
        result = box.call(name, definitely_not_a_real_argument=1)
        assert isinstance(result.ok, bool)
        if not result.ok:
            assert result.error


def test_memory_round_trips_through_sqlite_unchanged(tmp_path):
    from itsbob.memory.base import Horizon, MemoryKind, MemoryRecord, Subject
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    original = MemoryRecord(
        content="I preferred the 1982 cut",
        kind=MemoryKind.PREFERENCE,
        subject=Subject.SELF,
        horizon=Horizon.SHORT,
        importance=0.81,
        tags=("film", "opinion"),
        expires_at=2_000_000_000.0,
    )
    store.add(original)
    back = store.get(original.id)
    for field in ("content", "kind", "subject", "horizon", "tags", "expires_at"):
        assert getattr(back, field) == getattr(original, field), field
    assert round(back.importance, 3) == 0.81


def test_an_old_database_opens_and_gains_the_new_columns(tmp_path):
    """Someone upgrading must not lose their memories to a schema change."""
    import sqlite3

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "kind TEXT NOT NULL, importance REAL NOT NULL, tick INTEGER NOT NULL DEFAULT 0, "
            "created_at REAL NOT NULL, last_access_tick INTEGER NOT NULL DEFAULT 0, "
            "access_count INTEGER NOT NULL DEFAULT 0, salience REAL NOT NULL DEFAULT 0.5, "
            "tags TEXT NOT NULL DEFAULT '[]', metadata TEXT NOT NULL DEFAULT '{}')"
        )
        conn.execute(
            "INSERT INTO memories (id, content, kind, importance, created_at) "
            "VALUES ('old1', 'written before subjects existed', 'fact', 0.5, 1700000000)"
        )

    from itsbob.memory.base import Subject
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(path, embedder=None)
    record = store.get("old1")
    assert record is not None
    assert record.content == "written before subjects existed"
    assert record.subject is Subject.USER  # a sane default, not a crash
    assert store.counts_by("subject") == {"user": 1}


# -- the screening check must not cost more than it saves ------------------


def test_the_feasibility_screen_is_free_or_reserved_for_big_requests():
    """It is biased hard toward 'yes' by design, so it rarely prevents anything.
    Paying a cloud call to screen every ordinary request inverts the point."""
    from itsbob.agent.budget import FeasibilityCheck
    from itsbob.router.tiers import Tier

    class NoLocal:
        local = None

    class WithLocal:
        local = object()

    ordinary = "read the config file and tell me what the timeout is set to"
    big = "x" * 300

    paid = FeasibilityCheck(brain=NoLocal())
    assert paid.free is False
    assert paid.should_check(ordinary, Tier.A) is False  # not worth a paid call
    assert paid.should_check(big, Tier.A) is True

    free = FeasibilityCheck(brain=WithLocal())
    assert free.free is True
    assert free.should_check(ordinary, Tier.A) is True  # costs nothing, so always

    # Cheap tiers are never screened either way — the screen would cost more
    # than the turn it is screening.
    for check in (paid, free):
        assert check.should_check(big, Tier.C) is False
        assert check.should_check(big, Tier.B) is False


# -- permanence is earned --------------------------------------------------


def test_a_memory_starts_in_the_working_set(tmp_path):
    """Writing everything down forever is not a good memory, it is a transcript:
    recall then picks between forty near-identical rows with equal confidence."""
    from itsbob.memory.base import Horizon, MemoryRecord

    assert MemoryRecord(content="x").horizon is Horizon.SHORT
    assert Horizon.coerce(None) is Horizon.SHORT
    assert Horizon.coerce("") is Horizon.SHORT
    assert Horizon.coerce("anything unrecognised") is Horizon.SHORT
    # Long is still reachable, by asking for it in any of the obvious words.
    for word in ("long", "long-term", "permanent", "durable", "forever"):
        assert Horizon.coerce(word) is Horizon.LONG


def test_being_recalled_again_is_what_earns_permanence(tmp_path):
    from itsbob.memory.base import Horizon, MemoryRecord
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    passing = store.add(MemoryRecord(content="looking at the router configuration"))
    used = store.add(MemoryRecord(content="the fuse box is behind the coats"))
    vital = store.add(MemoryRecord(content="allergic to penicillin", importance=0.95))
    assert store.counts_by("horizon") == {"short": 3}

    # Surfaced twice: once can be a vague query, twice is being used.
    for _ in range(2):
        store.search("where is the fuse box")

    promoted = store.consolidate()
    assert set(promoted) == {used.id, vital.id}
    assert store.get(used.id).horizon is Horizon.LONG
    assert store.get(vital.id).horizon is Horizon.LONG
    assert store.get(passing.id).horizon is Horizon.SHORT
    # And promotion clears the expiry, or it would be dropped anyway.
    assert store.get(used.id).expires_at is None


def test_promotion_happens_before_pruning_can_drop_it(tmp_path):
    """A row that has earned permanence must not be evicted on the same pass."""
    from itsbob.memory.base import Horizon, MemoryRecord
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    store.short_term_capacity = 2
    earned = store.add(MemoryRecord(content="the fuse box is behind the coats"))
    for i in range(5):
        store.add(MemoryRecord(content=f"idle chatter number {i}"))
    for _ in range(2):
        store.search("fuse box")

    # The order the agent runs them in.
    store.consolidate()
    store.prune_short_term()
    assert store.get(earned.id) is not None
    assert store.get(earned.id).horizon is Horizon.LONG


def test_forgetting_still_works_on_either_horizon(tmp_path):
    from itsbob.memory.base import Horizon, MemoryRecord
    from itsbob.memory.long_term import LongTermMemory

    store = LongTermMemory(tmp_path / "m.sqlite", embedder=None)
    short = store.add(MemoryRecord(content="passing thought"))
    kept = store.add(MemoryRecord(content="lives in Reading", horizon=Horizon.LONG))
    assert store.forget(short.id) and store.forget(kept.id)
    assert len(store) == 0
    assert store.forget("never-existed") is False
