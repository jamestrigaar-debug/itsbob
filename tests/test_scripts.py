"""The foundation scripts, and the guards that stop them doing damage.

The guards get more attention than the features: two of these tools delete
files and one kills processes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from itsbob.scripts import describe_scripts, script_tools
from itsbob.tools.base import Risk, ToolContext, ToolError


# -- registration ----------------------------------------------------------


def test_every_script_registers_its_tools():
    names = {tool.name for tool in script_tools()}
    assert {"system_status", "check_network", "list_processes", "scan_junk",
            "list_scheduled_tasks"} <= names


def test_risk_is_assigned_by_what_a_call_can_destroy():
    by_name = {tool.name: tool.risk for tool in script_tools()}
    assert by_name["system_status"] is Risk.READ
    assert by_name["scan_junk"] is Risk.READ
    assert by_name["organize_folder"] is Risk.WRITE
    assert by_name["start_process"] is Risk.EXECUTE
    for destructive in ("stop_process", "clean_junk", "remove_task", "repair_network"):
        assert by_name[destructive] is Risk.DESTRUCTIVE, destructive


def test_there_is_no_tool_that_runs_a_task():
    """A task's instruction is run *by the agent*, so a tool that triggered one
    would let a turn start another turn — and the first thing an agent does
    with that is schedule a task that runs itself."""
    assert not [t for t in script_tools() if "run" in t.name and "task" in t.name]


def test_scripts_are_described_for_the_interface():
    rows = describe_scripts()
    assert {r["name"] for r in rows} >= {"system_monitor", "file_cleaner"}
    assert all(r["summary"] and r["tools"] for r in rows)


# -- system monitor --------------------------------------------------------


def test_a_reading_is_taken_and_judged():
    from itsbob.scripts.system_monitor import read_system

    state = read_system(sample_seconds=0.05)
    assert state.verdict in ("ok", "warn", "critical")
    assert isinstance(state.safe_for_heavy_work, bool)
    assert state.cpu_count >= 1


def test_missing_sensors_are_absent_not_wrong():
    """Plenty of machines have no battery and no thermal zone."""
    from itsbob.scripts.system_monitor import read_system

    state = read_system(sample_seconds=0.05)
    for field in (state.battery_percent, state.temperature_c):
        assert field is None or field >= 0


def test_thresholds_turn_a_reading_into_a_verdict():
    from itsbob.scripts.system_monitor import Thresholds, read_system

    impossible = Thresholds(cpu_percent=-1, memory_percent=-1, disk_percent=-1)
    state = read_system(impossible, sample_seconds=0.05)
    assert state.concerns and state.safe_for_heavy_work is False
    assert "critical" == state.verdict


def test_a_reading_renders_without_optional_fields():
    from itsbob.scripts.system_monitor import SystemState

    assert "verdict" in SystemState().render()


# -- process manager -------------------------------------------------------


def test_processes_are_listed_with_their_command():
    from itsbob.scripts.process_manager import list_processes

    found = list_processes(limit=5)
    assert found and all(p.pid > 0 and p.command for p in found)


def test_an_invalid_match_pattern_is_an_error_not_a_crash():
    from itsbob.scripts.process_manager import list_processes

    with pytest.raises(ToolError, match="invalid pattern"):
        list_processes(match="[unclosed")


def test_init_can_never_be_stopped():
    from itsbob.scripts.process_manager import stop_process

    with pytest.raises(ToolError, match="init process"):
        stop_process(pid=1, dry_run=True)


def test_itsbob_cannot_stop_itself():
    """An agent that can kill its own daemon eventually will, while tidying up."""
    from itsbob.scripts.process_manager import stop_process

    with pytest.raises(ToolError, match="itsbob itself"):
        stop_process(pid=os.getpid(), dry_run=True)


def test_a_protected_process_is_refused_by_name():
    from itsbob.scripts.process_manager import ProcessInfo, refusal_reason

    info = ProcessInfo(pid=4242, name="systemd", command="/lib/systemd/systemd",
                       user=os.environ.get("USER", ""), age_seconds=99999)
    assert "session-critical" in (refusal_reason(info) or "")


def test_a_kernel_thread_is_refused():
    from itsbob.scripts.process_manager import ProcessInfo, refusal_reason

    info = ProcessInfo(pid=2, name="kthreadd", command="[kthreadd]", age_seconds=99999)
    assert "kernel thread" in (refusal_reason(info) or "")


def test_another_users_process_is_refused():
    from itsbob.scripts.process_manager import ProcessInfo, refusal_reason

    info = ProcessInfo(pid=4243, name="thing", command="/bin/thing",
                       user="somebody-else", age_seconds=99999)
    assert "belongs to" in (refusal_reason(info) or "")


def test_a_missing_pid_is_reported_plainly():
    from itsbob.scripts.process_manager import stop_process

    with pytest.raises(ToolError, match="no process with PID"):
        stop_process(pid=999_999, dry_run=True)


# -- file cleaner ----------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "sub" / "__pycache__").mkdir(parents=True)
    old = time.time() - 30 * 86400
    for name in ("keep.md", "app.log", "scratch.tmp", "sub/__pycache__/x.pyc", "sub/keep.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * 500)
        os.utime(path, (old, old))
    os.utime(root / "sub" / "__pycache__", (old, old))
    return root


def _ctx(workspace):
    return ToolContext(workspace=workspace)


def test_only_junk_is_found(workspace):
    from itsbob.scripts.file_cleaner import scan

    found = {Path(f.path).name for f in scan([workspace], ctx=_ctx(workspace)).found}
    assert "app.log" in found and "scratch.tmp" in found and "__pycache__" in found
    assert "keep.md" not in found and "keep.py" not in found


def test_recent_files_are_left_alone(workspace):
    from itsbob.scripts.file_cleaner import scan

    fresh = workspace / "today.log"
    fresh.write_text("new")
    found = {Path(f.path).name for f in scan([workspace], ctx=_ctx(workspace)).found}
    assert "today.log" not in found


@pytest.mark.parametrize("target", ["/", "/etc", "/usr/lib", "/var", "/boot"])
def test_system_directories_are_refused(workspace, target):
    from itsbob.scripts.file_cleaner import scan

    with pytest.raises(ToolError, match="refused"):
        scan([target], ctx=_ctx(workspace))


def test_the_home_directory_is_refused_wholesale(workspace, monkeypatch):
    from itsbob.scripts.file_cleaner import scan

    with pytest.raises(ToolError, match="refused"):
        scan([str(Path.home())], ctx=_ctx(workspace))


def test_itsbobs_own_state_is_refused(tmp_path, monkeypatch):
    """memory.sqlite living one directory up is not a reason to lose your memory."""
    from itsbob.scripts.file_cleaner import refuse_reason

    home = tmp_path / "itsbob-home"
    home.mkdir()
    monkeypatch.setenv("ITSBOB_HOME", str(home))
    assert "itsbob's own state" in (refuse_reason(home, [tmp_path]) or "")
    assert "itsbob's own state" in (refuse_reason(home / "memory.sqlite", [tmp_path]) or "")
    assert "contains itsbob's state" in (refuse_reason(tmp_path, [tmp_path]) or "")


def test_the_workspace_inside_the_home_is_still_cleanable(tmp_path, monkeypatch):
    """The workspace lives inside ITSBOB_HOME by default. Protecting the whole
    home refused the one directory cleaning is actually for."""
    from itsbob.scripts.file_cleaner import refuse_reason

    home = tmp_path / "itsbob-home"
    workspace = home / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("ITSBOB_HOME", str(home))
    assert refuse_reason(workspace, [workspace]) is None
    assert refuse_reason(workspace / "logs", [workspace]) is None
    # …while its siblings are still protected.
    assert refuse_reason(home / "memory.sqlite", [workspace]) is not None


def test_a_path_outside_the_roots_is_refused(workspace, tmp_path):
    from itsbob.scripts.file_cleaner import scan

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ToolError, match="outside the cleanable roots"):
        scan([outside], ctx=_ctx(workspace))


def test_extra_roots_come_from_the_environment(tmp_path, monkeypatch):
    from itsbob.scripts.file_cleaner import allowed_roots

    extra = tmp_path / "downloads"
    extra.mkdir()
    monkeypatch.setenv("ITSBOB_CLEAN_ROOTS", str(extra))
    assert extra.resolve() in allowed_roots(None)


def test_cleaning_is_a_dry_run_by_default(workspace):
    from itsbob.scripts.file_cleaner import clean

    report = clean([workspace], ctx=_ctx(workspace))
    assert report.dry_run and report.found
    assert (workspace / "app.log").exists(), "a dry run must not delete anything"


def test_cleaning_for_real_removes_only_junk(workspace):
    from itsbob.scripts.file_cleaner import clean

    report = clean([workspace], ctx=_ctx(workspace), dry_run=False)
    assert report.bytes_freed > 0 and not report.failed
    assert not (workspace / "app.log").exists()
    assert not (workspace / "sub" / "__pycache__").exists()
    assert (workspace / "keep.md").exists()
    assert (workspace / "sub" / "keep.py").exists()


def test_an_unknown_category_is_rejected(workspace):
    from itsbob.scripts.file_cleaner import scan

    with pytest.raises(ToolError, match="unknown categories"):
        scan([workspace], ctx=_ctx(workspace), categories=["nonsense"])


def test_organize_moves_and_never_overwrites(tmp_path):
    from itsbob.scripts.file_cleaner import organize

    root = tmp_path / "downloads"
    (root / "images").mkdir(parents=True)
    (root / "a.png").write_text("one")
    (root / "images" / "a.png").write_text("already here")
    (root / "notes.pdf").write_text("doc")

    result = organize(root, ctx=ToolContext(workspace=root), dry_run=False)
    assert result["moved"] == 2
    assert (root / "images" / "a.png").read_text() == "already here"
    assert (root / "images" / "a (1).png").read_text() == "one"
    assert (root / "documents" / "notes.pdf").exists()


def test_organize_is_a_dry_run_by_default(tmp_path):
    from itsbob.scripts.file_cleaner import organize

    root = tmp_path / "downloads"
    root.mkdir()
    (root / "a.png").write_text("x")
    assert organize(root, ctx=ToolContext(workspace=root))["moved"] == 1
    assert (root / "a.png").exists()


# -- network checker -------------------------------------------------------


def test_a_state_diagnoses_which_layer_is_broken():
    from itsbob.scripts.network_checker import NetworkState, Probe

    down = NetworkState(link_up=False)
    assert "no network interface" in down.diagnosis

    no_route = NetworkState(link_up=True, interfaces=["wlan0"],
                            probes=[Probe("x", ok=False, error="timed out")])
    assert "nothing outside this machine answers" in no_route.diagnosis

    dns_broken = NetworkState(link_up=True, interfaces=["wlan0"], dns_ok=False,
                              probes=[Probe("x", ok=True, latency_ms=12.0)])
    assert "DNS is broken" in dns_broken.diagnosis

    fine = NetworkState(link_up=True, interfaces=["wlan0"], dns_ok=True,
                        probes=[Probe("x", ok=True, latency_ms=12.0)])
    assert fine.online and "online" in fine.diagnosis


def test_latency_is_the_best_responder_not_the_first():
    from itsbob.scripts.network_checker import NetworkState, Probe

    state = NetworkState(probes=[Probe("slow", ok=True, latency_ms=300.0),
                                 Probe("fast", ok=True, latency_ms=11.0),
                                 Probe("dead", ok=False)])
    assert state.latency_ms == 11.0


def test_repair_never_suggests_running_root_commands_itself():
    from itsbob.scripts.network_checker import NetworkState, Probe, repair_suggestions

    broken = NetworkState(link_up=True, interfaces=["wlan0"], dns_ok=False,
                          probes=[Probe("x", ok=True, latency_ms=9.0)])
    steps = repair_suggestions(broken)
    assert steps
    assert any(s["needs_root"] for s in steps), "root steps must still be reported"


def test_nothing_to_repair_when_online():
    from itsbob.scripts.network_checker import NetworkState, Probe, repair_suggestions

    fine = NetworkState(link_up=True, interfaces=["e"], dns_ok=True,
                        probes=[Probe("x", ok=True, latency_ms=5.0)])
    assert repair_suggestions(fine) == []


# -- scheduler -------------------------------------------------------------


def _scheduler_ctx(tmp_path, store):
    return ToolContext(workspace=tmp_path, extras={"task_store": store})


def test_scheduling_reads_and_writes_the_daemons_own_store(tmp_path):
    from itsbob.daemon.tasks import TaskStore
    from itsbob.scripts.scheduler import tools

    store = TaskStore(":memory:")
    by_name = {t.name: t for t in tools()}
    ctx = _scheduler_ctx(tmp_path, store)

    result = by_name["schedule_task"].run(
        {"name": "tidy", "instruction": "clean the workspace", "schedule": "every 30m"}, ctx
    )
    assert result.ok
    assert [t.name for t in store.all()] == ["tidy"]
    assert "tidy" in by_name["list_scheduled_tasks"].run({}, ctx).output


def test_a_duplicate_name_is_refused_rather_than_replacing(tmp_path):
    from itsbob.daemon.tasks import TaskStore
    from itsbob.scripts.scheduler import tools

    store = TaskStore(":memory:")
    store.create("tidy", "old instruction", "every 30m")
    schedule = {t.name: t for t in tools()}["schedule_task"]
    with pytest.raises(ToolError, match="already exists"):
        schedule.run({"name": "tidy", "instruction": "new", "schedule": "every 1h"},
                     _scheduler_ctx(tmp_path, store))
    assert store.find("tidy").prompt == "old instruction"


def test_a_bad_schedule_is_rejected_before_storing(tmp_path):
    from itsbob.daemon.tasks import TaskStore
    from itsbob.scripts.scheduler import tools

    store = TaskStore(":memory:")
    schedule = {t.name: t for t in tools()}["schedule_task"]
    with pytest.raises(ToolError, match="could not parse"):
        schedule.run({"name": "x", "instruction": "y", "schedule": "sometimes"},
                     _scheduler_ctx(tmp_path, store))
    assert len(store) == 0


def test_pausing_keeps_the_task_and_its_history(tmp_path):
    from itsbob.daemon.tasks import TaskStore
    from itsbob.scripts.scheduler import tools

    store = TaskStore(":memory:")
    store.create("nightly", "do it", "daily at 02:00")
    pause = {t.name: t for t in tools()}["pause_task"]
    pause.run({"name": "nightly", "enabled": False}, _scheduler_ctx(tmp_path, store))
    assert store.find("nightly").enabled is False
    pause.run({"name": "nightly", "enabled": True}, _scheduler_ctx(tmp_path, store))
    assert store.find("nightly").enabled is True
