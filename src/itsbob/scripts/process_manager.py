"""Listing, starting and stopping background processes — carefully.

Stopping the wrong process is not like deleting the wrong file: there is no
recycle bin, the failure is immediate, and on a desktop it can log you out or
take the network down mid-task. So the guards here are deliberately stricter
than the tools' risk levels alone would imply, and they are checked *before*
anything is signalled rather than trusted to a well-behaved caller.

Nothing may be stopped that is:

* not owned by the current user — no reaching into another account's session;
* PID 1, a kernel thread, or on the protected-name list (the session manager,
  the display server, sshd, the desktop shell) — losing any of these ends the
  session, and no automated cleanup is worth that;
* this process, its parent, or anything in its own process group — an agent
  that can kill its own daemon will eventually do so, usually while tidying up.

Stopping is SIGTERM first with a grace period, then SIGKILL only if asked.
Going straight to SIGKILL denies a process the chance to flush and close, which
is how half-written files happen.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["ProcessInfo", "list_processes", "stop_process", "start_process", "tools"]

try:  # pragma: no cover - presence depends on the install
    import psutil
except ImportError:
    psutil = None

#: Processes whose death ends the session, takes the machine off the network,
#: or locks the screen. Matched on the executable name, case-insensitively.
PROTECTED_NAMES = frozenset({
    "init", "systemd", "systemd-logind", "systemd-journald", "systemd-udevd",
    "systemd-resolved", "dbus-daemon", "dbus-broker", "logind", "login",
    "sshd", "ssh-agent", "gnome-shell", "gnome-session-binary", "plasmashell",
    "Xorg", "Xwayland", "wayland", "gdm", "gdm3", "sddm", "lightdm",
    "NetworkManager", "wpa_supplicant", "polkitd", "pipewire", "pulseaudio",
    "kernel", "kthreadd", "containerd", "dockerd",
})

#: A process that has been running less than this is probably still starting
#: up; killing it tends to leave lock files and half-written state behind.
YOUNG_PROCESS_SECONDS = 2.0


@dataclass
class ProcessInfo:
    pid: int
    name: str
    command: str
    user: str = ""
    cpu_percent: float | None = None
    memory_mb: float | None = None
    age_seconds: float | None = None
    status: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "command": self.command[:300],
            "user": self.user,
            "cpu_percent": self.cpu_percent,
            "memory_mb": round(self.memory_mb, 1) if self.memory_mb else None,
            "age_seconds": round(self.age_seconds) if self.age_seconds else None,
            "status": self.status,
        }

    def render(self) -> str:
        memory = f"{self.memory_mb:6.0f}MB" if self.memory_mb else "        "
        cpu = f"{self.cpu_percent:5.1f}%" if self.cpu_percent is not None else "      "
        return f"{self.pid:>7}  {cpu} {memory}  {self.name:<22} {self.command[:70]}"


# -- reading ---------------------------------------------------------------


def _boot_time() -> float:
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _read_proc(pid: int, clock_ticks: int, boot: float) -> ProcessInfo | None:
    base = Path("/proc") / str(pid)
    try:
        stat = (base / "stat").read_text(encoding="utf-8")
        # The comm field is parenthesised and may itself contain spaces or
        # brackets, so split on the last ')' rather than on whitespace.
        close = stat.rindex(")")
        name = stat[stat.index("(") + 1 : close]
        fields = stat[close + 2 :].split()
        status_char = fields[0]
        starttime = float(fields[19])
        rss_pages = float(fields[21])
    except (OSError, ValueError, IndexError):
        return None

    try:
        raw = (base / "cmdline").read_bytes()
        command = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        command = ""
    # An empty cmdline means a kernel thread; they are shown in brackets.
    if not command:
        command = f"[{name}]"

    try:
        uid = base.stat().st_uid
    except OSError:
        uid = -1

    return ProcessInfo(
        pid=pid,
        name=name,
        command=command,
        user=_username(uid),
        memory_mb=rss_pages * os.sysconf("SC_PAGE_SIZE") / 1_048_576 if rss_pages else None,
        age_seconds=max(0.0, time.time() - (boot + starttime / clock_ticks)) if boot else None,
        status=status_char,
    )


def _username(uid: int) -> str:
    if uid < 0:
        return ""
    try:
        import pwd

        return pwd.getpwuid(uid).pw_name
    except (KeyError, ImportError):  # pragma: no cover - platform dependent
        return str(uid)


def list_processes(
    *,
    match: str | None = None,
    mine_only: bool = True,
    limit: int = 40,
    sort_by: str = "memory",
) -> list[ProcessInfo]:
    """Running processes, biggest first. ``match`` is a regex over name+command."""
    pattern = None
    if match:
        try:
            pattern = re.compile(match, re.IGNORECASE)
        except re.error as exc:
            raise ToolError(f"invalid pattern {match!r}: {exc}") from exc

    me = _username(os.getuid()) if hasattr(os, "getuid") else ""
    clock_ticks = int(os.sysconf("SC_CLK_TCK") or 100)
    boot = _boot_time()

    found: list[ProcessInfo] = []
    proc = Path("/proc")
    if not proc.is_dir():  # pragma: no cover - non-Linux
        raise ToolError("process listing needs /proc (Linux)")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        info = _read_proc(int(entry.name), clock_ticks, boot)
        if info is None:
            continue  # exited between listing and reading; normal, not an error
        if mine_only and me and info.user != me:
            continue
        if pattern and not (pattern.search(info.name) or pattern.search(info.command)):
            continue
        found.append(info)

    key = {
        "memory": lambda p: -(p.memory_mb or 0),
        "age": lambda p: -(p.age_seconds or 0),
        "pid": lambda p: p.pid,
        "name": lambda p: p.name.lower(),
    }.get(sort_by, lambda p: -(p.memory_mb or 0))
    found.sort(key=key)
    return found[:limit]


# -- guards ----------------------------------------------------------------


def _own_process_group() -> set[int]:
    """This process, its parent, and everything in its process group."""
    protected = {os.getpid()}
    try:
        protected.add(os.getppid())
    except OSError:  # pragma: no cover
        pass
    try:
        protected.add(os.getpgrp())
    except (OSError, AttributeError):  # pragma: no cover
        pass
    return protected


def refusal_reason(info: ProcessInfo) -> str | None:
    """Why this process must not be stopped, or None if it may be."""
    if info.pid <= 1:
        return f"PID {info.pid} is the init process — stopping it halts the machine"
    if info.command.startswith("[") and info.command.endswith("]"):
        return f"{info.name} is a kernel thread and cannot be stopped from userspace"
    if info.name.lower() in {n.lower() for n in PROTECTED_NAMES}:
        return (
            f"{info.name} is a session-critical process (display, login, network or "
            "service manager) — stopping it would end your session"
        )
    if info.pid in _own_process_group():
        return (
            f"PID {info.pid} is itsbob itself (or its parent) — stopping it would "
            "kill the process making this request"
        )
    me = _username(os.getuid()) if hasattr(os, "getuid") else ""
    if me and info.user and info.user != me:
        return f"PID {info.pid} belongs to {info.user}, not you"
    return None


def _find(pid: int | None, name: str | None) -> list[ProcessInfo]:
    if pid is not None:
        info = _read_proc(int(pid), int(os.sysconf("SC_CLK_TCK") or 100), _boot_time())
        if info is None:
            raise ToolError(f"no process with PID {pid}")
        return [info]
    if not name:
        raise ToolError("give either a pid or a name")
    matches = list_processes(match=re.escape(name), mine_only=True, limit=20)
    if not matches:
        raise ToolError(f"no process of yours matches {name!r}")
    return matches


# -- acting ----------------------------------------------------------------


def stop_process(
    *,
    pid: int | None = None,
    name: str | None = None,
    force: bool = False,
    grace_seconds: float = 5.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stop one process. SIGTERM, then SIGKILL only if ``force``."""
    candidates = _find(pid, name)
    if pid is None and len(candidates) > 1:
        listing = ", ".join(f"{p.pid} ({p.name})" for p in candidates[:6])
        raise ToolError(
            f"{len(candidates)} processes match {name!r}: {listing}. "
            "Name one by pid — stopping several at once is not something to guess at."
        )

    target = candidates[0]
    refusal = refusal_reason(target)
    if refusal:
        raise ToolError(f"refused: {refusal}")
    if target.age_seconds is not None and target.age_seconds < YOUNG_PROCESS_SECONDS:
        raise ToolError(
            f"{target.name} (PID {target.pid}) started {target.age_seconds:.1f}s ago and may "
            "still be initialising — wait a moment and try again"
        )
    if dry_run:
        return {"pid": target.pid, "name": target.name, "dry_run": True,
                "would_signal": "SIGKILL" if force else "SIGTERM"}

    os.kill(target.pid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if not Path(f"/proc/{target.pid}").exists():
            return {"pid": target.pid, "name": target.name, "signal": "SIGTERM", "stopped": True}
        time.sleep(0.1)

    if not force:
        return {
            "pid": target.pid, "name": target.name, "signal": "SIGTERM", "stopped": False,
            "note": f"still running after {grace_seconds:g}s; pass force=true to SIGKILL it",
        }
    os.kill(target.pid, signal.SIGKILL)
    time.sleep(0.2)
    return {
        "pid": target.pid, "name": target.name, "signal": "SIGKILL",
        "stopped": not Path(f"/proc/{target.pid}").exists(),
    }


def start_process(command: str, ctx: ToolContext, *, log_name: str | None = None) -> dict[str, Any]:
    """Launch a detached background process, with its output captured to a file.

    Detached on purpose: the point is a process that outlives the turn that
    started it. Output goes to a file rather than a pipe because nothing will
    be reading a pipe once this returns, and a full pipe buffer silently blocks
    the child forever.
    """
    if not command.strip():
        raise ToolError("command is empty")

    logs = ctx.workspace.resolve() / ".itsbob" / "processes"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^a-z0-9]+", "-", (log_name or shlex.split(command)[0]).lower()).strip("-")
    log_path = logs / f"{safe or 'process'}-{stamp}.log"

    env = ctx.policy.subprocess_env(ctx.env or os.environ) if ctx.policy else dict(os.environ)
    with log_path.open("wb") as handle:
        popen_kwargs: dict[str, Any] = {
            "cwd": str(ctx.workspace.resolve()),
            "env": env,
            "stdout": handle,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(command, shell=True, **popen_kwargs)
        except OSError as exc:
            raise ToolError(f"could not start: {exc}") from exc

    time.sleep(0.3)  # long enough to catch an immediate failure
    exited = process.poll()
    return {
        "pid": process.pid,
        "command": command,
        "log": str(log_path),
        "running": exited is None,
        "exit_code": exited,
    }


# -- tools -----------------------------------------------------------------


def _list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    found = list_processes(
        match=params.get("match"),
        mine_only=bool(params.get("mine_only", True)),
        limit=int(params.get("limit", 25)),
        sort_by=str(params.get("sort_by", "memory")),
    )
    if not found:
        return ToolResult(ok=True, output="(no matching processes)", data={"processes": []})
    header = f"{'PID':>7}  {'CPU':>5} {'MEM':>8}  {'NAME':<22} COMMAND"
    body = "\n".join(p.render() for p in found)
    return ToolResult(
        ok=True,
        output=f"{header}\n{body}",
        data={"processes": [p.as_dict() for p in found], "count": len(found)},
    )


def _stop(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    result = stop_process(
        pid=int(params["pid"]) if params.get("pid") is not None else None,
        name=params.get("name"),
        force=bool(params.get("force", False)),
        grace_seconds=float(params.get("grace_seconds", 5.0)),
        dry_run=bool(ctx.dry_run),
    )
    if result.get("dry_run"):
        return ToolResult(ok=True, dry_run=True,
                          output=f"would send {result['would_signal']} to "
                                 f"{result['name']} (PID {result['pid']})", data=result)
    stopped = result.get("stopped")
    detail = f"{result['signal']} to {result['name']} (PID {result['pid']})"
    return ToolResult(
        ok=bool(stopped),
        output=f"stopped: {detail}" if stopped else f"{detail} — {result.get('note', 'still running')}",
        error=None if stopped else "process did not stop",
        data=result,
    )


def _start(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    result = start_process(params["command"], ctx, log_name=params.get("name"))
    if result["running"]:
        return ToolResult(
            ok=True,
            output=f"started PID {result['pid']}, output going to {result['log']}",
            data=result,
        )
    return ToolResult(
        ok=False,
        error=f"exited immediately with code {result['exit_code']}",
        output=f"see {result['log']} for why",
        data=result,
    )


def tools() -> list[Tool]:
    return [
        Tool(
            name="list_processes",
            description=(
                "List running processes with memory use, newest command line and age. "
                "Use to find something before stopping it, or to see what is using the machine."
            ),
            run=_list,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "match": {"type": "string", "description": "Regex over name and command line."},
                    "mine_only": {"type": "boolean", "description": "Only your own processes. Default true."},
                    "limit": {"type": "integer"},
                    "sort_by": {"type": "string", "description": "memory (default), age, pid or name."},
                },
            },
        ),
        Tool(
            name="start_process",
            description=(
                "Start a long-running background process that outlives this turn. "
                "Output is captured to a log file whose path is returned. For short "
                "commands whose output you need now, use run_shell instead."
            ),
            run=_start,
            risk=Risk.EXECUTE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "name": {"type": "string", "description": "Label used for the log filename."},
                },
                "required": ["command"],
            },
        ),
        Tool(
            name="stop_process",
            description=(
                "Stop a process by pid, or by name if exactly one of yours matches. "
                "Sends SIGTERM and waits; force=true escalates to SIGKILL. Refuses "
                "system-critical processes, other users' processes, and itsbob itself."
            ),
            run=_stop,
            risk=Risk.DESTRUCTIVE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "pid": {"type": "integer"},
                    "name": {"type": "string", "description": "Only used when pid is not given."},
                    "force": {"type": "boolean", "description": "SIGKILL if SIGTERM does not work."},
                    "grace_seconds": {"type": "number"},
                },
            },
        ),
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="List or stop processes.")
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--match")
    listing.add_argument("--limit", type=int, default=25)
    listing.add_argument("--all-users", action="store_true")
    stopping = sub.add_parser("stop")
    stopping.add_argument("target", help="pid or name")
    stopping.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "list":
        for info in list_processes(match=args.match, mine_only=not args.all_users, limit=args.limit):
            print(info.render())
        return 0
    try:
        pid = int(args.target)
        result = stop_process(pid=pid, force=args.force)
    except ValueError:
        result = stop_process(name=args.target, force=args.force)
    print(result)
    return 0 if result.get("stopped") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
