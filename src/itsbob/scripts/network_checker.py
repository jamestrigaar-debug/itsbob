"""Is the network actually working, and if not, which part isn't?

"No internet" is four different faults with four different fixes, and a checker
that only reports up/down makes you find out which one by hand. So each layer
is tested separately:

* **link** — is any interface up and carrying?
* **route** — can a packet reach an address outside this machine?
* **dns** — do names resolve?
* **reachability** — do real hosts answer, and how fast?

TCP connects, not ICMP ping: raw sockets need root, and plenty of networks drop
ICMP while passing TCP perfectly well — so a ping-based check reports an outage
that isn't there. Connecting to port 443 measures the thing that actually
matters, which is whether a request would work.

Repair deliberately stops at the user/session boundary. Restarting
NetworkManager needs root, and a tool that quietly acquires root to fix a
flaky wifi connection is a much bigger problem than the flaky wifi. Anything
needing privilege is reported as a command for a person to run.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tools.base import Risk, Tool, ToolContext, ToolResult

__all__ = ["NetworkState", "check_network", "tools"]

#: Anycast resolvers and a couple of majors. Spread across operators so one
#: provider having a bad day does not read as "the internet is down".
DEFAULT_TARGETS: tuple[tuple[str, str, int], ...] = (
    ("cloudflare", "1.1.1.1", 443),
    ("google-dns", "8.8.8.8", 53),
    ("quad9", "9.9.9.9", 443),
)
DNS_NAMES = ("cloudflare.com", "google.com")


@dataclass
class Probe:
    label: str
    ok: bool
    latency_ms: float | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms is not None else None,
            "error": self.error,
        }


@dataclass
class NetworkState:
    link_up: bool = False
    interfaces: list[str] = field(default_factory=list)
    dns_ok: bool = False
    probes: list[Probe] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    @property
    def reachable(self) -> bool:
        return any(p.ok for p in self.probes)

    @property
    def online(self) -> bool:
        return self.reachable and self.dns_ok

    @property
    def latency_ms(self) -> float | None:
        """Best latency among the probes that answered."""
        values = [p.latency_ms for p in self.probes if p.ok and p.latency_ms is not None]
        return min(values) if values else None

    @property
    def diagnosis(self) -> str:
        if self.online:
            latency = f"{self.latency_ms:.0f}ms" if self.latency_ms else "unknown latency"
            return f"online ({latency} to the nearest responder)"
        if not self.link_up:
            return "no network interface is up — wifi off, cable out, or the radio is disabled"
        if not self.reachable:
            return (
                "an interface is up but nothing outside this machine answers — "
                "no route, a captive portal, or the connection is not actually associated"
            )
        return (
            "hosts answer by IP but names do not resolve — DNS is broken, "
            "which usually means a bad resolver or a captive portal intercepting port 53"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "online": self.online,
            "link_up": self.link_up,
            "dns_ok": self.dns_ok,
            "reachable": self.reachable,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms else None,
            "diagnosis": self.diagnosis,
            "interfaces": self.interfaces,
            "probes": [p.as_dict() for p in self.probes],
            "at": self.at,
        }

    def render(self) -> str:
        rows = [("online" if self.online else "OFFLINE") + f" — {self.diagnosis}"]
        if self.interfaces:
            rows.append(f"  interfaces up: {', '.join(self.interfaces)}")
        rows.append(f"  dns: {'ok' if self.dns_ok else 'FAILING'}")
        for probe in self.probes:
            timing = f"{probe.latency_ms:6.0f}ms" if probe.latency_ms is not None else "      —"
            rows.append(f"  {'ok' if probe.ok else '!!'} {probe.label:<14}{timing}"
                        + (f"  {probe.error}" if probe.error else ""))
        return "\n".join(rows)


# -- probes ----------------------------------------------------------------


def _interfaces_up() -> list[str]:
    """Interfaces that are up and carrying, excluding loopback."""
    up: list[str] = []
    root = Path("/sys/class/net")
    if not root.is_dir():  # pragma: no cover - non-Linux
        return up
    for entry in sorted(root.iterdir()):
        if entry.name == "lo":
            continue
        try:
            state = (entry / "operstate").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if state == "up":
            up.append(entry.name)
        elif state == "unknown":
            # Tunnels and some virtual devices report "unknown" while working;
            # carrier is the reliable signal there.
            try:
                if (entry / "carrier").read_text(encoding="utf-8").strip() == "1":
                    up.append(entry.name)
            except OSError:
                continue
    return up


def _tcp_probe(label: str, host: str, port: int, timeout: float) -> Probe:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        return Probe(label=label, ok=False, error=exc.strerror or str(exc)[:60])
    return Probe(label=label, ok=True, latency_ms=(time.perf_counter() - started) * 1000)


def _dns_ok(timeout: float) -> bool:
    original = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        for name in DNS_NAMES:
            try:
                socket.getaddrinfo(name, 443, proto=socket.IPPROTO_TCP)
                return True
            except OSError:
                continue
        return False
    finally:
        socket.setdefaulttimeout(original)


def check_network(*, timeout: float = 3.0, targets=DEFAULT_TARGETS) -> NetworkState:
    """One full check of every layer."""
    state = NetworkState(interfaces=_interfaces_up())
    state.link_up = bool(state.interfaces)
    state.probes = [_tcp_probe(label, host, port, timeout) for label, host, port in targets]
    # Only worth asking about DNS if something is reachable at all; otherwise a
    # DNS failure is a symptom of the outage, not an additional fault.
    state.dns_ok = _dns_ok(timeout) if state.reachable else False
    return state


# -- repair ----------------------------------------------------------------


def repair_suggestions(state: NetworkState) -> list[dict[str, Any]]:
    """What to try, in order, and whether itsbob may try it itself."""
    if state.online:
        return []
    steps: list[dict[str, Any]] = []
    has_nmcli = shutil.which("nmcli") is not None

    if not state.link_up:
        if has_nmcli:
            steps.append({
                "what": "turn the wifi radio back on",
                "command": "nmcli radio wifi on",
                "needs_root": False,
            })
            steps.append({
                "what": "reconnect the most recent wifi network",
                "command": "nmcli device connect",
                "needs_root": False,
            })
        steps.append({
            "what": "check whether the interface is hardware- or software-blocked",
            "command": "rfkill list",
            "needs_root": False,
        })
    elif not state.reachable:
        steps.append({
            "what": "re-request a lease and re-associate",
            "command": "nmcli networking off && nmcli networking on" if has_nmcli
                       else "sudo systemctl restart NetworkManager",
            "needs_root": not has_nmcli,
        })
        steps.append({
            "what": "open a browser — a captive portal produces exactly this",
            "command": None,
            "needs_root": False,
        })
    elif not state.dns_ok:
        steps.append({
            "what": "flush and restart the resolver",
            "command": "resolvectl flush-caches",
            "needs_root": False,
        })
        steps.append({
            "what": "restart systemd-resolved",
            "command": "sudo systemctl restart systemd-resolved",
            "needs_root": True,
        })
    return steps


def attempt_repair(state: NetworkState, *, timeout: float = 20.0) -> dict[str, Any]:
    """Run only the unprivileged suggestions, then re-check.

    Privileged steps are returned for a person to run. Escalating to root to fix
    a network blip is a far larger grant than the problem justifies, and a
    process that can restart the network can also take a remote machine off the
    air with no way back in.
    """
    attempted: list[dict[str, Any]] = []
    for step in repair_suggestions(state):
        command = step.get("command")
        if not command or step.get("needs_root"):
            continue
        try:
            done = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )
            attempted.append({
                "command": command,
                "exit_code": done.returncode,
                "output": (done.stdout or done.stderr or "").strip()[:300],
            })
        except (subprocess.SubprocessError, OSError) as exc:
            attempted.append({"command": command, "error": str(exc)[:200]})
        time.sleep(1.5)  # give the stack a moment before re-testing

    after = check_network()
    return {
        "attempted": attempted,
        "recovered": after.online,
        "state": after.as_dict(),
        "needs_a_person": [s for s in repair_suggestions(after) if s.get("needs_root")],
    }


# -- tools -----------------------------------------------------------------


def _check(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    state = check_network(timeout=float(params.get("timeout", 3.0)))
    return ToolResult(ok=state.online, output=state.render(),
                      error=None if state.online else state.diagnosis, data=state.as_dict())


def _repair(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    state = check_network()
    if state.online:
        return ToolResult(ok=True, output="already online — nothing to repair",
                          data=state.as_dict())
    if ctx.dry_run:
        steps = repair_suggestions(state)
        return ToolResult(ok=True, dry_run=True, data={"would_try": steps},
                          output="would try:\n" + "\n".join(f"  {s['what']}: {s['command']}"
                                                            for s in steps if s["command"]))
    result = attempt_repair(state)
    lines = [f"{'recovered' if result['recovered'] else 'still offline'} after "
             f"{len(result['attempted'])} attempt(s)"]
    for step in result["attempted"]:
        lines.append(f"  {step['command']} -> {step.get('exit_code', step.get('error'))}")
    for step in result["needs_a_person"]:
        lines.append(f"  needs a person (root): {step['command']}")
    return ToolResult(ok=result["recovered"], output="\n".join(lines),
                      error=None if result["recovered"] else "network still down", data=result)


def tools() -> list[Tool]:
    return [
        Tool(
            name="check_network",
            description=(
                "Test connectivity layer by layer — interface, route, DNS, latency — "
                "and say which one is broken. Use before anything that needs the network, "
                "and to tell a real outage from a slow provider."
            ),
            run=_check,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {"timeout": {"type": "number", "description": "Per-probe seconds. Default 3."}},
            },
        ),
        Tool(
            name="repair_network",
            description=(
                "Try the unprivileged fixes for a dropped connection (re-enable the radio, "
                "re-associate, flush DNS) and re-test. Anything needing root is reported "
                "for you to run, never attempted."
            ),
            run=_repair,
            risk=Risk.DESTRUCTIVE,
            mutates=True,
            parameters={"type": "object", "properties": {}},
        ),
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Check network connectivity.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    args = parser.parse_args(argv)

    while True:
        state = check_network()
        print(json.dumps(state.as_dict(), indent=2) if args.json else state.render())
        if args.repair and not state.online:
            print(json.dumps(attempt_repair(state), indent=2))
        if not args.watch:
            return 0 if state.online else 1
        time.sleep(max(1.0, args.watch))
        print()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
