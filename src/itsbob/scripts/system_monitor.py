"""Is this machine in a fit state to be given work?

The question the daemon actually needs answered is not "what is the CPU at" but
"should I start something expensive right now" — on a laptop that is on battery,
thermally throttled, or nearly out of disk, the honest answer is no. So the
headline of every reading is :attr:`SystemState.safe_for_heavy_work`, and the
numbers are there to explain it.

Everything is read straight from ``/proc`` and ``/sys`` on Linux, with
``psutil`` used when it is installed for the parts that are genuinely awkward
to do portably. No hard dependency: a monitor that cannot run because a library
is missing is worse than one that reports a little less.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tools.base import Risk, Tool, ToolContext, ToolResult

__all__ = ["SystemState", "Thresholds", "read_system", "tools"]

try:  # pragma: no cover - presence depends on the install
    import psutil
except ImportError:
    psutil = None


@dataclass(frozen=True)
class Thresholds:
    """Where "fine" stops. Tuned for a laptop that is also being used by a person."""

    cpu_percent: float = 85.0
    memory_percent: float = 88.0
    disk_percent: float = 90.0
    #: Below this on battery, heavy work waits. 25% leaves room to finish and save.
    battery_percent: float = 25.0
    temperature_c: float = 80.0
    #: Load average per core above which the machine is already busy.
    load_per_core: float = 2.0

    def as_dict(self) -> dict[str, float]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "battery_percent": self.battery_percent,
            "temperature_c": self.temperature_c,
            "load_per_core": self.load_per_core,
        }


@dataclass
class SystemState:
    """One reading, plus what it means."""

    cpu_percent: float | None = None
    memory_percent: float | None = None
    memory_available_mb: float | None = None
    disk_percent: float | None = None
    disk_free_gb: float | None = None
    battery_percent: float | None = None
    on_battery: bool | None = None
    temperature_c: float | None = None
    load_average: tuple[float, float, float] | None = None
    cpu_count: int = 1
    uptime_hours: float | None = None
    at: float = field(default_factory=time.time)
    #: Human-readable reasons the machine is not fit for heavy work.
    concerns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def safe_for_heavy_work(self) -> bool:
        return not self.concerns

    @property
    def verdict(self) -> str:
        if self.concerns:
            return "critical"
        return "warn" if self.warnings else "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "safe_for_heavy_work": self.safe_for_heavy_work,
            "concerns": self.concerns,
            "warnings": self.warnings,
            "cpu_percent": _round(self.cpu_percent),
            "memory_percent": _round(self.memory_percent),
            "memory_available_mb": _round(self.memory_available_mb),
            "disk_percent": _round(self.disk_percent),
            "disk_free_gb": _round(self.disk_free_gb, 2),
            "battery_percent": _round(self.battery_percent),
            "on_battery": self.on_battery,
            "temperature_c": _round(self.temperature_c),
            "load_average": [_round(v, 2) for v in self.load_average] if self.load_average else None,
            "cpu_count": self.cpu_count,
            "uptime_hours": _round(self.uptime_hours, 1),
            "at": self.at,
        }

    def render(self) -> str:
        rows = [f"verdict: {self.verdict.upper()}"
                + ("" if self.safe_for_heavy_work else "  — hold off on heavy work")]
        for concern in self.concerns:
            rows.append(f"  !! {concern}")
        for warning in self.warnings:
            rows.append(f"  ·  {warning}")

        def line(label: str, value: Any, suffix: str = "") -> None:
            if value is not None:
                rows.append(f"  {label:<12} {value}{suffix}")

        line("cpu", _round(self.cpu_percent), "%")
        if self.load_average:
            rows.append(
                f"  {'load':<12} {self.load_average[0]:.2f} / {self.load_average[1]:.2f}"
                f" / {self.load_average[2]:.2f}  ({self.cpu_count} cores)"
            )
        if self.memory_percent is not None:
            rows.append(
                f"  {'memory':<12} {self.memory_percent:.0f}% used"
                + (f", {self.memory_available_mb:.0f}MB free" if self.memory_available_mb else "")
            )
        if self.disk_percent is not None:
            rows.append(
                f"  {'disk':<12} {self.disk_percent:.0f}% used"
                + (f", {self.disk_free_gb:.1f}GB free" if self.disk_free_gb else "")
            )
        if self.battery_percent is not None:
            source = "on battery" if self.on_battery else "plugged in"
            rows.append(f"  {'battery':<12} {self.battery_percent:.0f}% ({source})")
        line("temperature", _round(self.temperature_c), "°C")
        line("uptime", _round(self.uptime_hours, 1), "h")
        return "\n".join(rows)


def _round(value: float | None, places: int = 0) -> float | None:
    if value is None:
        return None
    return round(float(value), places) if places else round(float(value))


# -- readers ---------------------------------------------------------------


def _cpu_percent(sample_seconds: float = 0.3) -> float | None:
    """Busy percentage across a short sample.

    Sampled rather than instantaneous: /proc/stat holds totals since boot, so a
    single read reports the average since the machine started, which on a
    laptop that has been up for days is always a comfortable-looking number and
    tells you nothing about now.
    """
    if psutil is not None:  # pragma: no cover - depends on the install
        return float(psutil.cpu_percent(interval=sample_seconds))
    first = _proc_stat_totals()
    if first is None:
        return None
    time.sleep(sample_seconds)
    second = _proc_stat_totals()
    if second is None:
        return None
    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _proc_stat_totals() -> tuple[float, float] | None:
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8").split("\n", 1)[0]
    except OSError:
        return None
    parts = line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return None
    try:
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    # idle + iowait are fields 4 and 5.
    idle = values[3] + (values[4] if len(values) > 4 else 0.0)
    return idle, sum(values)


def _memory() -> tuple[float | None, float | None]:
    if psutil is not None:  # pragma: no cover
        vm = psutil.virtual_memory()
        return float(vm.percent), vm.available / 1_048_576
    try:
        fields = {}
        for raw in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = raw.partition(":")
            fields[key.strip()] = float(rest.strip().split()[0])  # kB
    except (OSError, ValueError, IndexError):
        return None, None
    total = fields.get("MemTotal")
    # MemAvailable, not MemFree: free excludes reclaimable cache and so
    # under-reports what a program could actually get by a large margin.
    available = fields.get("MemAvailable", fields.get("MemFree"))
    if not total or available is None:
        return None, None
    return 100.0 * (1.0 - available / total), available / 1024


def _disk(path: str | Path = "/") -> tuple[float | None, float | None]:
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        return None, None
    if usage.total <= 0:
        return None, None
    return 100.0 * usage.used / usage.total, usage.free / 1_073_741_824


def _battery() -> tuple[float | None, bool | None]:
    if psutil is not None and hasattr(psutil, "sensors_battery"):  # pragma: no cover
        battery = psutil.sensors_battery()
        if battery is not None:
            return float(battery.percent), not battery.power_plugged
    root = Path("/sys/class/power_supply")
    if not root.is_dir():
        return None, None
    percent: float | None = None
    on_battery: bool | None = None
    for entry in sorted(root.iterdir()):
        try:
            kind = (entry / "type").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if kind == "Battery" and percent is None:
            for name in ("capacity", "charge_now"):
                try:
                    percent = float((entry / name).read_text(encoding="utf-8").strip())
                    break
                except (OSError, ValueError):
                    continue
            try:
                status = (entry / "status").read_text(encoding="utf-8").strip().lower()
                on_battery = status == "discharging"
            except OSError:
                pass
        elif kind == "Mains":
            try:
                online = (entry / "online").read_text(encoding="utf-8").strip()
                on_battery = online == "0"
            except OSError:
                continue
    return percent, on_battery


def _temperature() -> float | None:
    """Hottest sensor, which is the one that will throttle first."""
    if psutil is not None and hasattr(psutil, "sensors_temperatures"):  # pragma: no cover
        readings = psutil.sensors_temperatures() or {}
        values = [
            entry.current
            for group in readings.values()
            for entry in group
            if entry.current and 0 < entry.current < 150
        ]
        if values:
            return float(max(values))
    hottest: float | None = None
    root = Path("/sys/class/thermal")
    if root.is_dir():
        for zone in sorted(root.glob("thermal_zone*")):
            try:
                milli = float((zone / "temp").read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            celsius = milli / 1000.0
            # Sensors report in millidegrees, but not all of them; and a few
            # report obvious nonsense when unpopulated.
            if not 0 < celsius < 150:
                continue
            hottest = celsius if hottest is None else max(hottest, celsius)
    return hottest


def _uptime_hours() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0]) / 3600
    except (OSError, ValueError, IndexError):
        return None


def read_system(
    thresholds: Thresholds | None = None,
    *,
    disk_path: str | Path = "/",
    sample_seconds: float = 0.3,
) -> SystemState:
    """Take one reading and judge it."""
    limits = thresholds or Thresholds()
    memory_percent, memory_free = _memory()
    disk_percent, disk_free = _disk(disk_path)
    battery_percent, on_battery = _battery()

    try:
        load = os.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        load = None

    state = SystemState(
        cpu_percent=_cpu_percent(sample_seconds),
        memory_percent=memory_percent,
        memory_available_mb=memory_free,
        disk_percent=disk_percent,
        disk_free_gb=disk_free,
        battery_percent=battery_percent,
        on_battery=on_battery,
        temperature_c=_temperature(),
        load_average=load,
        cpu_count=os.cpu_count() or 1,
        uptime_hours=_uptime_hours(),
    )

    # Concerns block heavy work; warnings are worth saying but not acting on.
    if state.temperature_c is not None and state.temperature_c >= limits.temperature_c:
        state.concerns.append(
            f"running hot at {state.temperature_c:.0f}°C (limit {limits.temperature_c:.0f}°C) — "
            "more load will throttle it, not speed it up"
        )
    if state.on_battery and state.battery_percent is not None:
        if state.battery_percent <= limits.battery_percent:
            state.concerns.append(
                f"on battery at {state.battery_percent:.0f}% (limit {limits.battery_percent:.0f}%) "
                "— not enough headroom to finish long work"
            )
        else:
            state.warnings.append(f"on battery ({state.battery_percent:.0f}%)")
    if state.disk_percent is not None and state.disk_percent >= limits.disk_percent:
        state.concerns.append(
            f"disk {state.disk_percent:.0f}% full"
            + (f", {state.disk_free_gb:.1f}GB left" if state.disk_free_gb else "")
        )
    if state.memory_percent is not None and state.memory_percent >= limits.memory_percent:
        state.concerns.append(f"memory {state.memory_percent:.0f}% used")
    if state.cpu_percent is not None and state.cpu_percent >= limits.cpu_percent:
        state.warnings.append(f"cpu busy at {state.cpu_percent:.0f}%")
    if state.load_average and state.cpu_count:
        per_core = state.load_average[0] / state.cpu_count
        if per_core >= limits.load_per_core:
            state.warnings.append(f"load {per_core:.1f} per core — already busy")
    return state


# -- tools -----------------------------------------------------------------


def _check(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    limits = Thresholds()
    overrides = {k: float(v) for k, v in (params.get("thresholds") or {}).items()
                 if k in limits.as_dict()}
    if overrides:
        limits = Thresholds(**{**limits.as_dict(), **overrides})
    state = read_system(limits, disk_path=params.get("disk_path") or "/")
    return ToolResult(
        ok=True,
        output=state.render(),
        data={**state.as_dict(), "thresholds": limits.as_dict(), "psutil": psutil is not None},
    )


def tools() -> list[Tool]:
    return [
        Tool(
            name="system_status",
            description=(
                "CPU, memory, disk, battery and temperature, with a verdict on whether "
                "this machine is in a fit state for heavy work. Check this before "
                "starting anything long-running or expensive."
            ),
            run=_check,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "disk_path": {"type": "string", "description": "Filesystem to report on. Default '/'."},
                    "thresholds": {
                        "type": "object",
                        "description": (
                            "Override any of cpu_percent, memory_percent, disk_percent, "
                            "battery_percent, temperature_c, load_per_core."
                        ),
                    },
                },
            },
        )
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Report system health.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--disk", default="/")
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    args = parser.parse_args(argv)

    while True:
        state = read_system(disk_path=args.disk)
        print(json.dumps(state.as_dict(), indent=2) if args.json else state.render())
        if not args.watch:
            return 0 if state.safe_for_heavy_work else 1
        time.sleep(max(1.0, args.watch))
        print()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
