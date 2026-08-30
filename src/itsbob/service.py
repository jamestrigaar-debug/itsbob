"""``itsbob service`` — keeping the daemon running across reboots.

There is deliberately no supervisor here. Every platform this runs on already
has one that is better tested than anything this project would write, handles
restarts and logging, and is what an administrator expects to find. So this
generates a unit file for the one that is present and gets out of the way.

Both generated units carry two settings worth naming, because they are the
difference between "it works on my machine" and "it works at 3am":

* an absolute path to the installed console script, since a service manager
  starts with almost no environment and nothing on ``PATH``;
* ``ITSBOB_HOME`` set explicitly, since ``$HOME`` under a service manager is
  not always the home directory a person would guess.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["install_service", "uninstall_service", "service_status", "unit_text"]

_LABEL = "com.itsbob.daemon"
_SYSTEMD_NAME = "itsbob.service"


def _executable() -> str:
    """The absolute path of the `itsbob` entry point, however it was installed."""
    found = shutil.which("itsbob")
    if found:
        return str(Path(found).resolve())
    # Installed but not on PATH (a venv that is not active): derive it from the
    # interpreter running this code, which is by definition the right one.
    candidate = Path(sys.executable).with_name("itsbob")
    return str(candidate) if candidate.exists() else f"{sys.executable} -m itsbob"


def _systemd_dir() -> Path:
    return Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ).expanduser() / "systemd" / "user"


def _launchd_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_LABEL}.plist"


def unit_text(home: Path, *, mode: str | None = None, kind: str | None = None) -> str:
    """The unit file for this platform, as text. Also what ``--print`` shows."""
    kind = kind or ("launchd" if platform.system() == "Darwin" else "systemd")
    executable = _executable()
    args = ["serve"]
    if mode:
        args += ["--mode", mode]

    if kind == "launchd":
        arg_xml = "\n".join(
            f"        <string>{part}</string>" for part in [*executable.split(), *args]
        )
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{arg_xml}
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ITSBOB_HOME</key><string>{home}</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{home}/daemon.log</string>
    <key>StandardErrorPath</key><string>{home}/daemon.log</string>
    <key>WorkingDirectory</key><string>{home}</string>
</dict>
</plist>
"""

    return f"""[Unit]
Description=itsbob assistant daemon
Documentation=https://github.com/jamestrigaar-debug/itsbob
After=network-online.target

[Service]
Type=simple
ExecStart={executable} {' '.join(args)}
Environment=ITSBOB_HOME={home}
WorkingDirectory={home}
Restart=on-failure
RestartSec=10
# The daemon stops cleanly on SIGTERM and records the run in flight, so give it
# time to finish rather than killing it mid-task.
KillSignal=SIGTERM
TimeoutStopSec=90

[Install]
WantedBy=default.target
"""


def install_service(home: Path, *, mode: str | None = None, start: bool = True) -> tuple[bool, str]:
    """Write and enable the unit. Returns (ok, message)."""
    system = platform.system()
    home = Path(home).expanduser()
    home.mkdir(parents=True, exist_ok=True)

    if system == "Darwin":
        path = _launchd_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit_text(home, mode=mode, kind="launchd"), encoding="utf-8")
        if not start:
            return True, f"wrote {path} (not loaded — `launchctl load {path}` to start)"
        subprocess.run(["launchctl", "unload", str(path)], capture_output=True, check=False)
        result = subprocess.run(["launchctl", "load", str(path)], capture_output=True, text=True)
        if result.returncode:
            return False, f"wrote {path} but launchctl load failed: {result.stderr.strip()}"
        return True, f"installed and started (launchd). Logs: {home}/daemon.log"

    if system != "Linux":
        return False, (
            f"no service integration for {system}. Run `itsbob serve` under whatever "
            "keeps processes alive on this platform."
        )

    if not shutil.which("systemctl"):
        return False, (
            "systemd not found. Run `itsbob serve` under your init system, or in tmux."
        )

    directory = _systemd_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _SYSTEMD_NAME
    path.write_text(unit_text(home, mode=mode, kind="systemd"), encoding="utf-8")

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
    if not start:
        return True, f"wrote {path} (start with: systemctl --user enable --now itsbob)"
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", _SYSTEMD_NAME], capture_output=True, text=True
    )
    if result.returncode:
        return False, f"wrote {path} but enable failed: {result.stderr.strip()}"
    return True, (
        "installed and started (systemd --user).\n"
        "  status: systemctl --user status itsbob\n"
        "  logs:   journalctl --user -u itsbob -f\n"
        "  note:   `loginctl enable-linger $USER` keeps it running when you log out"
    )


def uninstall_service() -> tuple[bool, str]:
    system = platform.system()
    if system == "Darwin":
        path = _launchd_path()
        if not path.exists():
            return True, "nothing installed"
        subprocess.run(["launchctl", "unload", str(path)], capture_output=True, check=False)
        path.unlink()
        return True, f"removed {path}"

    path = _systemd_dir() / _SYSTEMD_NAME
    if not path.exists():
        return True, "nothing installed"
    subprocess.run(["systemctl", "--user", "disable", "--now", _SYSTEMD_NAME],
                   capture_output=True, check=False)
    path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
    return True, f"removed {path}"


def service_status() -> str:
    system = platform.system()
    if system == "Darwin":
        path = _launchd_path()
        if not path.exists():
            return "not installed"
        result = subprocess.run(["launchctl", "list", _LABEL], capture_output=True, text=True)
        return "running" if result.returncode == 0 else f"installed at {path}, not loaded"

    path = _systemd_dir() / _SYSTEMD_NAME
    if not path.exists():
        return "not installed"
    if not shutil.which("systemctl"):
        return f"unit at {path} (systemctl not available)"
    result = subprocess.run(
        ["systemctl", "--user", "is-active", _SYSTEMD_NAME], capture_output=True, text=True
    )
    return f"{result.stdout.strip() or 'unknown'} ({path})"
