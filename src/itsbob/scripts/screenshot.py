"""Screenshots: what is on the screen right now, as a file it can then look at.

Two tools, matching the two questions actually asked: *what is on this whole
screen* and *what is in the window I'm looking at*.

The design point is what happens next. A screenshot on its own is a PNG the
agent cannot read; paired with ``describe_image`` it becomes "what does that
error dialog say", "did the build finish", "what is this chart showing". So both
tools write into the workspace and return the path in a form the vision tool
takes directly.

Capture is done by whichever native tool is present, tried in order of how
likely they are to work unattended. There is no single portable way to do this:
X11 and Wayland disagree, macOS has its own, and a headless box has none. Rather
than pretend otherwise, an unsupported machine gets a clear refusal naming what
to install — a screenshot that silently produces a black rectangle is worse than
one that does not happen.

Active-window capture is genuinely unavailable on some setups (most Wayland
compositors without a portal). Where it is, the full-screen shot is taken
instead and the result says so, because a full screen is nearly always a useful
answer to "what am I looking at" and an error is never one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["capture", "Capture", "backends", "tools"]

SUMMARY = "Capture the screen or the active window, ready for `describe_image`."

#: A capture may not hang the agent waiting on a compositor prompt.
TIMEOUT = 25.0


@dataclass(frozen=True)
class Backend:
    """One capture command. ``args`` is built with the destination path."""

    name: str
    binary: str
    full: Any
    window: Any = None
    #: Some tools need a display; recorded so the failure message can say why.
    needs_display: bool = True


def _grim_full(path: Path) -> list[str]:
    return ["grim", str(path)]


def _spectacle_full(path: Path) -> list[str]:
    return ["spectacle", "-b", "-n", "-f", "-o", str(path)]


def _spectacle_window(path: Path) -> list[str]:
    return ["spectacle", "-b", "-n", "-a", "-o", str(path)]


def _gnome_full(path: Path) -> list[str]:
    return ["gnome-screenshot", "-f", str(path)]


def _gnome_window(path: Path) -> list[str]:
    return ["gnome-screenshot", "-w", "-f", str(path)]


def _import_full(path: Path) -> list[str]:
    return ["import", "-window", "root", str(path)]


def _scrot_full(path: Path) -> list[str]:
    return ["scrot", "-o", str(path)]


def _scrot_window(path: Path) -> list[str]:
    return ["scrot", "-o", "-u", str(path)]


def _screencapture_full(path: Path) -> list[str]:
    return ["screencapture", "-x", str(path)]


def _screencapture_window(path: Path) -> list[str]:
    # -o drops the window shadow, which is most of the wasted pixels.
    return ["screencapture", "-x", "-o", "-w", str(path)]


#: Tried in order. macOS first because it is unambiguous when present; then the
#: desktop-specific tools, which know about the compositor; then the X11
#: fallbacks, which work anywhere X does.
BACKENDS: tuple[Backend, ...] = (
    Backend("macos", "screencapture", _screencapture_full, _screencapture_window, False),
    Backend("gnome", "gnome-screenshot", _gnome_full, _gnome_window),
    Backend("kde", "spectacle", _spectacle_full, _spectacle_window),
    Backend("wayland", "grim", _grim_full, None),
    Backend("x11-scrot", "scrot", _scrot_full, _scrot_window),
    Backend("x11-import", "import", _import_full, None),
)


def backends(*, window: bool = False) -> list[Backend]:
    """Installed backends that can take the requested kind of shot."""
    return [
        b
        for b in BACKENDS
        if shutil.which(b.binary) and (b.window is not None if window else True)
    ]


def has_display(env: Any = None) -> bool:
    env = os.environ if env is None else env
    import platform

    if platform.system() == "Darwin":
        return True
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


@dataclass
class Capture:
    """One screenshot that was actually taken."""

    path: Path
    backend: str
    window: bool
    bytes: int
    #: Set when an active-window shot fell back to the whole screen.
    note: str = ""

    def render(self) -> str:
        what = "active window" if self.window else "full screen"
        line = f"captured the {what} to {self.path} ({self.bytes / 1024:.0f} KB, via {self.backend})"
        if self.note:
            line += f"\n{self.note}"
        return line + "\nPass this path to `describe_image` to read what is in it."

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "backend": self.backend,
            "window": self.window,
            "bytes": self.bytes,
            "note": self.note,
        }


def _run(command: Sequence[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed binaries, argument list, no shell
            list(command), capture_output=True, text=True, timeout=TIMEOUT, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "").strip()[:200]
    return True, ""


def capture(destination: Path, *, window: bool = False, env: Any = None) -> Capture:
    """Take a screenshot to ``destination``. Raises :class:`ToolError` if it cannot."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not has_display(env):
        raise ToolError(
            "there is no display attached (no DISPLAY or WAYLAND_DISPLAY), so there "
            "is nothing to screenshot"
        )

    problems: list[str] = []
    note = ""
    for kind_window in (True, False) if window else (False,):
        for backend in backends(window=kind_window):
            builder = backend.window if kind_window else backend.full
            ok, error = _run(builder(destination))
            if ok and destination.is_file() and destination.stat().st_size > 0:
                if window and not kind_window:
                    note = (
                        "(active-window capture is not available on this desktop, so "
                        "this is the full screen)"
                    )
                return Capture(
                    path=destination,
                    backend=backend.name,
                    window=kind_window,
                    bytes=destination.stat().st_size,
                    note=note,
                )
            problems.append(f"{backend.binary}: {error or 'produced no file'}")

    if not problems:
        raise ToolError(
            "no screenshot tool is installed. Install one of: grim (Wayland), "
            "gnome-screenshot, spectacle (KDE), scrot or imagemagick (X11). "
            "On macOS `screencapture` is built in."
        )
    raise ToolError("every screenshot tool failed — " + "; ".join(problems)[:400])


def _destination(params: dict[str, Any], ctx: ToolContext, prefix: str) -> Path:
    given = str(params.get("path") or "").strip()
    if given:
        return ctx.resolve(given)
    # Timestamped, inside the workspace, so repeated captures do not overwrite
    # each other and the vision tool can reach them without a policy exception.
    name = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.png"
    return ctx.resolve(Path("screenshots") / name)


def _full(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    shot = capture(_destination(params, ctx, "screen"), window=False, env=ctx.env)
    return ToolResult(ok=True, output=shot.render(), data=shot.as_dict())


def _window(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    shot = capture(_destination(params, ctx, "window"), window=True, env=ctx.env)
    return ToolResult(ok=True, output=shot.render(), data=shot.as_dict())


_PARAMS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Where to save it, relative to the workspace. Optional — defaults "
                "to a timestamped file under screenshots/."
            ),
        }
    },
}


def tools() -> list[Tool]:
    return [
        Tool(
            name="screenshot_full_window",
            description=(
                "Capture the whole screen to a PNG in the workspace. Follow with "
                "`describe_image` on the returned path to read what is on it."
            ),
            run=_full,
            risk=Risk.WRITE,
            mutates=True,
            parameters=_PARAMS,
        ),
        Tool(
            name="screenshot_current_window",
            description=(
                "Capture just the window that currently has focus. Falls back to "
                "the full screen on desktops that do not allow it, and says so."
            ),
            run=_window,
            risk=Risk.WRITE,
            mutates=True,
            parameters=_PARAMS,
        ),
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Take a screenshot.")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--window", action="store_true", help="Active window only.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    destination = Path(
        args.path or f"screenshot-{time.strftime('%Y%m%d-%H%M%S')}.png"
    ).expanduser()
    try:
        shot = capture(destination, window=args.window)
    except ToolError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(shot.as_dict(), indent=2) if args.json else shot.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
