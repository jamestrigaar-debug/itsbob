"""Looking at the screen, and at pictures — capture and vision, joined up.

:mod:`itsbob.scripts.screenshot` can take a picture of the screen and
:mod:`itsbob.tools.vision` can read a picture. Separately they are two tools and
a filename passed between them, which works and is what the agent did before
this module existed. It is worth joining up anyway, for two reasons.

**A round trip costs a step.** Capture, read the path out of the result, call
the vision tool with it, wait for the model: two full model calls to answer
"what's on my screen?". Here it is one tool call, one model call, and the answer
comes back as words. On a five-rung ladder where every step is billed, halving
the steps for the most common visual question is worth a module.

**The intermediate file is an implementation detail.** "What does that dialog
say" is not a question about a PNG. So captures land in a dated folder under the
workspace and are cleaned up behind you by default — the file is kept only when
you ask for it, and the path is always reported so it can be looked at again.

Three tools:

``look_at_screen``
    Capture the whole screen and answer a question about it.
``look_at_window``
    The same for the focused window, falling back to the full screen where the
    compositor forbids window capture — a full screen is nearly always a useful
    answer to "what am I looking at" and an error never is.
``look_at_image``
    A picture already on disk. The same path as the two above, so the answer
    reads the same whether it came from the screen or from a file.

All three need ``GOOGLE_API_KEY`` for the vision model, and say so plainly when
it is missing rather than capturing an image nobody can read.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult
from ..tools.vision import describe_image, prepare_image, vision_models
from .screenshot import capture

__all__ = ["look", "read_image", "tools"]

SUMMARY = "Look at the screen or at a picture, and say what is on it."

DEFAULT_QUESTION = (
    "Describe what is on this screen. Name the applications and windows you can "
    "see and what each is showing. If there is text — an error, a dialog, a "
    "terminal, a document — transcribe the parts that matter accurately. Be "
    "concrete and do not speculate about anything you cannot actually see."
)


@dataclass
class Sight:
    """One look, and what it cost to take."""

    answer: str
    path: Path
    model: str
    backend: str = ""
    kept: bool = False
    bytes_sent: int = 0
    note: str = ""

    def render(self) -> str:
        lines = [self.answer]
        trail = [f"model {self.model}", f"{self.bytes_sent / 1024:.0f} KB sent"]
        if self.backend:
            trail.insert(0, f"captured with {self.backend}")
        lines.append("")
        lines.append(
            f"({', '.join(trail)}; "
            + (f"kept at {self.path}" if self.kept else "the image was discarded")
            + ")"
        )
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "path": str(self.path) if self.kept else None,
            "model": self.model,
            "backend": self.backend,
            "kept": self.kept,
            "bytes_sent": self.bytes_sent,
            "note": self.note,
        }


def _api_key(ctx: ToolContext) -> str:
    env = ctx.env if ctx.env is not None else os.environ
    key = str(env.get("GOOGLE_API_KEY", "")).strip()
    if not key:
        raise ToolError(
            "GOOGLE_API_KEY is not set, so there is no vision model to read the "
            "image with. Add it to ~/.itsbob/.env and restart — "
            "`screenshot_full_window` still works on its own if you only need the file."
        )
    return key


def _ask(path: Path, question: str, ctx: ToolContext) -> tuple[str, str, int]:
    """Send one image to the vision model. Returns (answer, model, bytes sent)."""
    key = _api_key(ctx)
    policy = getattr(ctx, "policy", None)
    if policy is not None:
        host_reason = policy.check_url("https://generativelanguage.googleapis.com/")
        if host_reason:
            raise ToolError(host_reason)
    data, mime = prepare_image(path)
    env = ctx.env if ctx.env is not None else os.environ
    answer, model = describe_image(
        data=data, mime=mime, prompt=question, api_key=key, models=vision_models(env)
    )
    return answer, model, len(data)


def look(
    ctx: ToolContext,
    *,
    question: str = "",
    window: bool = False,
    keep: bool = False,
) -> Sight:
    """Capture the screen and answer a question about it, in one step."""
    # Checked before capturing rather than after: taking a picture nobody can
    # read, then failing, wastes the capture and tells you nothing useful.
    _api_key(ctx)

    name = f"{'window' if window else 'screen'}-{time.strftime('%Y%m%d-%H%M%S')}.png"
    destination = ctx.resolve(Path("screenshots") / name)
    shot = capture(destination, window=window, env=ctx.env)
    try:
        answer, model, sent = _ask(shot.path, question.strip() or DEFAULT_QUESTION, ctx)
    finally:
        if not keep:
            # The PNG is an implementation detail of the question. Kept only on
            # request, and removed even when the model call failed — a failed
            # look should not leave litter in the workspace.
            try:
                shot.path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - permissions
                pass
    return Sight(
        answer=answer,
        path=shot.path,
        model=model,
        backend=shot.backend,
        kept=keep,
        bytes_sent=sent,
        note=shot.note,
    )


def read_image(ctx: ToolContext, *, path: str, question: str = "") -> Sight:
    """Answer a question about a picture already on disk."""
    resolved = ctx.resolve(path, must_exist=True)
    if not resolved.is_file():
        raise ToolError(f"{ctx.relative(resolved)} is not a file")
    answer, model, sent = _ask(
        resolved,
        question.strip()
        or (
            "Describe this image. If it contains text, transcribe it accurately. "
            "Be specific and do not speculate about what you cannot see."
        ),
        ctx,
    )
    return Sight(answer=answer, path=resolved, model=model, kept=True, bytes_sent=sent)


# -- tools -----------------------------------------------------------------


def _look_screen(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    sight = look(
        ctx,
        question=str(params.get("question") or ""),
        window=False,
        keep=bool(params.get("keep", False)),
    )
    return ToolResult(ok=True, output=sight.render(), data=sight.as_dict())


def _look_window(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    sight = look(
        ctx,
        question=str(params.get("question") or ""),
        window=True,
        keep=bool(params.get("keep", False)),
    )
    return ToolResult(ok=True, output=sight.render(), data=sight.as_dict())


def _look_image(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    sight = read_image(
        ctx, path=str(params["path"]), question=str(params.get("question") or "")
    )
    return ToolResult(ok=True, output=sight.render(), data=sight.as_dict())


_CAPTURE_PARAMS = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "What you want to know about it. Optional — defaults to a full "
                "description with any text transcribed."
            ),
        },
        "keep": {
            "type": "boolean",
            "description": (
                "Keep the PNG in the workspace afterwards. Default false — the "
                "image is a means to the answer, not the answer."
            ),
        },
    },
}


def tools() -> list[Tool]:
    return [
        Tool(
            name="look_at_screen",
            description=(
                "Take a screenshot and tell you what is on it, in one step. Use "
                "this for 'what's on my screen', 'what does that error say', "
                "'is it finished yet'. Needs a display and GOOGLE_API_KEY."
            ),
            run=_look_screen,
            risk=Risk.NETWORK,
            parameters=_CAPTURE_PARAMS,
        ),
        Tool(
            name="look_at_window",
            description=(
                "The same, but only the window that currently has focus. Falls "
                "back to the whole screen on desktops that do not allow it, and "
                "says so."
            ),
            run=_look_window,
            risk=Risk.NETWORK,
            parameters=_CAPTURE_PARAMS,
        ),
        Tool(
            name="look_at_image",
            description=(
                "Look at a picture already saved in the workspace and answer a "
                "question about it — a photo, a chart, a screenshot taken earlier."
            ),
            run=_look_image,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Image path, relative to the workspace.",
                    },
                    "question": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    from ..agent import default_home
    from ..tools import build_toolbox

    parser = argparse.ArgumentParser(description="Look at the screen, or at an image.")
    parser.add_argument("question", nargs="?", default="", help="what you want to know")
    parser.add_argument("--image", help="read this file instead of the screen")
    parser.add_argument("--window", action="store_true", help="the focused window only")
    parser.add_argument("--keep", action="store_true", help="keep the captured PNG")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    ctx = build_toolbox(workspace=default_home() / "workspace").context()
    try:
        sight = (
            read_image(ctx, path=args.image, question=args.question)
            if args.image
            else look(ctx, question=args.question, window=args.window, keep=args.keep)
        )
    except ToolError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(sight.as_dict(), indent=2) if args.json else sight.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
