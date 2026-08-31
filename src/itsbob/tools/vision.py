"""Looking at images: what is in this picture, and what does it say.

Two tools. ``image_info`` is local and free — dimensions, format, mode, EXIF
orientation — and answers a surprising share of the questions actually asked
about a file ("is this the screenshot or the photo?"). ``describe_image`` sends
the image to a vision model and asks a question about it.

Three decisions worth naming.

**It talks to Gemini's REST endpoint directly.** The rest of the ladder goes
through an OpenAI-compatible shim whose message type is a plain string, and
widening that to multimodal content parts would touch every provider, every
router and every prompt builder for the sake of one tool. A hundred lines of
``urllib`` here is the smaller change.

**Images are downscaled before they are sent.** A modern phone photo is 4000px
wide and costs a great deal to send at full size, for a description that would
be identical from a 1024px copy. Pillow does the resize when it is installed;
without it the file is sent as-is, with a size ceiling that refuses rather than
silently spends.

**A missing Pillow is not an error.** It is an optional extra, and the two
things that genuinely need it — resizing and local metadata — degrade to
"send it whole" and "here are the bytes and the format from the header".
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from .base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = [
    "vision_tools",
    "describe_image",
    "prepare_image",
    "vision_models",
    "pillow_available",
]

#: Refused outright above this, since it is being uploaded.
MAX_BYTES = 20 * 1024 * 1024
#: Longest edge after downscaling. Enough for text in a screenshot to stay
#: legible; small enough that the call is cheap.
MAX_EDGE = 1024
TIMEOUT = 90.0

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

#: Vision-capable Gemini models, cheapest first. Overridable with
#: ITSBOB_VISION_MODEL for the same reason every other model id is.
VISION_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-flash-lite-latest")


def pillow_available() -> bool:
    try:
        import PIL  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _resolve(path: str, ctx: ToolContext) -> Path:
    """An image path, through the same workspace jail every file tool uses."""
    resolved = ctx.resolve(path, must_exist=True)
    if not resolved.is_file():
        raise ToolError(f"{ctx.relative(resolved)} is not a file")
    return resolved


def prepare_image(path: Path) -> tuple[bytes, str]:
    """Image bytes ready to send, and their mime type.

    Public because :mod:`itsbob.scripts.screen_reader` needs exactly this and
    reimplementing the downscale there would mean two size ceilings that drift.
    """
    size = path.stat().st_size
    raw = path.read_bytes()
    mime = _MIME.get(path.suffix.lower(), "image/jpeg")

    try:
        from PIL import Image  # noqa: PLC0415 - optional extra
    except ImportError:
        if size > MAX_BYTES:
            raise ToolError(
                f"{path.name} is {size / 1e6:.1f} MB, over the {MAX_BYTES / 1e6:.0f} MB "
                "ceiling. Install the vision extra (`pip install -e '.[vision]'`) so it "
                "can be resized, or point at a smaller copy."
            ) from None
        return raw, mime

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            if max(image.size) > MAX_EDGE:
                image.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue(), "image/jpeg"
    except Exception as exc:  # noqa: BLE001 - an unreadable image is a tool error
        raise ToolError(f"could not read {path.name} as an image: {exc}") from exc


def describe_image(
    *,
    data: bytes,
    mime: str,
    prompt: str,
    api_key: str,
    models: tuple[str, ...] = VISION_MODELS,
) -> tuple[str, str]:
    """Ask a vision model about an image. Returns ``(answer, model used)``."""
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime,
                                "data": base64.b64encode(data).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
        }
    ).encode("utf-8")

    problems: list[str] = []
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            problems.append(f"{model}: HTTP {exc.code} {detail}")
            continue
        except Exception as exc:  # noqa: BLE001 - try the next model
            problems.append(f"{model}: {type(exc).__name__}: {exc}")
            continue

        parts = (
            (payload.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
        )
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        if text:
            return text, model
        problems.append(f"{model}: empty reply")

    raise ToolError("no vision model answered — " + "; ".join(problems)[:400])


def vision_models(env: Mapping[str, str]) -> tuple[str, ...]:
    """The vision ladder, with ``ITSBOB_VISION_MODEL`` promoted to the front."""
    override = str(env.get("ITSBOB_VISION_MODEL", "")).strip()
    if not override:
        return VISION_MODELS
    return (override, *(m for m in VISION_MODELS if m != override))


def _describe(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    env = ctx.env if ctx.env is not None else os.environ
    api_key = str(env.get("GOOGLE_API_KEY", "")).strip()
    if not api_key:
        raise ToolError("GOOGLE_API_KEY is not set, so there is no vision model to ask")

    path = _resolve(params["path"], ctx)
    data, mime = prepare_image(path)
    question = str(params.get("question") or "").strip() or (
        "Describe this image. If it contains text, transcribe it accurately. Be "
        "specific and concrete; do not speculate about anything you cannot see."
    )
    answer, model = describe_image(
        data=data, mime=mime, prompt=question, api_key=api_key, models=vision_models(env)
    )
    return ToolResult(
        ok=True,
        output=answer,
        data={"path": str(path), "model": model, "bytes_sent": len(data), "mime": mime},
    )


def _info(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve(params["path"], ctx)
    size = path.stat().st_size
    facts: dict[str, Any] = {
        "path": str(path),
        "bytes": size,
        "suffix": path.suffix.lower(),
    }
    try:
        from PIL import Image  # noqa: PLC0415 - optional extra

        with Image.open(path) as image:
            facts.update(
                format=image.format,
                mode=image.mode,
                width=image.width,
                height=image.height,
            )
    except ImportError:
        facts["note"] = (
            "Pillow is not installed, so only the file's own metadata is available. "
            "`pip install -e '.[vision]'` adds dimensions and format."
        )
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not read {path.name} as an image: {exc}") from exc

    lines = [f"{k}: {v}" for k, v in facts.items()]
    return ToolResult(ok=True, output="\n".join(lines), data=facts)


def vision_tools() -> list[Tool]:
    return [
        Tool(
            name="describe_image",
            description=(
                "Look at an image file and answer a question about it — what is in "
                "it, what a screenshot says, whether a chart shows what you expect. "
                "Downscaled before sending, so large photos are fine."
            ),
            run=_describe,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Image path, absolute or relative to the workspace.",
                    },
                    "question": {
                        "type": "string",
                        "description": "What you want to know. Defaults to a full description.",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="image_info",
            description=(
                "Dimensions, format and size of an image file, read locally. Free "
                "and instant — use it before `describe_image` when the question is "
                "about the file rather than the picture."
            ),
            run=_info,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]
