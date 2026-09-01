"""Asking DeepSeek in a browser, and getting a usable answer back.

The point is cost. A hard question on the premium tier costs roughly twelve
times the same question on the cheapest one, and hard questions are the long
ones. Routed through a chat site that is free to use, the reasoning costs
nothing and the only paid work left is a cheap local pass to put the answer in
the right shape.

That trade is only worth making if the handoff is reliable, which is what
:mod:`itsbob.integrations.delegate` is for: the question goes out inside an
envelope asking for one fenced JSON block, and what comes back is parsed, or
shaped locally, or reported as failed. There is no path where a login wall or a
half-rendered page becomes something the agent treats as an answer.

Two honest limitations, stated because they decide whether this suits you:

**It is slower than an API.** Thirty seconds to two minutes, against two to ten.
Fine for a scheduled report or a genuinely hard question, wrong for anything
interactive.

**It depends on someone else's markup.** Selectors change without notice. When
they do this fails loudly and the tier ladder answers instead — so the failure
mode is "you paid for the answer" rather than "you got a wrong one".

Whether to lean on it is a judgement about the terms of service of the site
involved, and that judgement is the operator's. It is off unless switched on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from ..integrations.browser import (
    BrowserError,
    PlaywrightSession,
    XdotoolSession,
    available,
)
from ..integrations.delegate import Delegate, Delegation
from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["ask_deepseek", "build_delegate", "tools", "DeepSeekConfig"]

SUMMARY = "Ask DeepSeek in a browser — free reasoning, shaped locally."

#: Tried in order. Sites move their markup; a list of candidates survives a
#: rename where a single selector does not.
INPUT_SELECTORS = (
    "textarea#chat-input",
    "textarea[placeholder*='Message']",
    "div[contenteditable='true']",
    "textarea",
)
REPLY_SELECTOR = "div[class*='markdown'], div[class*='message'] div[class*='content']"


@dataclass
class DeepSeekConfig:
    url: str = "https://chat.deepseek.com"
    headless: bool = True
    #: Long, because the whole point is a question worth thinking about.
    max_wait_ms: int = 180_000
    settle_ms: int = 4000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DeepSeekConfig":
        env = os.environ if env is None else env
        return cls(
            url=str(env.get("ITSBOB_DEEPSEEK_URL", "")).strip() or cls.url,
            headless=str(env.get("ITSBOB_BROWSER_HEADLESS", "1")).strip().lower()
            not in ("0", "false", "no"),
            max_wait_ms=int(str(env.get("ITSBOB_DEEPSEEK_TIMEOUT_MS", "") or 180_000)),
        )


def enabled(env: Mapping[str, str] | None = None) -> bool:
    """Off unless switched on. Driving a third-party site is the operator's call."""
    env = os.environ if env is None else env
    return str(env.get("ITSBOB_DEEPSEEK", "")).strip().lower() in ("1", "true", "yes", "on")


def ask_deepseek(prompt: str, *, config: DeepSeekConfig | None = None, env: Any = None) -> str:
    """Send one prompt, return the raw reply. Playwright first, xdotool second."""
    config = config or DeepSeekConfig.from_env(env)
    ready = available(env)
    problems: list[str] = []

    if ready["playwright"]["ready"]:
        try:
            return PlaywrightSession(
                headless=config.headless, timeout_ms=config.max_wait_ms
            ).ask(
                config.url,
                prompt,
                input_selectors=INPUT_SELECTORS,
                reply_selector=REPLY_SELECTOR,
                settle_ms=config.settle_ms,
                max_wait_ms=config.max_wait_ms,
            )
        except Exception as exc:  # noqa: BLE001 - the other mechanism may still work
            problems.append(f"playwright: {type(exc).__name__}: {exc}"[:200])
    else:
        problems.append(f"playwright: {ready['playwright']['why']}")

    if ready["xdotool"]["ready"]:
        try:
            return XdotoolSession().ask(config.url, prompt)
        except BrowserError as exc:
            problems.append(f"xdotool: {exc}"[:200])
    else:
        problems.append(f"xdotool: {ready['xdotool']['why']}")

    raise BrowserError("; ".join(problems))


def build_delegate(formatter: Any = None, env: Any = None) -> Delegate:
    """A :class:`~itsbob.integrations.delegate.Delegate` over the browser bridge."""
    return Delegate(
        transport=lambda prompt: ask_deepseek(prompt, env=env),
        formatter=formatter,
        name="deepseek (browser)",
    )


def _ask(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    env = ctx.env if ctx.env is not None else os.environ
    if not enabled(env):
        raise ToolError(
            "the DeepSeek bridge is off. It drives a third-party chat site in a "
            "browser, so it is opt-in: set ITSBOB_DEEPSEEK=1 in .env if that suits "
            "your use of that site."
        )
    question = str(params.get("question", "")).strip()
    if not question:
        raise ToolError("question is empty")
    config = DeepSeekConfig.from_env(env)
    policy = getattr(ctx, "policy", None)
    if policy is not None:
        host_reason = policy.check_url(config.url)
        if host_reason:
            raise ToolError(host_reason)

    formatter = ctx.extras.get("shape_json") if getattr(ctx, "extras", None) else None
    # Keep the transport's small public call shape: it is injectable in tests
    # and by local integrations.  ``config`` above is only the preflight host
    # check; ``ask_deepseek`` resolves the identical configuration itself.
    result: Delegation = build_delegate(formatter, env).ask(
        question, context=str(params.get("context") or "")
    )
    if not result.ok:
        raise ToolError(
            f"{result.error}. The tier ladder can answer this instead — it just costs."
        )
    return ToolResult(ok=True, output=result.render(), data=result.as_dict())


def tools() -> list[Tool]:
    return [
        Tool(
            name="ask_deepseek",
            description=(
                "Send a hard question to DeepSeek through a browser and get a "
                "structured answer back — free, but slow (30s–2min). Worth it for "
                "genuine reasoning or analysis you would otherwise pay the top "
                "tier for; not for anything quick or interactive. Off unless "
                "ITSBOB_DEEPSEEK=1."
            ),
            run=_ask,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The full question, self-contained — there is no follow-up.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Any material it needs. Kept short; there is no caching here.",
                    },
                },
                "required": ["question"],
            },
        )
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ask DeepSeek through a browser.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--context", default="")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = DeepSeekConfig.from_env()
    config.headless = not args.headed
    delegate = Delegate(
        transport=lambda prompt: ask_deepseek(prompt, config=config),
        name="deepseek (browser)",
    )
    result = delegate.ask(" ".join(args.question), context=args.context)
    print(json.dumps(result.as_dict(), indent=2) if args.json else result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
