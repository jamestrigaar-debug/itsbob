"""Deciding whether to interrupt, and then doing it.

An always-on assistant that reports every scheduled run is a spam generator
you will mute inside a week, and a muted assistant is worth nothing. So a
notification is two separate decisions:

1. **Is this worth saying?** A cheap model reads the result and answers yes or
   no. It runs on Tier C, because judging noteworthiness costs far less than
   the work that produced the result, and paying premium rates to decide
   whether to speak would invert the whole point of the ladder.
2. **How does it reach the user?** A sink — desktop notification, a file, a
   webhook, or the terminal.

The gate is biased toward silence, and the prompt says so explicitly. The
failure mode people actually experience is too many notifications, not too
few, and the second failure is recoverable by reading the log while the first
ends with notifications turned off.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ..llm.base import LLMRequest, system, user
from ..router.tiers import Tier

__all__ = ["Notification", "Sink", "ConsoleSink", "FileSink", "DesktopSink", "WebhookSink",
           "MultiSink", "NoticeGate", "default_sink"]


@dataclass
class Notification:
    title: str
    body: str
    task: str = ""
    urgency: str = "normal"  # low | normal | high
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "task": self.task,
            "urgency": self.urgency,
            "at": self.at,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.at)),
        }


class Sink(Protocol):
    def send(self, notification: Notification) -> bool: ...


@dataclass
class ConsoleSink:
    stream: Any = None

    def send(self, notification: Notification) -> bool:
        import sys

        stream = self.stream or sys.stdout
        stamp = time.strftime("%H:%M", time.localtime(notification.at))
        print(f"\n[{stamp}] {notification.title}\n{notification.body}\n", file=stream, flush=True)
        return True


@dataclass
class FileSink:
    """Append-only JSONL. The record that survives a closed terminal."""

    path: Path

    def send(self, notification: Notification) -> bool:
        path = Path(self.path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(notification.as_dict(), default=str) + "\n")
        return True


@dataclass
class DesktopSink:
    """Native notification: notify-send on Linux, osascript on macOS.

    Returns False rather than raising when the mechanism is missing, so it can
    always be in the chain and simply not fire on a headless box.
    """

    app_name: str = "itsbob"

    def send(self, notification: Notification) -> bool:
        body = notification.body[:400]
        try:
            if platform.system() == "Darwin":
                script = (
                    f'display notification {json.dumps(body)} '
                    f'with title {json.dumps(notification.title)}'
                )
                subprocess.run(["osascript", "-e", script], check=True, timeout=10,
                               capture_output=True)
                return True
            if shutil.which("notify-send"):
                subprocess.run(
                    ["notify-send", "-a", self.app_name, "-u", notification.urgency,
                     notification.title, body],
                    check=True, timeout=10, capture_output=True,
                )
                return True
        except (subprocess.SubprocessError, OSError):
            return False
        return False


@dataclass
class WebhookSink:
    """POST the notification as JSON. For phones, Slack, or anything with a URL."""

    url: str
    timeout: float = 15.0

    def send(self, notification: Notification) -> bool:
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {"text": f"{notification.title}\n{notification.body}", **notification.as_dict()}
        ).encode()
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, TimeoutError):
            return False


@dataclass
class MultiSink:
    """Try each sink; succeed if any did. A dead webhook must not lose the message."""

    sinks: list[Any] = field(default_factory=list)

    def send(self, notification: Notification) -> bool:
        delivered = False
        for sink in self.sinks:
            try:
                delivered = bool(sink.send(notification)) or delivered
            except Exception:  # noqa: BLE001 - one broken sink must not block the rest
                continue
        return delivered


_GATE_SYSTEM = (
    "You decide whether a background task's result is worth interrupting someone "
    "for. Interrupting has a real cost: an assistant that pings about routine "
    "successes gets muted, and a muted assistant is useless.\n\n"
    "Say yes only for: something that went wrong, something that needs a decision, "
    "a meaningful change since last time, or a result the person explicitly asked "
    "to be told about.\n"
    "Say no for: routine success, nothing changed, the same news as last time, or "
    "anything the person can read later without cost.\n\n"
    'Reply as strict JSON: {"notify": true|false, "title": "<under 60 chars>", '
    '"body": "<2 sentences max>", "urgency": "low|normal|high"}\n'
    "When genuinely unsure, answer false."
)


@dataclass
class NoticeGate:
    """Cheap-tier judgement on whether a result deserves an interruption."""

    brain: Any
    tier: Tier = Tier.C
    enabled: bool = True

    def judge(self, *, task_name: str, prompt: str, result: str, previous: str = "") -> Notification | None:
        if not self.enabled or not result.strip():
            return None

        body = (
            f"Task: {task_name}\nIts standing instruction: {prompt[:500]}\n\n"
            f"What it produced this run:\n{result[:3000]}"
        )
        if previous:
            body += f"\n\nWhat it produced last run (do not repeat this):\n{previous[:1000]}"

        try:
            payload, _ = self.brain.complete_json(
                self.tier,
                LLMRequest(
                    messages=[system(_GATE_SYSTEM), user(body)],
                    temperature=0.0,
                    max_tokens=500,
                ),
                purpose="notify.gate",
                default={"notify": False},
            )
        except Exception:  # noqa: BLE001 - a broken gate stays silent rather than spamming
            return None

        if not payload.get("notify"):
            return None
        urgency = str(payload.get("urgency", "normal")).lower()
        return Notification(
            title=str(payload.get("title") or task_name)[:120],
            body=str(payload.get("body") or result)[:1000],
            task=task_name,
            urgency=urgency if urgency in ("low", "normal", "high") else "normal",
        )


def default_sink(home: Path, *, console: bool = True, webhook: str | None = None) -> MultiSink:
    """Desktop where available, always a file, console when attached."""
    sinks: list[Any] = [FileSink(path=Path(home) / "notifications.jsonl"), DesktopSink()]
    if webhook or os.environ.get("ITSBOB_WEBHOOK_URL", "").strip():
        sinks.append(WebhookSink(url=webhook or os.environ["ITSBOB_WEBHOOK_URL"].strip()))
    if console:
        sinks.append(ConsoleSink())
    return MultiSink(sinks=sinks)
