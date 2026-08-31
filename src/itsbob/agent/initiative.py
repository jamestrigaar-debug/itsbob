"""Speaking first: what itsbob does when nobody has asked it anything.

Everything else in the system is reactive. A message arrives, or a schedule
fires, and a turn runs. That covers the assistant half of the job and none of
the *company* half — the thing that makes a channel worth having open is that
something occasionally appears in it that you did not ask for.

So when there is nothing scheduled and nothing being typed, an initiative turn
runs: one prompt, chosen from a rotation, asking itsbob to find something worth
saying and say it — or to say nothing, which is explicitly the expected answer
most of the time.

Three rules keep this from becoming the thing you mute in a week.

**Silence is the default and the prompt says so.** Every prompt here ends by
naming "nothing worth saying" as a correct and common answer. Without that, a
model asked to produce something will always produce something, and a channel
of manufactured observations is worse than an empty one.

**It only fires when genuinely idle, and rarely.** No due tasks, nothing in
flight, no queue, and at most once every few hours with jitter so it does not
become a metronome. Waking hours only, because 3am is not company.

**Interest before novelty.** The rotation is weighted toward prompts grounded
in something real — the machine's state, what memory holds, what changed —
rather than open invitations to be creative, which is what produces the
"here's a fun fact!" texture that people mute.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["Initiative", "Prompt", "PROMPTS"]


@dataclass(frozen=True)
class Prompt:
    """One thing itsbob might go and do on its own."""

    name: str
    text: str
    #: Relative likelihood. Grounded prompts are weighted above open ones.
    weight: float = 1.0


_CLOSING = (
    "\n\nIf there is nothing genuinely worth saying, reply with exactly "
    "'nothing worth saying' and call no tools. That is the correct answer most "
    "of the time and costs you nothing. Only speak if you would want to hear it."
)

PROMPTS: tuple[Prompt, ...] = (
    Prompt(
        "machine",
        "Check this machine — disk, memory, battery, anything running that should "
        "not be. Tell me only what I would actually want to know about, and only "
        "if it needs doing something about." + _CLOSING,
        weight=2.5,
    ),
    Prompt(
        "follow_up",
        "Look back through what you remember for something left unfinished, a "
        "decision that was never made, or a problem that has come up more than "
        "once. Raise the single most useful one, briefly, with what you would "
        "suggest doing." + _CLOSING,
        weight=2.5,
    ),
    Prompt(
        "news",
        "Check the news for anything geopolitically significant or large in scale "
        "that broke since we last spoke. One short paragraph if there is something; "
        "nothing if it is an ordinary day." + _CLOSING,
        weight=2.0,
    ),
    Prompt(
        "tidy",
        "Look for something small worth tidying — junk files piling up, a folder "
        "gone chaotic, a scheduled task that has been failing quietly. Say what you "
        "found and what you would do, and do not do anything irreversible without "
        "asking." + _CLOSING,
        weight=1.5,
    ),
    Prompt(
        "notice",
        "Think about what you now know that you did not a while ago — about how I "
        "work, what I keep coming back to, what I seem to care about. If you have "
        "noticed something real and non-obvious, say it in a sentence or two. Write "
        "it to memory as your own observation (subject bob) if it is worth keeping."
        + _CLOSING,
        weight=1.0,
    ),
    Prompt(
        "curiosity",
        "Say something you actually find interesting — something you have been "
        "thinking about, an opinion of your own, a connection between two things "
        "you have read. Yours, not a fact you have retrieved to fill the silence. "
        "Remember it as your own view (subject bob) if you mean it." + _CLOSING,
        weight=1.0,
    ),
)

#: Answers that mean "there was nothing", in whatever form they come back.
_QUIET_MARKERS = (
    "nothing worth saying",
    "nothing to report",
    "nothing worth reporting",
    "nothing notable",
    "no news worth",
)


def is_quiet(answer: str) -> bool:
    """Whether an initiative turn decided there was nothing to say."""
    text = " ".join(answer.strip().lower().split())
    if not text:
        return True
    if len(text) < 200 and any(marker in text for marker in _QUIET_MARKERS):
        return True
    return text.rstrip(".!") in ("nothing", "none", "no")


@dataclass
class Initiative:
    """Decides when itsbob speaks first, and what about."""

    enabled: bool = True
    #: Shortest gap between two initiative turns. Hours, because this is
    #: company rather than monitoring — the machine health check that needs to
    #: be timely is a scheduled task, not this.
    min_interval: float = 3 * 3600.0
    #: Random extra delay on top, so it never becomes a metronome.
    jitter: float = 3600.0
    #: Local hours it may speak in. 3am is not company.
    waking_hours: tuple[int, int] = (8, 22)
    prompts: Sequence[Prompt] = PROMPTS
    rng: random.Random = field(default_factory=random.Random)
    #: Set on the first check rather than at construction, so a process that
    #: starts and stops all day does not fire every time it comes up.
    next_at: float | None = None
    last_name: str | None = None
    fired: int = 0
    spoke: int = 0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Initiative":
        env = os.environ if env is None else env
        raw = str(env.get("ITSBOB_INITIATIVE", "")).strip().lower()
        hours = _float(env.get("ITSBOB_INITIATIVE_HOURS"), 3.0)
        return cls(
            enabled=raw not in ("0", "off", "false", "no"),
            min_interval=max(300.0, hours * 3600.0),
            waking_hours=_hours(env.get("ITSBOB_INITIATIVE_WAKING"), (8, 22)),
        )

    # -- timing ------------------------------------------------------------

    def _schedule(self, now: float) -> None:
        self.next_at = now + self.min_interval + self.rng.uniform(0, self.jitter)

    def awake(self, now: float) -> bool:
        start, end = self.waking_hours
        hour = time.localtime(now).tm_hour
        return start <= hour < end if start <= end else (hour >= start or hour < end)

    def due(self, now: float | None = None) -> bool:
        """Whether it is time to speak up. Never true twice without a reset."""
        now = time.time() if now is None else now
        if not self.enabled:
            return False
        if self.next_at is None:
            # First call only arms the clock. A restart is not a reason to talk.
            self._schedule(now)
            return False
        return now >= self.next_at and self.awake(now)

    def choose(self) -> Prompt:
        """Pick the next prompt, never the same one twice in a row."""
        pool = [p for p in self.prompts if p.name != self.last_name] or list(self.prompts)
        chosen = self.rng.choices(pool, weights=[p.weight for p in pool], k=1)[0]
        self.last_name = chosen.name
        return chosen

    def fire(self, now: float | None = None) -> Prompt:
        """Take the next prompt and re-arm the clock."""
        now = time.time() if now is None else now
        self._schedule(now)
        self.fired += 1
        return self.choose()

    def record(self, answer: str) -> bool:
        """Note whether the turn actually said something. Returns True if it did."""
        if is_quiet(answer or ""):
            return False
        self.spoke += 1
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "next_at": self.next_at,
            "in_s": round(self.next_at - time.time()) if self.next_at else None,
            "fired": self.fired,
            "spoke": self.spoke,
            "last": self.last_name,
            "waking_hours": list(self.waking_hours),
        }


def _float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _hours(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    """``"8-22"`` into ``(8, 22)``, falling back rather than raising."""
    try:
        start, end = (int(part) for part in str(value).split("-", 1))
    except (TypeError, ValueError):
        return default
    if not (0 <= start <= 23 and 0 <= end <= 24):
        return default
    return start, end
