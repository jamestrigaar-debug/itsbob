"""Catching an answer that promised a list and then did not give one.

The failure this addresses is specific and was reproduced: asked for a
matchday's results, the reply said "10 matches were played" and then named two.
Half of that was mechanical — eight of the ten had been clipped out of the tool
output before the model saw them, which :mod:`itsbob.integrations.shaping`
fixes — and half is a real habit models have of announcing a count instead of
paying it out.

The obvious remedy is a second model call after every turn asking "did you list
everything?". That doubles the bill to catch something that happens in a small
minority of turns, which is the wrong trade for a system that is paid for by
the token.

So the check is deterministic and runs on the finished text for free. It only
asks for a rewrite when **both** halves of the failure are present:

* a tool this turn returned a shaped list of several rows, so there was
  genuinely something to enumerate; and
* the answer announces a quantity without paying it out — "10 matches were
  played", "several results", "and others" — and contains fewer list rows than
  the tool returned.

Both conditions together are rare, so the extra call is rare, and when it does
fire it is fixing exactly the thing the user complained about. Nothing here
touches an answer that simply *is* short: brevity is only a fault when
something was promised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["Shortfall", "inspect", "REWRITE_INSTRUCTION"]

#: A shaped list block opens with "N things, all listed:" or similar. This is
#: the count the answer is measured against.
_SHAPED_COUNT = re.compile(r"^\s*(\d+)\s+[\w \-()']+?(?:,? all listed| match\(es\))", re.M)

#: "10 matches were played", "found 12 articles", "there are 8 results".
_ANNOUNCED = re.compile(
    r"\b(\d+)\s+(?:more\s+)?"
    r"(matches?|results?|fixtures?|articles?|items?|entries|records?|rows?|"
    r"headlines?|teams?|scorers?|files?|tasks?)\b",
    re.I,
)

#: Phrases that stand in for a list instead of being one.
_HEDGES = (
    "and others", "among others", "and more", "the rest", "several ", "a number of",
    "various ", "some of the", "etc.", "and so on", "a few ", "many of",
    "were played", "i was unable to compile", "could not compile",
    "full list", "not exhaustive",
)

#: Lines that count as an enumerated item in an answer.
_ITEM_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.M)

REWRITE_INSTRUCTION = (
    "Your answer names a number of items but does not list them. Rewrite it now, "
    "listing every single one from the tool output you already have — one per "
    "line, with all the detail each carries. Do not summarise, do not say how "
    "many there are instead of naming them, and do not drop any to save space. "
    "If the tool genuinely did not return some of them, say which are missing "
    "and why, after the ones you do have."
)


@dataclass
class Shortfall:
    """What was promised, what was delivered, and whether that is a problem."""

    available: int = 0
    announced: int = 0
    listed: int = 0
    hedge: str = ""

    @property
    def short(self) -> bool:
        return bool(self.available and self.listed < self.available)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "announced": self.announced,
            "listed": self.listed,
            "hedge": self.hedge,
            "short": self.short,
        }

    def note(self) -> str:
        return (
            f"the tools returned {self.available} items and the answer listed "
            f"{self.listed}"
            + (f" ({self.hedge.strip()!r})" if self.hedge else "")
        )


def rows_available(observations: Sequence[str]) -> int:
    """The largest shaped list any tool returned this turn."""
    best = 0
    for text in observations:
        for match in _SHAPED_COUNT.finditer(text or ""):
            try:
                best = max(best, int(match.group(1)))
            except ValueError:  # pragma: no cover - the regex guarantees digits
                continue
    return best


def inspect(answer: str, observations: Sequence[str]) -> Shortfall:
    """Decide whether ``answer`` under-delivered on a list it had in hand."""
    text = answer or ""
    available = rows_available(observations)
    if available < 3:
        # One or two items is not a list anybody can under-deliver on, and the
        # cost of being wrong here is a needless extra model call.
        return Shortfall()

    listed = len(_ITEM_LINE.findall(text))
    announced = 0
    for match in _ANNOUNCED.finditer(text):
        try:
            announced = max(announced, int(match.group(1)))
        except ValueError:  # pragma: no cover
            continue
    lowered = text.lower()
    hedge = next((h for h in _HEDGES if h in lowered), "")

    # A promise is either an explicit count or a hedge standing in for one.
    # Without either, a short answer is just a short answer.
    if not (announced >= 3 or hedge):
        return Shortfall(available=0, announced=announced, listed=listed)
    return Shortfall(available=available, announced=announced, listed=listed, hedge=hedge)
