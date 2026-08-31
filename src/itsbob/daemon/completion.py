"""Did the task actually get done, and what to do when it did not.

A scheduled task fails differently from a chat message. In a conversation a
thin answer is self-correcting: the person reads it, says "no, all of them",
and the next turn fixes it. Nobody is there for a 07:00 run. A task that
answers "I found several fixtures today" instead of listing them is recorded
as ``ok``, the output is delivered, and it is wrong every morning until
somebody happens to look.

So the check happens here instead, and it is deliberately narrow:

* **It asks one question** — does this output do what the prompt asked? Not
  whether it is good, or well written, or what the reader hoped for. Those are
  judgements a model will answer differently every time, and a retry loop
  driven by an unstable judge is a loop that never settles.
* **It runs on the local model.** It is a chore, it is free there, and paying a
  cloud tier to grade the output of a cloud tier is how a verification pass
  ends up costing more than the work.
* **It fails open.** If the judge errors, times out, or answers something
  unparseable, the run is complete. A broken judge must not be able to trigger
  an escalation loop on every task in the list.

When it does find a shortfall, the retry is not a re-run: the previous attempt
and the specific gap go *into* the next prompt, at a higher grade. Discarding
the first attempt would throw away the part that worked and pay for it twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..llm.base import LLMRequest, system, user
from ..router.tiers import Tier

__all__ = ["Completion", "CompletionCheck", "next_grade"]

#: One rung up, and S is the ceiling. A task that cannot be done at S will not
#: be done at S twice, so escalation stops rather than looping.
_UP = {Tier.C: Tier.B, Tier.B: Tier.A, Tier.A: Tier.S, Tier.S: Tier.S}


def next_grade(tier: Tier) -> Tier:
    return _UP.get(tier, Tier.A)


_SYSTEM = (
    "You check whether a piece of work was actually done. You are not judging "
    "quality, style or usefulness — only whether the request was carried out.\n\n"
    "Answer NO if the reply:\n"
    "- says it found or will do something without actually giving it\n"
    "- summarises or counts a list the request asked to see ('several matches')\n"
    "- answers a different, easier question than the one asked\n"
    "- reports an error, a refusal, or running out of time or steps\n"
    "- was asked for a report or a review and gives a paragraph\n\n"
    "Answer YES if the request was carried out, even briefly, and even if the "
    "honest answer turned out to be short. A correct 'nothing was scheduled "
    "today' is complete. So is work that names the part it could not do and "
    "why.\n\n"
    'Reply as strict JSON: {"complete": true|false, "missing": "<what is '
    'absent, one sentence, empty when complete>"}'
)


@dataclass
class Completion:
    """The verdict on one attempt."""

    complete: bool
    missing: str = ""
    checked: bool = False

    def carry_forward(self, prompt: str, attempt: str) -> str:
        """The prompt for the next attempt, with this one's work folded in.

        The previous output goes in whole. It is usually most of the answer —
        the failure is a missing part, not a wrong start — and re-deriving it
        costs a second full run to arrive somewhere it had already reached.
        """
        return (
            f"{prompt}\n\n"
            "---\n"
            "An earlier attempt at this produced the text below. It did not "
            f"finish the job: {self.missing or 'it did not carry out the request'}.\n\n"
            "Keep everything in it that is right, fill in what is missing, and "
            "return the complete result — not a description of what changed.\n\n"
            f"### Earlier attempt\n{attempt.strip()[:6000]}"
        )


@dataclass
class CompletionCheck:
    """Asks the cheap model whether an attempt did what was asked."""

    brain: Any
    tier: Tier = Tier.C
    #: Below this there is nothing for a model to read. Deliberately tiny: a
    #: length threshold is a bad proxy for completeness, because a correct
    #: answer is often short — "nothing was scheduled today" is 27 characters
    #: and finished. Only genuinely empty output is decided without asking.
    min_chars: int = 2
    errors: int = 0
    last_error: str | None = None

    def judge(self, *, prompt: str, output: str, status: str = "ok") -> Completion:
        text = (output or "").strip()
        if status == "failed":
            return Completion(False, "the run failed outright", checked=True)
        if len(text) < self.min_chars:
            return Completion(False, "it produced nothing at all", checked=True)

        request = LLMRequest(
            messages=[
                system(_SYSTEM),
                user(f"REQUEST:\n{prompt.strip()[:2000]}\n\nREPLY:\n{text[:6000]}"),
            ],
            temperature=0.0,
            max_tokens=200,
            # Free on the local model, which is the whole reason this is
            # affordable to run after every task.
            metadata={"local_ok": True},
        )
        try:
            payload, _ = self.brain.complete_json(self.tier, request, purpose="task.complete")
        except Exception as exc:  # noqa: BLE001 - a broken judge must not block delivery
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return Completion(True, "", checked=False)

        if not isinstance(payload, dict) or "complete" not in payload:
            return Completion(True, "", checked=False)
        return Completion(
            complete=bool(payload.get("complete")),
            missing=str(payload.get("missing") or "").strip()[:300],
            checked=True,
        )
