"""Step 2: the Gatekeeper — how hard is this, really?

It answers one question per step: which tier can handle this safely and most
cheaply. It never answers the request itself, never writes prose for the user,
and is given a tiny token budget so it cannot start trying to.

Two classifiers, same output:

* the **model** classifier, run on the cheapest thing available (the local
  Back Brain if Ollama is up, otherwise Tier C);
* a **rule-based** fallback for when no model answers, so a missing key or a
  stopped Ollama degrades accuracy rather than stopping the system.

The rules encode three things a cheap model gets wrong often enough to be
worth hard-coding:

**Irreversibility outranks apparent simplicity.** "delete the old backups" is
a short, clear sentence and a Tier A decision, because being wrong costs data.
Anything matching the destructive vocabulary floors at Tier A regardless of
how simple it reads.

**Ambiguous verbs are matched as phrases.** "release" is a noun at least as
often as it is a verb, so it is matched as "release the"/"release to" rather
than bare — otherwise "fetch the latest release notes" buys a premium model
for a file read.

**Asking about a thing is not doing it.** "what did I say about the deploy?"
is a memory lookup that happens to contain the word "deploy"; reading that
noun as an instruction sends every question about past work to the premium
tier. Recall forms are matched before the destructive vocabulary — but *after*
the judgement vocabulary, so "should I delete the backups?" stays Tier A.

**Length is a weak signal, used last.** It is the only one that survives when
nothing else matches, not the primary axis.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Sequence

from ..llm.base import LLMRequest, Provider, system, user
from .ingestion import Snapshot
from .tiers import GATEKEEPER_TAGS, LEGACY_TAGS, GateDecision, Tier

__all__ = ["Gatekeeper", "classify_heuristically"]

_TAG_RE = re.compile(
    r"\[?(ROUTINE|SCRIPT|TRIVIAL|CHEAP|SIMPLE|LIGHT|STANDARD|CLOUD_B|COMPLEX|CLOUD_A|"
    r"PREMIUM|LOCAL_SUM)\]?",
    re.I,
)

# Hard to undo, or visible to other people. Floors at Tier A.
#
# Split by ambiguity. The first list is words that are almost always the verb
# they look like. The second is words that are just as often nouns —
# "the release notes", "a merge conflict", "the payment page" — and so are
# matched only in a phrase that puts them in verb position. Matching bare
# "release" sent "fetch the latest release notes" to the premium tier, which is
# the cost of over-broad matching: not a wrong answer, but a bill for nothing.
_IRREVERSIBLE = (
    "delete", "remove", "rm -", "wipe", "erase", "destroy", "purge",
    "uninstall", "revoke", "overwrite", "truncate", "drop table", "drop database",
    "deploy", "publish", "force push", "force-push",
    "pay", "buy", "purchase", "refund", "chargeback",
    "shut down", "shutdown", "reboot",
)

_IRREVERSIBLE_PHRASES = (
    "release the", "release to", "release it",
    "push to", "push the", "merge the", "merge it", "merge into",
    "send the", "send it", "send an", "send a", "email the", "email it",
    "post to", "post the", "tweet", "reply to",
    "transfer the", "transfer to", "invoice the",
    "migrate the", "migrate to", "rotate the", "reset the", "reset my",
    "revert the", "roll back", "rollback the", "restart the",
    "format the", "format /",
)

# Genuine judgement: several options, unclear criteria, or a long horizon.
_JUDGEMENT = (
    "should i", "should we", "recommend", "advise", "trade-off", "tradeoff",
    "compare", "versus", "vs", "decide", "decision", "strategy", "plan for",
    "design", "architect", "why did", "why does", "root cause", "diagnose",
    "risk", "safe to", "worth it", "pros and cons", "best way",
)

# Needs the world, or the machine, but not much thinking about it.
_TOOL_WORK = (
    "read", "open", "list", "show", "find", "search", "grep", "look up",
    "run", "execute", "check", "fetch", "download", "call the", "api",
    "file", "folder", "directory", "script", "log", "install", "build", "test",
)

# Asking *about* something, including about something destructive. Checked
# before the irreversible list, because "what did I say about the deploy?" is
# a memory lookup that happens to contain the word "deploy" — reading a noun
# as an instruction sends every question about past work to the premium tier.
_RECALL = (
    "what did i", "what do i", "what did we", "did i", "did we",
    "do you remember", "do you know", "remind me", "when did", "when is",
    "who is", "who was", "what time", "what's my", "whats my", "what is my",
    "tell me about", "what happened",
)

# Small talk and text manipulation on content already in hand. Cheapest tier.
_TRIVIAL = (
    "hello", "hi", "hey", "thanks", "thank you", "good morning", "good night",
    "how are you", "summarize", "summarise", "rephrase", "shorten", "translate",
    "spell", "proofread", "tidy up this",
)


@lru_cache(maxsize=None)
def _phrase_re(needles: tuple[str, ...]) -> re.Pattern[str]:
    """Word-boundary matcher for a vocabulary list.

    Plain substring matching is wrong here in a way that is easy to miss:
    "merge the" is a substring of "merge these two CSV files", so a request to
    merge two spreadsheets was classified as a branch merge. Boundaries are
    added only where the phrase actually ends in a word character, so entries
    like "rm -" and "format /" still match.
    """
    parts = []
    for needle in needles:
        body = re.escape(needle.strip())
        prefix = r"\b" if needle.strip()[:1].isalnum() else ""
        suffix = r"\b" if needle.strip()[-1:].isalnum() else ""
        parts.append(f"{prefix}{body}{suffix}")
    return re.compile("|".join(parts))


def _mentions(text: str, needles: Sequence[str]) -> str | None:
    match = _phrase_re(tuple(needles)).search(text)
    return match.group(0).strip() if match else None


def classify_heuristically(snapshot: Snapshot, *, routines: Sequence[str] = ()) -> GateDecision:
    """Rule-based tier choice. Never raises, never calls anything."""
    started = time.perf_counter()
    text = snapshot.render().lower()
    size = len(text)

    hit = _mentions(text, _JUDGEMENT)
    if hit:
        # Ordered above recall so "should I delete the backups?" stays at the
        # top: it is a question, but the thing being asked is a judgement call.
        tag, why = "COMPLEX", f"asks for judgement ({hit!r})"
    elif (hit := _mentions(text, _RECALL)) and size < 400:
        tag, why = "TRIVIAL", f"asking about something, not doing it ({hit!r})"
    elif (hit := _mentions(text, _IRREVERSIBLE)) or (hit := _mentions(text, _IRREVERSIBLE_PHRASES)):
        tag, why = "COMPLEX", f"mentions {hit!r} — hard to undo, so judgement before action"
    elif (hit := _mentions(text, _TRIVIAL)) and size < 400:
        tag, why = "TRIVIAL", f"small talk or text manipulation ({hit!r})"
    elif (hit := _mentions(text, _TOOL_WORK)):
        tag, why = "STANDARD", f"needs tools ({hit!r})"
    elif size < 200:
        tag, why = "TRIVIAL", f"short and unremarkable ({size} chars)"
    elif size < 600:
        tag, why = "SIMPLE", f"moderate ({size} chars), no tool or judgement signal"
    else:
        tag, why = "STANDARD", f"no strong signal, {size} chars"

    return GateDecision(
        tier=GATEKEEPER_TAGS[tag],
        fingerprint=fingerprint_for(snapshot, tag),
        source="heuristic",
        reasoning=f"rule-based: {why}",
        latency_ms=(time.perf_counter() - started) * 1000,
        metadata={"raw_tag": tag, "char_count": size},
    )


def _build_system_prompt(routines: Sequence[str]) -> str:
    routine_line = (
        f"[ROUTINE] — one of these saved routines does exactly this, no thinking needed: "
        f"{', '.join(routines)}. Name it in \"routine\".\n"
        if routines
        else ""
    )
    return (
        "You are the Gatekeeper. You do NOT answer the request. You decide which "
        "tier of intelligence should handle it, and nothing else. Each tier up "
        "costs several times more, so pick the cheapest that can do the job.\n\n"
        f"{routine_line}"
        "[TRIVIAL] — greetings, thanks, chit-chat, recalling something the user "
        "already told you, rephrasing or shortening text you were given.\n"
        "[SIMPLE] — a short factual answer, a recommendation, one obvious tool "
        "call (read a file, check the time, look something up in memory).\n"
        "[STANDARD] — real work: several tool calls, reading and writing files, "
        "running commands, calling an API. Multi-step, but the steps are clear.\n"
        "[COMPLEX] — genuine judgement: several defensible options, ambiguous "
        "instructions, planning, OR anything hard to undo (deleting, sending, "
        "deploying, paying, publishing). When in doubt between STANDARD and "
        "COMPLEX on something irreversible, choose COMPLEX.\n\n"
        "Also output a 5-word lowercase fingerprint capturing the *kind* of "
        "request, for caching — not its specific details.\n"
        'Reply as strict JSON and nothing else: {"tag": "<TAG>", '
        '"fingerprint": "<five words>", "routine": "<name, or null>"}'
    )


@dataclass
class Gatekeeper:
    """Classifies a :class:`Snapshot` into a :class:`~itsbob.router.tiers.Tier`."""

    #: Names the classifier may return as a Tier D routine. Anything not in
    #: here is treated as a classification miss, never executed.
    routines: Sequence[str] = ()
    #: Cheapest model available. Ollama when it is running, else a Tier C router.
    local_provider: Provider | None = None
    local_model: str | None = None
    #: Called as ``fn(request) -> text`` when there is no local provider. Lets
    #: the agent classify on Tier C without this module knowing about the brain.
    cloud_classifier: Any = None
    #: A registry-like object with ``.first_triggered(snapshot)`` for the
    #: deterministic Tier D check, when one is wired up.
    registry: Any = None

    def classify(self, snapshot: Snapshot) -> GateDecision:
        # Tier D first: a deterministic hit is cheapest and fastest, so no
        # model is asked at all.
        if self.registry is not None:
            try:
                name = self.registry.first_triggered(snapshot)
            except Exception:  # noqa: BLE001 - a broken trigger must not block routing
                name = None
            if name:
                return GateDecision(
                    tier=Tier.D,
                    fingerprint=fingerprint_for(snapshot, name),
                    source="trigger",
                    reasoning=f"deterministic trigger matched: {name}",
                    metadata={"routine": name},
                )

        if snapshot.is_empty:
            return GateDecision(
                tier=Tier.C,
                fingerprint="empty input",
                source="heuristic",
                reasoning="nothing to classify",
            )

        for attempt in (self._classify_with_local, self._classify_with_cloud):
            decision = attempt(snapshot)
            if decision is not None:
                return decision
        return classify_heuristically(snapshot, routines=self.routines)

    # -- model paths -------------------------------------------------------

    def _request(self, snapshot: Snapshot) -> LLMRequest:
        return LLMRequest(
            messages=[
                system(_build_system_prompt(tuple(self.routines))),
                user(snapshot.render(max_chars=2000)),
            ],
            # Small models sometimes spend budget on hidden reasoning before
            # the first visible token; too tight a cap yields an empty body and
            # a pointless fall back to the heuristic.
            max_tokens=200,
            temperature=0.0,
            json_mode=True,
            # Classification is on the critical path of every turn, so it keeps
            # the tight budget even though the local provider now allows a long
            # one for actually answering.
            metadata={"timeout": 8.0, "local_ok": True},
        )

    def _classify_with_local(self, snapshot: Snapshot) -> GateDecision | None:
        if self.local_provider is None:
            return None
        started = time.perf_counter()
        try:
            response = self.local_provider.complete_with_fallback(
                self._request(snapshot), preferred_model=self.local_model
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"local: {type(exc).__name__}: {exc}"[:200]
            return None
        return self._decision_from(
            snapshot, response.text, (time.perf_counter() - started) * 1000, response.model, "local"
        )

    def _classify_with_cloud(self, snapshot: Snapshot) -> GateDecision | None:
        if self.cloud_classifier is None:
            return None
        started = time.perf_counter()
        try:
            text = self.cloud_classifier(self._request(snapshot))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"cloud: {type(exc).__name__}: {exc}"[:200]
            return None
        return self._decision_from(
            snapshot, text or "", (time.perf_counter() - started) * 1000, "cloud", "gatekeeper"
        )

    def _decision_from(
        self, snapshot: Snapshot, text: str, latency_ms: float, model: str, source: str
    ) -> GateDecision | None:
        tag, fingerprint, routine = _parse_reply(text)
        if tag is None:
            return None  # unusable answer: let the next classifier try

        metadata: dict[str, Any] = {"raw_tag": tag, "model": model}
        reasoning = f"{source} model tagged [{tag}]"
        tier = GATEKEEPER_TAGS[tag]

        if tag == "ROUTINE":
            if routine and routine in self.routines:
                metadata["routine"] = routine
                reasoning += f" -> {routine}"
            else:
                # Tagged trivial-enough-for-a-routine but named none that
                # exists. That is a classification miss, not evidence the
                # request is unhandleable — treat it as standard work.
                tier = Tier.B
                reasoning += f" but named no known routine ({routine!r}); routing as standard"

        return GateDecision(
            tier=tier,
            fingerprint=fingerprint or fingerprint_for(snapshot, tag),
            source=source,
            reasoning=reasoning,
            latency_ms=latency_ms,
            metadata=metadata,
        )


def _parse_reply(text: str) -> tuple[str | None, str | None, str | None]:
    from ..llm.router import extract_json

    parsed = extract_json(text) or {}
    match = _TAG_RE.search(str(parsed.get("tag", ""))) or _TAG_RE.search(text or "")
    if match is None:
        return None, None, None
    tag = match.group(1).upper()
    tag = LEGACY_TAGS.get(tag, tag)
    if tag not in GATEKEEPER_TAGS:
        return None, None, None
    fingerprint = str(parsed.get("fingerprint", "")).strip() or None
    routine = str(parsed.get("routine") or "").strip() or None
    return tag, fingerprint, (None if routine in ("null", "none") else routine)


def fingerprint_for(snapshot: Snapshot, tag: str) -> str:
    """Cheap five-ish-word fingerprint when the model did not supply one."""
    words = [w for w in re.findall(r"[a-z0-9]+", snapshot.text.lower()) if len(w) > 2][:4]
    if not words:
        words = [f"{k}:{v}" for k, v in list(snapshot.facts.items())[:4]]
    words.append(tag.lower())
    return " ".join(str(w) for w in words[:5]) or tag.lower()
