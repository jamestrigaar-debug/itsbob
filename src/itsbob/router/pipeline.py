"""Step 3, the Execution Handshake, plus Phase 2's Dynamic Tier Escalation.

::

    SCRIPT     -> execute the named macro immediately (Tier D)
    LOCAL_SUM  -> feed raw data to the local model's generation head (Tier C)
    CLOUD_B/A  -> compressed state + goal -> cheap/premium API (Tier B/A)

Escalation, per Phase 2 of the spec: a cloud call that doesn't return valid
JSON within ``cloud_timeout_seconds`` downgrades to the local Back Brain for
a safe generic action; if that also fails, it escalates to Tier S — pause,
and ask the human.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.base import AllProvidersFailed, LLMRequest, Provider, system, user
from ..llm.router import LLMRouter, extract_json
from .cache import SemanticCache
from .gatekeeper import Gatekeeper
from .ingestion import GameState, compress
from .scripts import ScriptRegistry, ScriptResult
from .tiers import GateDecision, Tier

__all__ = ["RouteResult", "ComplexityRouter"]

#: "You are itsbob. Respond concisely in 30 words or less. Output a strict
#: JSON array of action commands." — prepended to every cloud call to keep
#: output tokens (where most of the cost lives) to a minimum.
CLOUD_SYSTEM_PREFIX = (
    "You are itsbob. Respond concisely in 30 words or less. Output a strict "
    'JSON object: {"actions": ["<SCRIPT_NAME>", ...], "note": "<short reason>"}. '
    "Only use script names from the provided list — never invent one."
)

#: Phase 1's own target: screen-scrape to script execution, end to end.
END_TO_END_LATENCY_BUDGET_MS = 1800.0


@dataclass
class RouteResult:
    """Everything about one pass through the pipeline — the GUI/CLI render this."""

    tier: Tier
    decision: GateDecision
    ok: bool
    actions: list[str] = field(default_factory=list)
    script_results: list[ScriptResult] = field(default_factory=list)
    note: str = ""
    cache_hit: bool = False
    escalated_from: Tier | None = None
    needs_user: bool = False
    total_latency_ms: float = 0.0
    provider: str | None = None
    model: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "tier_label": self.tier.label,
            "decision": self.decision.as_dict(),
            "ok": self.ok,
            "actions": self.actions,
            "script_results": [
                {"action": r.action, "ok": r.ok, "detail": r.detail} for r in self.script_results
            ],
            "note": self.note,
            "cache_hit": self.cache_hit,
            "escalated_from": self.escalated_from.value if self.escalated_from else None,
            "needs_user": self.needs_user,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "within_budget": self.total_latency_ms <= END_TO_END_LATENCY_BUDGET_MS,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


class _CallTimedOut(Exception):
    """Raised by :func:`_call_with_timeout` when the deadline passes."""


def _call_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """Run ``fn`` on a daemon thread; raise :class:`_CallTimedOut` if it
    hasn't finished within ``timeout`` seconds.

    Deliberately a plain daemon :class:`threading.Thread`, not
    :class:`concurrent.futures.ThreadPoolExecutor` — an executor's worker
    threads are *not* daemons, so an abandoned call left running past its
    timeout would make the Python interpreter's own exit hook block on it
    (up to the provider's full network timeout, tens of seconds) the next
    time the process tries to shut down. A daemon thread carries no such
    obligation: the interpreter exits without waiting for it, matching
    Phase 2's "the system does not wait" for the caller *and* for the
    process as a whole.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise _CallTimedOut(f"call did not finish within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


class ComplexityRouter:
    """Ingest -> classify (Gatekeeper) -> route -> execute, with caching and escalation."""

    def __init__(
        self,
        *,
        registry: ScriptRegistry,
        gatekeeper: Gatekeeper,
        cloud_router: LLMRouter | None = None,
        premium_router: LLMRouter | None = None,
        cache: SemanticCache | None = None,
        goal: str = "win the league",
        cloud_timeout_seconds: float = 3.0,
        on_user_alert: Callable[[RouteResult], None] | None = None,
    ) -> None:
        self.registry = registry
        self.gatekeeper = gatekeeper
        #: Tier B — the cheap workhorse. Falls back to ``premium_router`` if unset.
        self.cloud_router = cloud_router
        #: Tier A — used sparingly. Defaults to the same router as Tier B when
        #: a separate expensive-model router isn't configured.
        self.premium_router = premium_router or cloud_router
        self.cache = cache or SemanticCache()
        self.goal = goal
        self.cloud_timeout_seconds = cloud_timeout_seconds
        self.on_user_alert = on_user_alert

    def route(self, raw_state: Any, *, event_window: int = 20) -> RouteResult:
        started = time.perf_counter()
        state = compress(raw_state, event_window=event_window)
        decision = self.gatekeeper.classify(state)

        cached = self.cache.get(decision.fingerprint) if decision.tier.is_cloud else None
        if cached is not None:
            result = RouteResult(
                tier=decision.tier,
                decision=decision,
                ok=True,
                actions=list(cached.get("actions", [])),
                note=cached.get("note", ""),
                cache_hit=True,
                provider=cached.get("provider"),
                model=cached.get("model"),
            )
            result.script_results = self._execute_actions(state, result.actions)
            result.total_latency_ms = (time.perf_counter() - started) * 1000
            return result

        result = self._dispatch(state, decision)
        result.total_latency_ms = (time.perf_counter() - started) * 1000
        if result.needs_user and self.on_user_alert is not None:
            self.on_user_alert(result)
        return result

    # -- dispatch by tier ------------------------------------------------------

    def _dispatch(self, state: GameState, decision: GateDecision) -> RouteResult:
        if decision.tier is Tier.D:
            return self._run_script(state, decision)
        if decision.tier is Tier.C:
            return self._run_local_summary(state, decision)
        if decision.tier in (Tier.B, Tier.A):
            return self._run_cloud(state, decision)
        return self._tier_s(state, decision, reason="gatekeeper returned Tier S directly")

    def _run_script(self, state: GameState, decision: GateDecision) -> RouteResult:
        name = decision.metadata.get("script")
        if not name or not self.registry.has(name):
            # The Gatekeeper tagged this as trivial enough for a script but
            # didn't (or couldn't) name a registered one — that's a
            # classification miss, not evidence the state is unparseable.
            # Degrade the same way an unusable cloud reply does (local safe
            # pick -> MAINTAIN_FORMATION) rather than jumping straight to a
            # user-facing halt for what's often just "no script fit."
            return self._escalate_to_local(state, decision, reason=f"no valid script named ({name!r})")
        script_result = self.registry.execute(name, state)
        return RouteResult(
            tier=Tier.D,
            decision=decision,
            ok=script_result.ok,
            actions=[name],
            script_results=[script_result],
            note=script_result.detail,
        )

    def _run_local_summary(self, state: GameState, decision: GateDecision) -> RouteResult:
        provider = self.gatekeeper.local_provider
        if provider is None:
            return self._tier_s(state, decision, reason="Tier C selected but no local model configured")
        try:
            request = LLMRequest(
                messages=[
                    system("Paraphrase this game state as one short sentence for the user."),
                    user(state.render()),
                ],
                max_tokens=80,
                temperature=0.3,
            )
            response = provider.complete_with_fallback(
                request, preferred_model=self.gatekeeper.local_model
            )
        except Exception as exc:  # noqa: BLE001
            return self._tier_s(state, decision, reason=f"local summary failed: {exc}")
        return RouteResult(
            tier=Tier.C,
            decision=decision,
            ok=True,
            note=response.text.strip(),
            provider=response.provider,
            model=response.model,
        )

    def _run_cloud(self, state: GameState, decision: GateDecision) -> RouteResult:
        router = self.premium_router if decision.tier is Tier.A else self.cloud_router
        if router is None:
            return self._escalate_to_local(state, decision, reason="no cloud router configured")

        prompt = (
            f"Goal: {self.goal}\n"
            f"State: {state.render()}\n"
            f"Available script names: {', '.join(self.registry.names())}"
        )
        request = LLMRequest(
            messages=[system(CLOUD_SYSTEM_PREFIX), user(prompt)],
            # A "30 words" answer is maybe ~50 tokens, but the JSON envelope,
            # multiple action names, and some models' non-zero reasoning
            # overhead before the first visible token eat into the same
            # budget — 200 was tight enough to truncate real responses
            # mid-array ('{"actions": ["WING_' with no closing brace).
            max_tokens=350,
            temperature=0.4,
        )

        try:
            payload, response = _call_with_timeout(
                lambda: router.complete_json(request, purpose=f"route.{decision.tier.value.lower()}"),
                timeout=self.cloud_timeout_seconds,
            )
        except _CallTimedOut:
            # Phase 2's Timeout monitor, enforced for real: the caller gets
            # control back at cloud_timeout_seconds regardless of how long
            # the vendor actually takes to answer — "the system does not
            # wait" — rather than blocking for the full round trip and only
            # discarding it as late afterwards. The call keeps running on its
            # daemon thread; its eventual result (success or error) is never
            # awaited or acted on.
            return self._escalate_to_local(state, decision, reason="cloud response exceeded timeout")
        except (AllProvidersFailed, ValueError) as exc:
            return self._escalate_to_local(state, decision, reason=str(exc))

        actions = [a for a in payload.get("actions", []) if self.registry.has(a)]
        unknown = [a for a in payload.get("actions", []) if not self.registry.has(a)]
        note = str(payload.get("note", "")).strip()
        if unknown:
            note = f"{note} (ignored unrecognized action names: {unknown})".strip()
        if not actions:
            return self._escalate_to_local(state, decision, reason="cloud reply named no known scripts")

        self.cache.put(
            decision.fingerprint,
            {"actions": actions, "note": note, "provider": response.provider, "model": response.model},
        )
        script_results = self._execute_actions(state, actions)
        return RouteResult(
            tier=decision.tier,
            decision=decision,
            ok=all(r.ok for r in script_results),
            actions=actions,
            script_results=script_results,
            note=note,
            provider=response.provider,
            model=response.model,
        )

    # -- Phase 2: escalation ----------------------------------------------------

    def _escalate_to_local(self, state: GameState, decision: GateDecision, *, reason: str) -> RouteResult:
        provider = self.gatekeeper.local_provider
        if provider is not None:
            try:
                request = LLMRequest(
                    messages=[
                        system(
                            "The cloud tactical call failed or timed out. Pick ONE safe, "
                            'generic action. Reply as JSON: {"action": "<SCRIPT_NAME>"}'
                        ),
                        user(f"State: {state.render()}\nOptions: {', '.join(self.registry.names())}"),
                    ],
                    max_tokens=40,
                    temperature=0.0,
                    json_mode=True,
                )
                response = provider.complete_with_fallback(
                    request, preferred_model=self.gatekeeper.local_model
                )
                payload = extract_json(response.text)
                name = str((payload or {}).get("action", "")).strip()
                if name and self.registry.has(name):
                    script_result = self.registry.execute(name, state)
                    return RouteResult(
                        tier=Tier.C,
                        decision=decision,
                        ok=script_result.ok,
                        actions=[name],
                        script_results=[script_result],
                        note=f"downgraded from {decision.tier.value}: {reason}",
                        escalated_from=decision.tier,
                    )
            except Exception:  # noqa: BLE001 - fall through to the hardcoded safe default
                pass

        # Local Back Brain also unavailable/unhelpful: the spec's named safe default.
        if self.registry.has("MAINTAIN_FORMATION"):
            script_result = self.registry.execute("MAINTAIN_FORMATION", state)
            return RouteResult(
                tier=Tier.C,
                decision=decision,
                ok=True,
                actions=["MAINTAIN_FORMATION"],
                script_results=[script_result],
                note=f"downgraded to hardcoded safe default: {reason}",
                escalated_from=decision.tier,
            )

        return self._tier_s(state, decision, reason=f"cloud and local both failed: {reason}")

    def _tier_s(self, state: GameState, decision: GateDecision, *, reason: str) -> RouteResult:
        return RouteResult(
            tier=Tier.S,
            decision=decision,
            ok=False,
            note="Unrecognized state. Manual override required.",
            needs_user=True,
            error=reason,
            escalated_from=decision.tier if decision.tier is not Tier.S else None,
        )

    # -- shared -----------------------------------------------------------------

    def _execute_actions(self, state: GameState, actions: list[str]) -> list[ScriptResult]:
        results = []
        for name in actions:
            if self.registry.has(name):
                results.append(self.registry.execute(name, state))
        return results
