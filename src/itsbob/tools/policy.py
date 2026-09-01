"""The gate: may *this* call, with *these* arguments, run right now?

The registry enforces that only a known tool name executes. That stops a
hallucinated action; it does nothing about a perfectly well-formed
``run_shell("rm -rf ~/Documents")``. This module is the second lock.

Four modes, each a mapping from :class:`~itsbob.tools.base.Risk` to a verdict:

===========  ===================================================================
``readonly``  observation only — everything that changes anything is refused
``guarded``   the default: reads and workspace writes run; anything that leaves
              the machine or executes code needs a human to say yes
``dry_run``   mutating tools report what they *would* do and change nothing
``trusted``   everything runs unattended except deletion and reaching outside
              the workspace, which still ask
===========  ===================================================================

Two things this is honest about:

**The command deny-list is a guardrail, not a security boundary.** It catches
``rm -rf /`` and ``curl … | sh``, which is worth doing because those are what a
confused model actually emits. It does not stop a determined adversary, who
can spell any of it a hundred other ways. The real boundaries are the
workspace jail (enforced on resolved paths, in
:meth:`~itsbob.tools.base.ToolContext.resolve`), the confirm gate, and the
scrubbed subprocess environment.

**Confirmation fails closed.** If a call needs a human and no confirmation
handler is wired up — an unattended daemon, a web request, a cron run — the
answer is no. A prompt nobody can see is not consent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .base import Risk, Tool, ToolCall

__all__ = ["Mode", "Verdict", "Policy", "DENY_PATTERNS"]


class Mode(str, Enum):
    READONLY = "readonly"
    GUARDED = "guarded"
    DRY_RUN = "dry_run"
    TRUSTED = "trusted"


class _Gate(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"
    REHEARSE = "rehearse"


_MODE_GATES: dict[Mode, dict[Risk, _Gate]] = {
    Mode.READONLY: {
        Risk.READ: _Gate.ALLOW,
        Risk.WRITE: _Gate.DENY,
        Risk.NETWORK: _Gate.DENY,
        Risk.EXECUTE: _Gate.DENY,
        Risk.DESTRUCTIVE: _Gate.DENY,
    },
    Mode.GUARDED: {
        Risk.READ: _Gate.ALLOW,
        Risk.WRITE: _Gate.ALLOW,
        Risk.NETWORK: _Gate.CONFIRM,
        Risk.EXECUTE: _Gate.CONFIRM,
        Risk.DESTRUCTIVE: _Gate.CONFIRM,
    },
    Mode.DRY_RUN: {
        Risk.READ: _Gate.ALLOW,
        Risk.WRITE: _Gate.REHEARSE,
        Risk.NETWORK: _Gate.REHEARSE,
        Risk.EXECUTE: _Gate.REHEARSE,
        Risk.DESTRUCTIVE: _Gate.REHEARSE,
    },
    Mode.TRUSTED: {
        Risk.READ: _Gate.ALLOW,
        Risk.WRITE: _Gate.ALLOW,
        Risk.NETWORK: _Gate.ALLOW,
        Risk.EXECUTE: _Gate.ALLOW,
        Risk.DESTRUCTIVE: _Gate.CONFIRM,
    },
}


#: Shapes that are almost never what was meant, and are unrecoverable when they
#: aren't. Matched case-insensitively against the whole command string.
DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-z]*\s+)*-[a-z]*[rf][a-z]*\s+(/|~|\$HOME)\s*($|[;&|])", "recursive delete of a root or home directory"),
    (r":\(\)\s*\{.*\}\s*;?\s*:", "fork bomb"),
    (r"\bmkfs(\.\w+)?\b", "filesystem format"),
    (r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|disk)", "raw write to a block device"),
    (r">\s*/dev/(sd|nvme|hd|disk)\w*", "raw write to a block device"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "power state change"),
    (r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|k|da)?sh\b", "piping a download straight into a shell"),
    (r"\bchmod\s+(-[a-z]+\s+)*777\s+/(\s|$)", "world-writable root"),
    (r"\bsudo\b", "privilege escalation"),
    (r"\bhistory\s+-c\b|\brm\b[^\n]*\.bash_history", "clearing shell history"),
    (r"\b(nc|ncat|netcat)\b[^\n]*\s-[a-z]*e[a-z]*\s", "netcat reverse shell"),
    (r"\bgit\b[^\n]*\bpush\b[^\n]*--force\b", "force push"),
)

_COMPILED = tuple((re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in DENY_PATTERNS)


@lru_cache(maxsize=32)
def _compiled_extra(patterns: tuple[tuple[str, str], ...]) -> tuple[tuple[re.Pattern[str], str], ...]:
    """Compile caller-supplied deny patterns once, not once per tool call.

    A bad pattern is dropped with its reason rather than raising: a typo in one
    extra rule must not take the whole gate down, since a gate that raises is a
    gate that is not gating.
    """
    compiled = []
    for pattern, reason in patterns:
        try:
            compiled.append((re.compile(pattern, re.IGNORECASE), reason))
        except re.error:
            continue
    return tuple(compiled)


@dataclass
class Verdict:
    """The gate's answer, with a reason usable in a message to the user."""

    allowed: bool
    reason: str = ""
    dry_run: bool = False
    confirmed: bool = False
    #: True when a human was asked and said no, as opposed to a flat policy
    #: refusal — the agent should stop rather than look for a way around it.
    refused_by_user: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "dry_run": self.dry_run,
            "confirmed": self.confirmed,
            "refused_by_user": self.refused_by_user,
        }


ConfirmFn = Callable[[Tool, Mapping[str, Any], ToolCall], bool]


@dataclass
class Policy:
    """What the agent may do on this machine, and when it has to ask."""

    mode: Mode = Mode.GUARDED
    workspace: Path = field(default_factory=lambda: Path.cwd())
    #: Asked before any call the mode gates as ``confirm``. ``None`` denies
    #: those calls outright — see the module docstring on failing closed.
    confirm: ConfirmFn | None = None
    #: Names that always run without asking, whatever their risk. For tools you
    #: have read and trust — the escape hatch that keeps ``guarded`` usable.
    auto_allow: frozenset[str] = frozenset()
    #: Whole *risk levels* that run without asking. `auto_allow` is per tool,
    #: which is the wrong grain for a decision like "reading the web is fine":
    #: there are eight network tools and more arrive with every API, so naming
    #: them one at a time means a list that is out of date the moment it is
    #: written. Set by the setup question "may itsbob use the network freely?"
    #: and by ITSBOB_AUTO_ALLOW_RISKS. `always_confirm` still overrides it, so a
    #: single tool can be pulled back out without giving up the whole level.
    auto_allow_risks: frozenset[Risk] = frozenset()
    #: Names that always ask, whatever the mode. Wins over ``auto_allow``.
    always_confirm: frozenset[str] = frozenset()
    #: Names that never run at all.
    blocked: frozenset[str] = frozenset()
    #: Extra command patterns to refuse, on top of :data:`DENY_PATTERNS`.
    extra_deny: tuple[tuple[str, str], ...] = ()
    #: Wall-clock seconds any single subprocess may take.
    timeout_seconds: float = 60.0
    #: Bytes of captured output kept per call.
    max_output_bytes: int = 200_000
    #: Environment variable names a subprocess is allowed to inherit. Anything
    #: else — every API key included — is withheld, so a generated script
    #: cannot read a credential it was never handed.
    env_allowlist: frozenset[str] = frozenset({"PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM", "USER", "SHELL", "TMPDIR"})
    #: Hosts built-in network tools may reach. Empty means "any" — narrow it
    #: if the agent only ever needs the APIs you configured.
    allowed_hosts: frozenset[str] = frozenset()

    def gate_for(self, risk: Risk) -> _Gate:
        return _MODE_GATES[self.mode][risk]

    def evaluate(
        self, tool: Tool, params: Mapping[str, Any], *, call: ToolCall | None = None
    ) -> Verdict:
        call = call or ToolCall(tool.name, dict(params))

        if tool.name in self.blocked:
            return Verdict(False, f"{tool.name} is blocked by policy")

        pattern_reason = self._scan_command(params)
        if pattern_reason:
            # Never confirmable. These are the shapes with no recoverable
            # outcome, so "are you sure?" is the wrong question to offer.
            return Verdict(False, f"refused: {pattern_reason}")

        host_reason = self._check_host(tool, params)
        if host_reason:
            return Verdict(False, host_reason)

        gate = self.gate_for(tool.risk)
        if tool.name in self.always_confirm:
            gate = _Gate.CONFIRM
        elif gate is not _Gate.DENY and (
            tool.name in self.auto_allow or tool.risk in self.auto_allow_risks
        ):
            gate = _Gate.ALLOW

        if gate is _Gate.ALLOW:
            return Verdict(True)
        if gate is _Gate.REHEARSE:
            return Verdict(True, "dry run", dry_run=True)
        if gate is _Gate.DENY:
            return Verdict(
                False,
                f"{tool.name} is {tool.risk.value}, which {self.mode.value} mode does not permit",
            )

        if self.confirm is None:
            return Verdict(
                False,
                f"{tool.name} needs confirmation ({tool.risk.value}) and nothing is available "
                f"to ask — running unattended. Re-run interactively, add it to auto_allow, "
                f"or switch to trusted mode.",
            )
        try:
            approved = bool(self.confirm(tool, params, call))
        except Exception as exc:  # noqa: BLE001 - a broken prompt must not mean "yes"
            return Verdict(False, f"confirmation failed ({type(exc).__name__}: {exc})")
        if approved:
            return Verdict(True, "confirmed by user", confirmed=True)
        return Verdict(False, "declined by user", refused_by_user=True)

    def _scan_command(self, params: Mapping[str, Any]) -> str | None:
        blob = " ".join(
            str(value) for key, value in params.items() if key in ("command", "code", "script", "args")
        )
        if not blob.strip():
            return None
        for pattern, reason in (*_COMPILED, *_compiled_extra(self.extra_deny)):
            if pattern.search(blob):
                return reason
        return None

    def _check_host(self, tool: Tool, params: Mapping[str, Any]) -> str | None:
        if not self.allowed_hosts or tool.risk is not Risk.NETWORK:
            return None
        url = str(params.get("url", ""))
        if not url:
            return None
        return self.check_url(url)

    def check_url(self, url: str) -> str | None:
        """Return a refusal when ``url`` falls outside the configured egress list.

        Network tools that construct their URL from a named API call use this
        after resolving it.  Keeping the check here makes the policy one source
        of truth instead of teaching every tool its own host-matching rules.
        """
        if not self.allowed_hosts:
            return None
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        allowed_hosts = {str(value).lower().strip(".") for value in self.allowed_hosts}
        if host in allowed_hosts:
            return None
        if any(host.endswith(f".{allowed}") for allowed in allowed_hosts):
            return None
        return f"host {host!r} is not in the allowed_hosts list"

    def subprocess_env(self, base: Mapping[str, str], extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """The environment a spawned process gets: allowlisted vars only."""
        env = {k: v for k, v in base.items() if k in self.env_allowlist}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        if extra:
            env.update(extra)
        return env

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "workspace": str(self.workspace),
            "interactive": self.confirm is not None,
            "auto_allow": sorted(self.auto_allow),
            "auto_allow_risks": sorted(r.value for r in self.auto_allow_risks),
            "always_confirm": sorted(self.always_confirm),
            "blocked": sorted(self.blocked),
            "timeout_seconds": self.timeout_seconds,
            "allowed_hosts": sorted(self.allowed_hosts) or ["(any)"],
            "gates": {risk.value: self.gate_for(risk).value for risk in Risk},
        }

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        confirm: ConfirmFn | None = None,
        workspace: str | Path | None = None,
    ) -> "Policy":
        import os

        env = os.environ if env is None else env
        raw = env.get("ITSBOB_TOOL_MODE", "").strip().lower() or Mode.GUARDED.value
        try:
            mode = Mode(raw)
        except ValueError:
            mode = Mode.GUARDED
        root = Path(workspace or env.get("ITSBOB_WORKSPACE", "").strip() or Path.cwd() / "workspace")
        return cls(
            mode=mode,
            workspace=Path(root).expanduser(),
            confirm=confirm,
            auto_allow=frozenset(_csv(env.get("ITSBOB_AUTO_ALLOW", ""))),
            auto_allow_risks=_risks(env.get("ITSBOB_AUTO_ALLOW_RISKS", "")),
            always_confirm=frozenset(_csv(env.get("ITSBOB_ALWAYS_CONFIRM", ""))),
            blocked=frozenset(_csv(env.get("ITSBOB_BLOCKED_TOOLS", ""))),
            allowed_hosts=frozenset(_csv(env.get("ITSBOB_ALLOWED_HOSTS", ""))),
            timeout_seconds=_float(env, "ITSBOB_TOOL_TIMEOUT", 60.0),
        )


def _risks(value: str) -> frozenset[Risk]:
    """``"network,read"`` into risk levels, ignoring anything unrecognised.

    Silently dropping a typo rather than raising: this is read at startup from
    an environment variable, and a misspelling must not stop itsbob booting.
    An unrecognised level simply grants nothing, which is the safe direction.
    """
    found = set()
    for name in _csv(value):
        try:
            found.add(Risk(name.strip().lower()))
        except ValueError:
            continue
    return frozenset(found)


def _csv(value: str) -> Iterable[str]:
    return (part.strip() for part in value.split(",") if part.strip())


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, "").strip() or default)
    except ValueError:
        return default
