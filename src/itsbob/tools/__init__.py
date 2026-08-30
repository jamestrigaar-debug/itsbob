"""The tool layer: the allow-list, the gate, and the capabilities themselves.

    from itsbob.tools import build_toolbox

    box = build_toolbox(memory=store)          # guarded mode, ./workspace
    result = box.call("read_file", path="notes.md")

:class:`Toolbox` bundles the four things every invocation needs — the registry,
the policy, the workspace and the audit log — so callers pass one object
instead of reassembling the context each time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import AuditLog
from .base import (
    InvalidParams,
    Risk,
    Tool,
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolError,
    ToolNotFound,
    ToolRegistry,
    ToolResult,
)
from .files import file_tools
from .http import ApiCatalog, ApiSpec, http_tools
from .memory_tools import memory_tools
from .policy import DENY_PATTERNS, Mode, Policy, Verdict
from .sandbox import run_command, sandbox_tools

__all__ = [
    "ApiCatalog",
    "ApiSpec",
    "AuditLog",
    "DENY_PATTERNS",
    "InvalidParams",
    "Mode",
    "Policy",
    "Risk",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolDenied",
    "ToolError",
    "ToolNotFound",
    "ToolRegistry",
    "ToolResult",
    "Toolbox",
    "Verdict",
    "build_toolbox",
    "default_registry",
    "run_command",
]


def default_registry(*, catalog: ApiCatalog | None = None, extra: Sequence[Tool] = ()) -> ToolRegistry:
    """Files, execution, network and memory. The full default surface."""
    from ..scripts import script_tools

    registry = ToolRegistry()
    for tool in (
        *file_tools(),
        *sandbox_tools(),
        *http_tools(catalog),
        *memory_tools(),
        # The foundation scripts: machine health, processes, network, cleanup,
        # and itsbob's own schedule. Imported here rather than at module level
        # because they reach back into config and the daemon's task store.
        *script_tools(),
    ):
        registry.register(tool)
    for tool in extra:
        registry.register(tool)
    return registry


@dataclass
class Toolbox:
    """Registry + policy + workspace + audit, wired together."""

    registry: ToolRegistry
    policy: Policy
    audit: AuditLog
    memory: Any = None
    catalog: ApiCatalog | None = None
    env: Mapping[str, str] | None = None

    def context(self, *, dry_run: bool = False, **extras: Any) -> ToolContext:
        return ToolContext(
            workspace=self.policy.workspace,
            policy=self.policy,
            memory=self.memory,
            audit=self.audit,
            env=self.env if self.env is not None else os.environ,
            dry_run=dry_run,
            extras=extras,
        )

    def invoke(self, call: ToolCall, *, dry_run: bool = False) -> ToolResult:
        """Run one call, turning every refusal into a result the loop can read.

        Denials and unknown names come back as failed :class:`ToolResult`s
        rather than exceptions on purpose: "you may not do that, because X" is
        an observation the agent should get a chance to respond to, not a crash.
        """
        try:
            return self.registry.execute(call, self.context(dry_run=dry_run))
        except (ToolNotFound, ToolDenied, InvalidParams) as exc:
            return ToolResult.failure(call.name, str(exc))

    def call(self, name: str, /, **params: Any) -> ToolResult:
        return self.invoke(ToolCall(name, params))

    def render_for_prompt(self) -> str:
        return self.registry.render_for_prompt()

    def describe(self) -> dict[str, Any]:
        return {
            "tools": self.registry.names(),
            "policy": self.policy.describe(),
            "apis": self.catalog.describe(self.env) if self.catalog else [],
            "audit": self.audit.stats(),
        }


def build_toolbox(
    *,
    memory: Any = None,
    workspace: str | Path | None = None,
    mode: Mode | str | None = None,
    confirm: Any = None,
    policy: Policy | None = None,
    catalog: ApiCatalog | None = None,
    audit_path: str | Path | None = None,
    extra_tools: Sequence[Tool] = (),
    env: Mapping[str, str] | None = None,
) -> Toolbox:
    """Assemble the default toolbox, honouring ``ITSBOB_*`` environment settings.

    The workspace is created if missing: the agent needs somewhere it is
    allowed to write before its first turn, not after its first failure.
    """
    env = os.environ if env is None else env
    catalog = catalog if catalog is not None else ApiCatalog.from_env(env)

    if policy is None:
        policy = Policy.from_env(env, confirm=confirm, workspace=workspace)
        if mode is not None:
            policy.mode = Mode(mode) if not isinstance(mode, Mode) else mode
    elif workspace is not None:
        policy.workspace = Path(workspace).expanduser()

    policy.workspace = policy.workspace.expanduser()
    policy.workspace.mkdir(parents=True, exist_ok=True)

    if audit_path is None:
        audit_path = env.get("ITSBOB_AUDIT_LOG", "").strip() or policy.workspace / ".itsbob" / "audit.jsonl"

    return Toolbox(
        registry=default_registry(catalog=catalog, extra=extra_tools),
        policy=policy,
        audit=AuditLog(path=Path(audit_path)),
        memory=memory,
        catalog=catalog,
        env=env,
    )
