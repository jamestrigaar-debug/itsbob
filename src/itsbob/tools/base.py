"""Tools: the only way anything reaches the world outside the process.

The original router's Golden Rule was that a model may *name* a pre-registered
macro and never emit code to run. That rule survives here verbatim — the
registry executes names, and an unregistered name is a routing error rather
than a best-effort execution.

What changes is that one of the registered tools is itself an executor. That
is a deliberate hole in the wall, and it is why :mod:`itsbob.tools.policy`
exists: the Golden Rule alone stops a hallucinated *action*, it does nothing
about a plausible-looking command that deletes the wrong directory. The
registry answers "may this name run at all"; the policy answers "may this
call, with these arguments, run right now, and does a human need to say so
first".

Every tool declares:

* a **JSON schema** for its parameters, which is what the model is shown and
  what its call is validated against before anything executes;
* a **risk** level, which the policy maps to allow / confirm / deny;
* whether it **mutates** anything, which is what makes a dry run possible.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

__all__ = [
    "Risk",
    "ToolError",
    "ToolNotFound",
    "ToolDenied",
    "InvalidParams",
    "ToolResult",
    "ToolCall",
    "ToolContext",
    "Tool",
    "ToolRegistry",
]


class Risk(str, Enum):
    """What a tool can do if it goes wrong. Ordered, and the ordering is used.

    The ordering is defined explicitly, in both directions. An earlier version
    supplied only ``__ge__``/``__gt__``, so ``<`` and ``<=`` fell through to
    ``str`` and compared alphabetically: ``Risk.READ < Risk.EXECUTE`` was
    **False**, and ``sorted()`` put destructive first. A policy check written
    the natural way round would have permitted exactly what it meant to refuse.

    All four comparisons are written out rather than derived with
    ``functools.total_ordering``, which does nothing useful on a ``str`` mixin:
    it only fills in operators the class does not already have, and ``str``
    supplies every one of them. Decorating this class with it left ``>``
    comparing alphabetically while ``<`` compared correctly — a worse state
    than before, because the two disagreed.
    """

    READ = "read"  #: observes only — a wrong call wastes time, nothing else
    WRITE = "write"  #: changes files inside the workspace
    NETWORK = "network"  #: leaves the machine; can leak as well as fetch
    EXECUTE = "execute"  #: runs arbitrary code
    DESTRUCTIVE = "destructive"  #: deletes, or reaches outside the workspace

    @property
    def level(self) -> int:
        return _RISK_ORDER[self]

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Risk):
            return self.level < other.level
        return NotImplemented

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Risk):
            return self.level <= other.level
        return NotImplemented

    def __gt__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Risk):
            return self.level > other.level
        return NotImplemented

    def __ge__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Risk):
            return self.level >= other.level
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Risk):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


_RISK_ORDER = {
    Risk.READ: 0,
    Risk.WRITE: 1,
    Risk.NETWORK: 2,
    Risk.EXECUTE: 3,
    Risk.DESTRUCTIVE: 4,
}


class ToolError(RuntimeError):
    """Base class for tool-layer failures."""


class ToolNotFound(ToolError):
    """The model named something that isn't registered — the Golden Rule firing."""


class ToolDenied(ToolError):
    """The policy refused this call. Carries the reason, for showing the user."""


class InvalidParams(ToolError):
    """Arguments didn't match the tool's schema."""


@dataclass
class ToolResult:
    """What a tool did. ``output`` is what the model sees next turn."""

    ok: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    tool: str = ""
    #: Set when the policy ran the call as a rehearsal instead of for real.
    dry_run: bool = False

    @classmethod
    def failure(cls, tool: str, error: str) -> "ToolResult":
        return cls(ok=False, tool=tool, error=error, output=f"error: {error}")

    def render(self, *, max_chars: int = 4000) -> str:
        """The observation string fed back into the loop."""
        body = self.output if self.ok else (self.error or "failed")
        if len(body) > max_chars:
            half = max_chars // 2
            omitted = len(body) - max_chars
            body = f"{body[:half]}\n... [{omitted} characters omitted] ...\n{body[-half:]}"
        prefix = "" if self.ok else "ERROR: "
        if self.dry_run:
            prefix = "DRY RUN (nothing was changed): "
        return f"{prefix}{body}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class ToolCall:
    """One requested invocation, before it is known to be legal."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Free text from the model about why — shown to the user at a confirm prompt.
    reason: str = ""

    def render(self) -> str:
        args = ", ".join(f"{k}={_short(v)}" for k, v in self.params.items())
        return f"{self.name}({args})"

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "reason": self.reason}


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else f"{text[:limit]}…"


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach.

    Tools take this rather than importing globals so a test, a dry run and the
    daemon can each hand over a different world without the tool knowing.
    """

    workspace: Path
    policy: Any = None
    memory: Any = None
    audit: Any = None
    env: Mapping[str, str] = field(default_factory=dict)
    #: Set by the policy for a rehearsal: mutating tools must return what they
    #: *would* do and change nothing.
    dry_run: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve ``path`` inside the workspace, or refuse.

        The check is on the *resolved* path, so ``../``, an absolute path and a
        symlink pointing outward are all caught by the same test rather than by
        three separate string checks that each miss a case.
        """
        candidate = Path(path).expanduser()
        root = self.workspace.resolve()
        full = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if full != root and root not in full.parents:
            raise ToolDenied(
                f"path {path!r} resolves to {full}, which is outside the workspace ({root})"
            )
        if must_exist and not full.exists():
            raise ToolError(f"no such path: {full.relative_to(root) if full != root else '.'}")
        return full

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace.resolve())) or "."
        except ValueError:  # pragma: no cover - resolve() already prevents this
            return str(path)


ToolFn = Callable[[dict[str, Any], ToolContext], ToolResult]


@dataclass
class Tool:
    """One capability, with the schema the model is shown."""

    name: str
    description: str
    run: ToolFn
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    risk: Risk = Risk.READ
    #: False for tools that only observe — lets a dry run skip the gate entirely.
    mutates: bool = False
    #: Rough guidance for the model about when this is the right tool.
    examples: tuple[str, ...] = ()

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.parameters.get("required", ()))

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self.parameters.get("properties", {}))

    def validate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Check and coerce arguments against the schema.

        Small on purpose — enough to catch the mistakes models actually make
        (a missing required field, a string where a number belongs, an invented
        argument) without pulling in a JSON Schema engine.
        """
        properties = self.properties
        missing = [key for key in self.required if params.get(key) in (None, "")]
        if missing:
            raise InvalidParams(f"{self.name}: missing required argument(s): {', '.join(missing)}")

        unknown = [key for key in params if key not in properties]
        if unknown and properties:
            raise InvalidParams(
                f"{self.name}: unknown argument(s): {', '.join(sorted(unknown))}. "
                f"Valid arguments: {', '.join(sorted(properties)) or '(none)'}"
            )

        cleaned: dict[str, Any] = {}
        for key, value in params.items():
            spec = properties.get(key, {})
            cleaned[key] = _coerce(self.name, key, value, spec)
        for key, spec in properties.items():
            if key not in cleaned and "default" in spec:
                cleaned[key] = spec["default"]
        return cleaned

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "risk": self.risk.value,
            "mutates": self.mutates,
        }

    def signature(self) -> str:
        """``name(arg: type, optional?: type)`` — how to *call* it."""
        args = []
        for key, spec in self.properties.items():
            kind = spec.get("type", "any")
            flag = "" if key in self.required else "?"
            args.append(f"{key}{flag}: {kind}")
        return f"{self.name}({', '.join(args)})"

    def render_for_prompt(self, *, described: bool = True) -> str:
        """One line per tool — this is what the model actually reads.

        ``described=False`` drops the prose, leaving the signature. The split
        matters because the two halves answer different questions: the
        description is how you *choose* a tool, the signature is how you
        *call* one. After the first step of a turn the choosing is largely
        done, and re-sending 37 descriptions on every subsequent step was the
        single largest fixed cost in the prompt (~2,000 tokens a step).
        """
        line = f"- {self.signature()}"
        if described:
            line += f" — {self.description}"
        if self.risk >= Risk.EXECUTE:
            line += f" [risk: {self.risk.value}]"
        return line


def _coerce(tool: str, key: str, value: Any, spec: Mapping[str, Any]) -> Any:
    """Accept the near-misses models produce, reject the rest with a usable message."""
    expected = spec.get("type")
    if expected is None or value is None:
        return value
    if expected == "string":
        return value if isinstance(value, str) else json.dumps(value, default=str)
    if expected in ("number", "integer"):
        if isinstance(value, bool):
            raise InvalidParams(f"{tool}: {key} must be a {expected}, got a boolean")
        try:
            return int(value) if expected == "integer" else float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidParams(f"{tool}: {key} must be a {expected}, got {value!r}") from exc
    if expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "yes", "no"}:
            return value.strip().lower() in {"true", "yes"}
        raise InvalidParams(f"{tool}: {key} must be a boolean, got {value!r}")
    if expected == "array":
        if isinstance(value, list):
            return value
        # Models routinely send a JSON array as a string, or one bare item.
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            return parsed if isinstance(parsed, list) else [parsed]
        return [value]
    if expected == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise InvalidParams(f"{tool}: {key} must be an object, got {value!r}") from exc
            if isinstance(parsed, dict):
                return parsed
        raise InvalidParams(f"{tool}: {key} must be an object, got {value!r}")
    return value


class ToolRegistry:
    """The allow-list. Nothing outside it can be invoked, by anyone."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def add(
        self,
        name: str,
        description: str,
        run: ToolFn,
        *,
        parameters: dict[str, Any] | None = None,
        risk: Risk = Risk.READ,
        mutates: bool = False,
    ) -> Tool:
        return self.register(
            Tool(
                name=name,
                description=description,
                run=run,
                parameters=parameters or {"type": "object", "properties": {}},
                risk=risk,
                mutates=mutates,
            )
        )

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self.names()]

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.all()]

    def render_for_prompt(
        self,
        *,
        max_risk: Risk | None = None,
        described: bool = True,
        describe_only: Collection[str] | None = None,
    ) -> str:
        """The tool list as the model sees it.

        ``describe_only`` keeps full descriptions for those names and reduces
        the rest to signatures — used from the second step of a turn onwards,
        where the tools already in play are the ones still being reasoned
        about and the rest only need to remain callable.
        """
        tools = self.all()
        if max_risk is not None:
            tools = [t for t in tools if not (t.risk > max_risk)]
        if not tools:
            return "- (no tools available)"
        lines = [
            tool.render_for_prompt(
                described=described
                if describe_only is None
                else (described and tool.name in describe_only)
            )
            for tool in tools
        ]
        return "\n".join(lines)

    def render_awareness(self, *, exclude: Collection[str] = ()) -> str:
        """Compact capability guide for the agent's standing tool pre-prompt."""
        excluded = set(exclude)
        return "\n".join(
            f"- {tool.name}: {tool.description[:24].rstrip()}…"
            for tool in self.all()
            if tool.name not in excluded
        )

    def suggest(self, name: str, *, limit: int = 3) -> list[str]:
        """Near-miss names, so a wrong guess gets a usable correction back."""
        import difflib

        return difflib.get_close_matches(name, self.names(), n=limit, cutoff=0.5)

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Validate, gate, run. The single chokepoint every invocation passes."""
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            suggestions = self.suggest(call.name)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ToolNotFound(
                f"no tool named {call.name!r}.{hint} "
                f"Available: {', '.join(self.names()) or '(none)'}"
            )

        params = tool.validate(call.params)

        verdict = None
        if ctx.policy is not None:
            verdict = ctx.policy.evaluate(tool, params, call=call)
            if not verdict.allowed:
                if ctx.audit is not None:
                    ctx.audit.record(call, None, denied=verdict.reason)
                raise ToolDenied(verdict.reason)

        dry_run = ctx.dry_run or bool(verdict and verdict.dry_run)
        if dry_run and tool.mutates:
            result = ToolResult(
                ok=True,
                tool=tool.name,
                dry_run=True,
                output=f"would run {call.render()}",
                data={"params": params},
            )
        else:
            try:
                result = tool.run(params, ctx)
            except ToolError as exc:
                result = ToolResult.failure(tool.name, str(exc))
            except Exception as exc:  # noqa: BLE001 - a broken tool must not kill the loop
                result = ToolResult.failure(tool.name, f"{type(exc).__name__}: {exc}")
            result.tool = tool.name

        result.duration_ms = (time.perf_counter() - started) * 1000
        if ctx.audit is not None:
            ctx.audit.record(call, result)
        return result

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
