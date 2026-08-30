"""The executor: the deliberate hole in the Golden Rule.

Everything else in :mod:`itsbob.tools` is a hand-written capability the model
can only *name*. These two tools let it supply the body — a shell command, or
a Python program — which is what makes the agent able to do things nobody
wrote a tool for.

That is a real widening, so it is fenced on four sides:

1. **Working directory** — the child starts in the workspace, and the tools
   that read its results are path-jailed to the same root.
2. **Environment** — the child inherits only
   :attr:`~itsbob.tools.policy.Policy.env_allowlist`. Every API key is
   withheld, so a generated script cannot read a credential it was not handed.
3. **Time** — a hard timeout, after which the process group is killed rather
   than left running.
4. **Consent** — both tools are :attr:`~itsbob.tools.base.Risk.EXECUTE`, which
   ``guarded`` mode gates behind a human, and which fails closed when nobody
   is there to ask.

What this is *not* is a container. A command that runs can still read your
home directory, because the OS says it may — the workspace jail binds the
tools, not the kernel. Run the daemon under a dedicated user account if that
matters, which on a laptop that exists to be the agent's is the natural setup
anyway.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["sandbox_tools", "run_command", "effective_timeout"]


def effective_timeout(ctx: ToolContext, requested: float | None) -> float:
    """The timeout actually applied: a caller may shorten it, never extend it."""
    ceiling = ctx.policy.timeout_seconds if ctx.policy else 60.0
    if requested is None or requested <= 0:
        return ceiling
    return min(float(requested), ceiling)


def run_command(
    argv: list[str] | str,
    ctx: ToolContext,
    *,
    shell: bool = False,
    timeout: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str, bool]:
    """Run a child process under the policy. Returns (code, stdout, stderr, timed_out)."""
    policy = ctx.policy
    timeout = effective_timeout(ctx, timeout)
    env = (
        policy.subprocess_env(ctx.env or os.environ, extra_env)
        if policy is not None
        else {**{k: v for k, v in (ctx.env or os.environ).items()}, **(extra_env or {})}
    )
    workspace = ctx.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # start_new_session puts the child in its own process group, so a timeout
    # can kill the whole tree. Without it, `sh -c 'sleep 999 & wait'` survives
    # the kill and leaks a process for every timed-out call.
    popen_kwargs: dict[str, Any] = {
        "cwd": str(workspace),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "errors": "replace",
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(argv, shell=shell, **popen_kwargs)
    except FileNotFoundError as exc:
        raise ToolError(f"command not found: {exc.filename or argv}") from exc
    except OSError as exc:
        raise ToolError(f"could not start command: {exc}") from exc

    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(process)
        stdout, stderr = process.communicate()
    return process.returncode or 0, stdout or "", stderr or "", timed_out


def _kill_tree(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    process.kill()


def _format(
    code: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    ctx: ToolContext,
    *,
    what: str,
    timeout: float,
) -> ToolResult:
    limit = ctx.policy.max_output_bytes if ctx.policy else 200_000
    stdout, truncated_out = _clip(stdout, limit)
    stderr, truncated_err = _clip(stderr, limit // 4)

    parts = []
    if stdout.strip():
        parts.append(stdout.rstrip())
    if stderr.strip():
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    if truncated_out or truncated_err:
        parts.append("[output truncated]")
    body = "\n".join(parts) or "(no output)"

    if timed_out:
        return ToolResult(
            ok=False,
            # The effective timeout, not the policy default: a per-call
            # timeout=0.5 that reports "timed out after 60s" sends whoever
            # reads it looking for the wrong problem.
            error=f"{what} timed out after {timeout:g}s and was killed",
            output=body,
            data={"exit_code": None, "timed_out": True},
        )
    ok = code == 0
    return ToolResult(
        ok=ok,
        output=body if ok else f"exit code {code}\n{body}",
        error=None if ok else f"{what} exited {code}",
        data={"exit_code": code, "timed_out": False, "stdout": stdout, "stderr": stderr},
    )


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _run_shell(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = params["command"].strip()
    if not command:
        raise ToolError("command is empty")
    timeout = effective_timeout(ctx, params.get("timeout"))
    code, out, err, timed_out = run_command(command, ctx, shell=True, timeout=timeout)
    return _format(code, out, err, timed_out, ctx, what="command", timeout=timeout)


def _run_python(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    code_text = params["code"]
    # Written into the workspace, not /tmp: the script is an artifact of what
    # the agent did, and the audit log names a file you can still open.
    scripts = ctx.workspace.resolve() / ".itsbob" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="run_", dir=scripts, delete=False, encoding="utf-8"
    )
    with handle:
        handle.write(code_text)
    script = Path(handle.name)

    timeout = effective_timeout(ctx, params.get("timeout"))
    try:
        code, out, err, timed_out = run_command([sys.executable, str(script)], ctx, timeout=timeout)
    finally:
        if not params.get("keep_script", False):
            script.unlink(missing_ok=True)
    result = _format(code, out, err, timed_out, ctx, what="python", timeout=timeout)
    result.data["script"] = str(script) if params.get("keep_script") else None
    return result


def sandbox_tools() -> list[Tool]:
    return [
        Tool(
            name="run_shell",
            description=(
                "Run a shell command in the workspace. Use for git, package managers, "
                "and existing scripts. Output is captured; there is no interactive input."
            ),
            run=_run_shell,
            risk=Risk.EXECUTE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The full command line."},
                    "timeout": {"type": "number", "description": "Seconds before it is killed. Capped by policy."},
                },
                "required": ["command"],
            },
            examples=("run_shell(command='git status --short')",),
        ),
        Tool(
            name="run_python",
            description=(
                "Run a Python program in the workspace and return its output. Use for "
                "computation, parsing, and anything with no dedicated tool. Print results — "
                "the return value of the script is not captured."
            ),
            run=_run_python,
            risk=Risk.EXECUTE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Complete Python source."},
                    "timeout": {"type": "number"},
                    "keep_script": {"type": "boolean", "description": "Leave the file on disk afterwards."},
                },
                "required": ["code"],
            },
        ),
    ]
