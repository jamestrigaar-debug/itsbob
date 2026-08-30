"""Filesystem tools. Every path is resolved inside the workspace or refused.

The jail is enforced in one place — :meth:`itsbob.tools.base.ToolContext.resolve`
— on the *resolved* path, so ``../``, an absolute path, and a symlink pointing
out of the workspace are all caught by the same check rather than by three
separate string tests that each miss a case.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from .base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["file_tools"]

#: Refuse to load something enormous into the model's context by accident.
MAX_READ_BYTES = 400_000


def _read_file(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = ctx.resolve(params["path"], must_exist=True)
    if path.is_dir():
        raise ToolError(f"{ctx.relative(path)} is a directory — use list_dir")
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(
            f"{ctx.relative(path)} is {size:,} bytes (limit {MAX_READ_BYTES:,}). "
            "Read a slice with start_line/max_lines, or search it with find_in_files."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"{ctx.relative(path)} is not UTF-8 text ({exc.reason})") from exc

    lines = text.splitlines()
    start = max(1, int(params.get("start_line", 1)))
    limit = params.get("max_lines")
    window = lines[start - 1 : (start - 1 + int(limit)) if limit else None]
    body = "\n".join(f"{start + i:>5}\t{line}" for i, line in enumerate(window))
    shown = f"{len(window)} of {len(lines)} lines"
    return ToolResult(
        ok=True,
        output=f"{ctx.relative(path)} ({shown})\n{body}",
        data={"path": ctx.relative(path), "lines": len(lines), "bytes": size},
    )


def _write_file(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = ctx.resolve(params["path"])
    content = params["content"]
    append = bool(params.get("append", False))
    existed = path.exists()
    if existed and not append and not params.get("overwrite", False):
        raise ToolError(
            f"{ctx.relative(path)} already exists. Pass overwrite=true to replace it, "
            "or append=true to add to it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(content)
    verb = "appended to" if append else ("replaced" if existed else "created")
    return ToolResult(
        ok=True,
        output=f"{verb} {ctx.relative(path)} ({len(content):,} chars)",
        data={"path": ctx.relative(path), "bytes": len(content.encode()), "created": not existed},
    )


def _list_dir(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = ctx.resolve(params.get("path", "."), must_exist=True)
    if not path.is_dir():
        raise ToolError(f"{ctx.relative(path)} is a file — use read_file")
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    limit = int(params.get("limit", 200))
    rows = []
    for entry in entries[:limit]:
        if entry.name.startswith(".") and not params.get("show_hidden", False):
            continue
        if entry.is_dir():
            rows.append(f"  {entry.name}/")
        else:
            rows.append(f"  {entry.name}  ({entry.stat().st_size:,} bytes)")
    more = f"\n  … {len(entries) - limit} more" if len(entries) > limit else ""
    return ToolResult(
        ok=True,
        output=f"{ctx.relative(path)}/\n" + ("\n".join(rows) or "  (empty)") + more,
        data={"path": ctx.relative(path), "count": len(entries)},
    )


def _find_files(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = ctx.resolve(params.get("path", "."), must_exist=True)
    pattern = params["pattern"]
    limit = int(params.get("limit", 100))
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__pycache__"))]
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                found.append(ctx.relative(Path(dirpath) / name))
                if len(found) >= limit:
                    break
        if len(found) >= limit:
            break
    return ToolResult(
        ok=True,
        output="\n".join(found) if found else f"no files matching {pattern!r} under {ctx.relative(root)}",
        data={"matches": found, "count": len(found)},
    )


def _find_in_files(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    import re

    root = ctx.resolve(params.get("path", "."), must_exist=True)
    try:
        needle = re.compile(params["pattern"], re.IGNORECASE if params.get("ignore_case", True) else 0)
    except re.error as exc:
        raise ToolError(f"invalid regex {params['pattern']!r}: {exc}") from exc
    glob = params.get("glob", "*")
    limit = int(params.get("limit", 60))

    hits: list[str] = []
    targets = [root] if root.is_file() else sorted(root.rglob(glob))
    for candidate in targets:
        if not candidate.is_file() or any(part.startswith((".", "__pycache__")) for part in candidate.parts):
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not an error, just not a match
        for number, line in enumerate(text.splitlines(), 1):
            if needle.search(line):
                hits.append(f"{ctx.relative(candidate)}:{number}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    break
        if len(hits) >= limit:
            break
    return ToolResult(
        ok=True,
        output="\n".join(hits) if hits else f"no matches for {params['pattern']!r}",
        data={"count": len(hits)},
    )


def _delete_file(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = ctx.resolve(params["path"], must_exist=True)
    if path.is_dir():
        if not params.get("recursive", False):
            raise ToolError(f"{ctx.relative(path)} is a directory — pass recursive=true to remove it")
        import shutil

        count = sum(1 for _ in path.rglob("*"))
        shutil.rmtree(path)
        return ToolResult(ok=True, output=f"removed {ctx.relative(path)}/ and {count} entries under it")
    path.unlink()
    return ToolResult(ok=True, output=f"removed {ctx.relative(path)}")


def file_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from the workspace. Returns numbered lines.",
            run=_read_file,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace."},
                    "start_line": {"type": "integer", "description": "1-based first line. Default 1."},
                    "max_lines": {"type": "integer", "description": "How many lines to return."},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="Create or replace a text file. Refuses to clobber unless overwrite=true.",
            run=_write_file,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean", "description": "Replace an existing file."},
                    "append": {"type": "boolean", "description": "Add to the end instead of replacing."},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="list_dir",
            description="List a directory in the workspace.",
            run=_list_dir,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Default '.'"},
                    "show_hidden": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
            },
        ),
        Tool(
            name="find_files",
            description="Find files by glob name pattern (e.g. '*.csv') under a directory.",
            run=_find_files,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob against the filename, e.g. '*.log'"},
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="find_in_files",
            description="Search file contents by regex. Returns path:line: match.",
            run=_find_in_files,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."},
                    "path": {"type": "string"},
                    "glob": {"type": "string", "description": "Restrict to filenames matching this glob."},
                    "ignore_case": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="delete_file",
            description="Delete a file, or a directory with recursive=true. Cannot be undone.",
            run=_delete_file,
            risk=Risk.DESTRUCTIVE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["path"],
            },
        ),
    ]
