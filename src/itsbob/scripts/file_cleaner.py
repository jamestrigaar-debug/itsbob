"""Reclaiming disk space, and tidying folders, without ever taking something back.

Deleting files automatically is the most dangerous thing in this whole suite,
so the design is built around one rule: **it must be impossible to delete
something that was not obviously disposable, in a place you did not nominate.**

Four independent conditions, all of which must hold:

1. **Inside a nominated root.** Never ``$HOME`` by default. The workspace, plus
   whatever ``ITSBOB_CLEAN_ROOTS`` lists — and each is checked on the *resolved*
   path, so a symlink pointing out is not a way around it.
2. **Not a protected location.** System directories and itsbob's own home are
   refused outright, whatever the roots say. ``memory.sqlite`` living one
   directory up from a nominated root is not a reason to lose your memory.
3. **Matches a junk pattern.** Editable, and deliberately narrow: caches,
   build droppings, editor backups, and logs. An unrecognised file is never
   junk, however old it is.
4. **Old enough.** Seven days by default. A ``.pyc`` written a minute ago may
   belong to a program that is running right now.

Scanning is separate from cleaning, and cleaning defaults to a dry run, so the
answer to "what would this delete" never requires finding out.

``organize`` moves rather than deletes — a wrong move is annoying, a wrong
delete is not — and never overwrites: a collision gets a numbered suffix.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..tools.base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["JunkFile", "CleanReport", "scan", "clean", "organize", "tools"]

#: Never touched, whatever a root says. Anything at or under these is refused.
PROTECTED_ROOTS = (
    "/", "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/opt", "/proc",
    "/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var",
)

#: Junk by category. Directory entries end with "/".
JUNK_PATTERNS: dict[str, tuple[str, ...]] = {
    "python": ("__pycache__/", "*.pyc", "*.pyo", ".pytest_cache/", ".mypy_cache/",
               ".ruff_cache/", "*.egg-info/"),
    "editor": ("*~", "*.swp", "*.swo", ".DS_Store", "Thumbs.db", "*.orig", "*.rej"),
    "temp": ("*.tmp", "*.temp", "*.partial", "*.crdownload", "*.part"),
    "logs": ("*.log", "*.log.[0-9]", "*.log.gz"),
    "build": (".sass-cache/", ".parcel-cache/", ".next/cache/", "*.o", "*.class"),
}

#: How files are grouped by `organize`.
FILE_GROUPS: dict[str, tuple[str, ...]] = {
    "images": (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".heic", ".bmp"),
    "documents": (".pdf", ".doc", ".docx", ".odt", ".txt", ".md", ".rtf", ".epub"),
    "spreadsheets": (".csv", ".xls", ".xlsx", ".ods", ".tsv"),
    "archives": (".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".deb", ".rpm"),
    "audio": (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"),
    "video": (".mp4", ".mkv", ".mov", ".avi", ".webm"),
    "code": (".py", ".js", ".ts", ".sh", ".rs", ".go", ".c", ".h", ".java", ".json", ".yaml", ".yml"),
    "installers": (".appimage", ".AppImage", ".exe", ".dmg", ".msi", ".snap", ".flatpak"),
}

DEFAULT_MIN_AGE_DAYS = 7.0


@dataclass
class JunkFile:
    path: Path
    size: int
    age_days: float
    category: str
    is_dir: bool = False

    def as_dict(self, root: Path | None = None) -> dict[str, Any]:
        shown = str(self.path)
        if root is not None:
            try:
                shown = str(self.path.relative_to(root))
            except ValueError:
                pass
        return {
            "path": shown,
            "size_bytes": self.size,
            "age_days": round(self.age_days, 1),
            "category": self.category,
            "is_dir": self.is_dir,
        }


@dataclass
class CleanReport:
    scanned_roots: list[str] = field(default_factory=list)
    found: list[JunkFile] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    bytes_found: int = 0
    bytes_freed: int = 0
    dry_run: bool = True

    def as_dict(self) -> dict[str, Any]:
        by_category: dict[str, dict[str, int]] = {}
        for item in self.found:
            entry = by_category.setdefault(item.category, {"count": 0, "bytes": 0})
            entry["count"] += 1
            entry["bytes"] += item.size
        return {
            "dry_run": self.dry_run,
            "roots": self.scanned_roots,
            "found": len(self.found),
            "bytes_found": self.bytes_found,
            "mb_found": round(self.bytes_found / 1_048_576, 1),
            "removed": len(self.removed),
            "bytes_freed": self.bytes_freed,
            "mb_freed": round(self.bytes_freed / 1_048_576, 1),
            "by_category": by_category,
            "failed": self.failed,
        }

    def render(self, limit: int = 15) -> str:
        summary = self.as_dict()
        verb = "would free" if self.dry_run else "freed"
        amount = summary["mb_found"] if self.dry_run else summary["mb_freed"]
        rows = [f"{verb} {amount:.1f}MB across {summary['found']} item(s) "
                f"in {', '.join(self.scanned_roots) or '(nothing)'}"]
        for category, totals in sorted(summary["by_category"].items(),
                                       key=lambda kv: -kv[1]["bytes"]):
            rows.append(f"  {category:<10} {totals['count']:>4} items  "
                        f"{totals['bytes'] / 1_048_576:>8.1f}MB")
        biggest = sorted(self.found, key=lambda f: -f.size)[:limit]
        if biggest:
            rows.append("  largest:")
            for item in biggest:
                rows.append(f"    {item.size / 1_048_576:>7.1f}MB  {item.age_days:>4.0f}d  {item.path}")
        if len(self.found) > limit:
            rows.append(f"    … {len(self.found) - limit} more")
        for failure in self.failed[:5]:
            rows.append(f"  !! {failure['path']}: {failure['error']}")
        if self.dry_run and self.found:
            rows.append("  (dry run — pass dry_run=false to actually delete)")
        return "\n".join(rows)


# -- the guards ------------------------------------------------------------


def allowed_roots(ctx: ToolContext | None = None, env: dict[str, str] | None = None) -> list[Path]:
    """Directories cleaning may touch: the workspace, plus ITSBOB_CLEAN_ROOTS."""
    env = dict(os.environ if env is None else env)
    roots: list[Path] = []
    if ctx is not None:
        roots.append(ctx.workspace.resolve())
    for raw in env.get("ITSBOB_CLEAN_ROOTS", "").split(os.pathsep):
        raw = raw.strip()
        if raw:
            roots.append(Path(raw).expanduser().resolve())
    seen: dict[Path, None] = {}
    for root in roots:
        seen.setdefault(root, None)
    return list(seen)


def _itsbob_home() -> Path:
    from ..config import itsbob_home

    return itsbob_home().resolve()


def refuse_reason(path: Path, roots: Iterable[Path]) -> str | None:
    """Why this path must not be cleaned, or None if it may be."""
    try:
        resolved = path.expanduser().resolve()
    except OSError as exc:  # pragma: no cover - broken symlink chain
        return f"cannot resolve {path}: {exc}"

    if str(resolved) in PROTECTED_ROOTS:
        return f"{resolved} is a system directory"
    for protected in PROTECTED_ROOTS:
        if protected != "/" and (resolved == Path(protected) or Path(protected) in resolved.parents):
            return f"{resolved} is inside the system directory {protected}"

    home = _itsbob_home()
    workspace = home / "workspace"
    # The workspace lives *inside* the home by default, and is the one place
    # cleaning is meant to happen. An earlier version protected everything under
    # the home, which refused the workspace itself — the guard is for
    # memory.sqlite, tasks.sqlite, .env and the audit log, not for the working
    # directory that sits beside them.
    inside_workspace = resolved == workspace or workspace in resolved.parents
    if not inside_workspace:
        if resolved == home or home in resolved.parents:
            return (
                f"{resolved} is itsbob's own state ({home}) — memory, tasks, keys "
                "and the audit log live there. The workspace underneath it is fine."
            )
        if resolved in home.parents:
            return f"{resolved} contains itsbob's state at {home}"
    if resolved == Path.home().resolve():
        return "refusing to clean your home directory wholesale — nominate a subfolder"

    roots = list(roots)
    if not roots:
        return (
            "no cleanable roots are configured. Set ITSBOB_CLEAN_ROOTS "
            "(e.g. ~/Downloads:~/tmp) or clean inside the workspace."
        )
    if not any(resolved == root or root in resolved.parents for root in roots):
        listing = ", ".join(str(r) for r in roots)
        return f"{resolved} is outside the cleanable roots ({listing})"
    return None


def _classify(path: Path) -> str | None:
    """Which junk category this is, or None if it is not junk."""
    name = path.name
    for category, patterns in JUNK_PATTERNS.items():
        for pattern in patterns:
            if pattern.endswith("/"):
                if path.is_dir() and fnmatch.fnmatch(name, pattern[:-1]):
                    return category
            elif not path.is_dir() and fnmatch.fnmatch(name, pattern):
                return category
    return None


def _directory_size(path: Path) -> int:
    total = 0
    for current, _dirs, filenames in os.walk(path, followlinks=False):
        for filename in filenames:
            try:
                total += (Path(current) / filename).lstat().st_size
            except OSError:
                continue
    return total


# -- scanning and cleaning -------------------------------------------------


def scan(
    paths: Iterable[str | Path],
    *,
    ctx: ToolContext | None = None,
    min_age_days: float = DEFAULT_MIN_AGE_DAYS,
    categories: Iterable[str] | None = None,
    max_items: int = 5000,
) -> CleanReport:
    """Find disposable files. Reads only; deletes nothing."""
    roots = allowed_roots(ctx)
    wanted = set(categories) if categories else set(JUNK_PATTERNS)
    unknown = wanted - set(JUNK_PATTERNS)
    if unknown:
        raise ToolError(
            f"unknown categories {sorted(unknown)}; available: {sorted(JUNK_PATTERNS)}"
        )

    report = CleanReport(dry_run=True)
    now = time.time()
    cutoff = now - min_age_days * 86400

    for raw in paths:
        target = Path(raw).expanduser()
        refusal = refuse_reason(target, roots)
        if refusal:
            raise ToolError(f"refused: {refusal}")
        target = target.resolve()
        if not target.is_dir():
            raise ToolError(f"{target} is not a directory")
        report.scanned_roots.append(str(target))

        for current, dirnames, filenames in os.walk(target, followlinks=False):
            here = Path(current)
            # Match directories before descending, and prune the ones that
            # match so their contents are not double-counted.
            matched_dirs = []
            for name in list(dirnames):
                candidate = here / name
                category = _classify(candidate)
                if category and category in wanted:
                    try:
                        age = (now - candidate.lstat().st_mtime) / 86400
                    except OSError:
                        continue
                    if candidate.lstat().st_mtime < cutoff:
                        size = _directory_size(candidate)
                        report.found.append(
                            JunkFile(candidate, size, age, category, is_dir=True)
                        )
                        report.bytes_found += size
                        matched_dirs.append(name)
            for name in matched_dirs:
                dirnames.remove(name)

            for name in filenames:
                candidate = here / name
                category = _classify(candidate)
                if not category or category not in wanted:
                    continue
                try:
                    stat = candidate.lstat()
                except OSError:
                    continue
                if stat.st_mtime >= cutoff:
                    continue
                report.found.append(
                    JunkFile(candidate, stat.st_size, (now - stat.st_mtime) / 86400, category)
                )
                report.bytes_found += stat.st_size
                if len(report.found) >= max_items:
                    return report
    return report


def clean(
    paths: Iterable[str | Path],
    *,
    ctx: ToolContext | None = None,
    min_age_days: float = DEFAULT_MIN_AGE_DAYS,
    categories: Iterable[str] | None = None,
    dry_run: bool = True,
) -> CleanReport:
    """Scan, then delete what was found. ``dry_run`` defaults to True."""
    report = scan(paths, ctx=ctx, min_age_days=min_age_days, categories=categories)
    report.dry_run = dry_run
    if dry_run:
        return report

    roots = allowed_roots(ctx)
    for item in report.found:
        # Re-checked immediately before deletion, not just at scan time: the
        # gap between the two is where a symlink swap would land.
        refusal = refuse_reason(item.path, roots)
        if refusal:
            report.failed.append({"path": str(item.path), "error": f"refused: {refusal}"})
            continue
        try:
            if item.is_dir:
                shutil.rmtree(item.path)
            else:
                item.path.unlink()
        except OSError as exc:
            report.failed.append({"path": str(item.path), "error": str(exc)[:120]})
            continue
        report.removed.append(str(item.path))
        report.bytes_freed += item.size
    return report


def organize(
    path: str | Path,
    *,
    ctx: ToolContext | None = None,
    by: str = "type",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Sort loose files in one folder into subfolders. Moves, never deletes."""
    roots = allowed_roots(ctx)
    target = Path(path).expanduser()
    refusal = refuse_reason(target, roots)
    if refusal:
        raise ToolError(f"refused: {refusal}")
    target = target.resolve()
    if not target.is_dir():
        raise ToolError(f"{target} is not a directory")
    if by not in ("type", "date"):
        raise ToolError("by must be 'type' or 'date'")

    extension_group = {ext.lower(): group for group, exts in FILE_GROUPS.items() for ext in exts}
    moves: list[dict[str, str]] = []

    for entry in sorted(target.iterdir()):
        if entry.is_dir() or entry.name.startswith("."):
            continue
        if by == "type":
            folder = extension_group.get(entry.suffix.lower(), "other")
        else:
            folder = time.strftime("%Y-%m", time.localtime(entry.stat().st_mtime))
        destination = target / folder / entry.name
        if destination.exists() or destination == entry:
            # Never overwrite: a collision gets a numbered suffix instead.
            stem, suffix = entry.stem, entry.suffix
            for index in range(1, 1000):
                candidate = target / folder / f"{stem} ({index}){suffix}"
                if not candidate.exists():
                    destination = candidate
                    break
        moves.append({"from": entry.name, "to": f"{folder}/{destination.name}"})
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(destination))

    return {
        "path": str(target),
        "by": by,
        "dry_run": dry_run,
        "moved": len(moves),
        "moves": moves[:60],
        "folders": sorted({m["to"].split("/")[0] for m in moves}),
    }


# -- tools -----------------------------------------------------------------


def _scan_tool(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    report = scan(
        params.get("paths") or [str(ctx.workspace)],
        ctx=ctx,
        min_age_days=float(params.get("min_age_days", DEFAULT_MIN_AGE_DAYS)),
        categories=params.get("categories"),
    )
    return ToolResult(ok=True, output=report.render(), data=report.as_dict())


def _clean_tool(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    # ctx.dry_run wins: a dry-run policy must not be overridden by an argument.
    dry_run = bool(ctx.dry_run) or bool(params.get("dry_run", True))
    report = clean(
        params.get("paths") or [str(ctx.workspace)],
        ctx=ctx,
        min_age_days=float(params.get("min_age_days", DEFAULT_MIN_AGE_DAYS)),
        categories=params.get("categories"),
        dry_run=dry_run,
    )
    return ToolResult(ok=not report.failed, output=report.render(),
                      dry_run=dry_run, data=report.as_dict())


def _organize_tool(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    dry_run = bool(ctx.dry_run) or bool(params.get("dry_run", True))
    result = organize(params["path"], ctx=ctx, by=str(params.get("by", "type")), dry_run=dry_run)
    verb = "would move" if dry_run else "moved"
    lines = [f"{verb} {result['moved']} file(s) into {', '.join(result['folders']) or '(nothing)'}"]
    lines += [f"  {m['from']} -> {m['to']}" for m in result["moves"][:20]]
    if result["moved"] > 20:
        lines.append(f"  … {result['moved'] - 20} more")
    return ToolResult(ok=True, output="\n".join(lines), dry_run=dry_run, data=result)


def tools() -> list[Tool]:
    categories = ", ".join(sorted(JUNK_PATTERNS))
    return [
        Tool(
            name="scan_junk",
            description=(
                "Find disposable files (caches, build droppings, editor backups, old logs) "
                f"and report how much space they use. Categories: {categories}. Deletes nothing."
            ),
            run=_scan_tool,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "description": "Directories to scan. Defaults to the workspace."},
                    "min_age_days": {"type": "number", "description": "Ignore anything newer. Default 7."},
                    "categories": {"type": "array", "description": f"Any of: {categories}."},
                },
            },
        ),
        Tool(
            name="clean_junk",
            description=(
                "Delete the files scan_junk finds. Defaults to a dry run — pass "
                "dry_run=false to actually delete. Only touches nominated roots "
                "(the workspace, plus ITSBOB_CLEAN_ROOTS) and never system paths "
                "or itsbob's own state."
            ),
            run=_clean_tool,
            risk=Risk.DESTRUCTIVE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "paths": {"type": "array"},
                    "min_age_days": {"type": "number"},
                    "categories": {"type": "array", "description": f"Any of: {categories}."},
                    "dry_run": {"type": "boolean", "description": "Default true."},
                },
            },
        ),
        Tool(
            name="organize_folder",
            description=(
                "Sort loose files in a folder into subfolders by type or by month. "
                "Moves rather than deletes, and never overwrites. Dry run by default."
            ),
            run=_organize_tool,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "by": {"type": "string", "description": "'type' (default) or 'date'."},
                    "dry_run": {"type": "boolean", "description": "Default true."},
                },
                "required": ["path"],
            },
        ),
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Find and remove disposable files.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--delete", action="store_true", help="actually delete (default is a dry run)")
    parser.add_argument("--min-age-days", type=float, default=DEFAULT_MIN_AGE_DAYS)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = clean(args.paths, min_age_days=args.min_age_days,
                   categories=args.categories, dry_run=not args.delete)
    print(json.dumps(report.as_dict(), indent=2) if args.json else report.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
