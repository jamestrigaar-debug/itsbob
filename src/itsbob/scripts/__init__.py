"""The foundation scripts: what itsbob can do to the machine it runs on.

Each module here is three things at once, deliberately:

* an importable library of plain functions, so it can be tested without an agent;
* a **tool** the agent sees in its prompt, with a schema and a risk level;
* a standalone CLI (``python -m itsbob.scripts.system_monitor``), so a person
  can run it, read it, and change it without going through itsbob at all.

That last one matters: these are meant to be edited. A capability you cannot
inspect is one you have to take on trust, and the whole design of the tool layer
is about not having to.

**Adding a script is dropping in a file.** There is no list to update. Any
module in this package exposing ``tools() -> list[Tool]`` is discovered and
registered, and so is any ``.py`` file in ``~/.itsbob/scripts/`` (or
``ITSBOB_SCRIPTS_DIR``) that does the same — which is the path for scripts that
belong to *this machine* rather than to the project, and which survive a
reinstall. Set ``SUMMARY = "..."`` in the module for the line shown in the
scripts panel; the first line of the docstring is used if you don't.

A script that fails to import does not take the others with it: it is reported
as a broken script, because a typo in one file silently removing an unrelated
capability is the worst version of this.

Risk levels are what the policy gates on, so they are assigned by what a call
can *destroy*, not by how it is usually used:

===================  ==========================================================
``READ``             system_status, list_processes, scan_junk, check_network,
                     list_scheduled_tasks, task_history
``WRITE``            organize_folder, schedule_task, pause_task, screenshots
``EXECUTE``          start_process
``DESTRUCTIVE``      stop_process, clean_junk, remove_task, repair_network
===================  ==========================================================

In ``guarded`` mode — the default — everything from EXECUTE down needs a human
to say yes, and is refused outright when nobody is there to ask.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from ..tools.base import Tool

__all__ = [
    "MODULES",
    "script_tools",
    "describe_scripts",
    "discover",
    "user_scripts_dir",
    "load_errors",
]

#: Shown first and in this order, because they are the ones the agent is told
#: about most often and a stable order makes the prompt cacheable. Anything
#: else discovered is appended alphabetically.
CURATED = (
    "system_monitor",
    "network_checker",
    "process_manager",
    "file_cleaner",
    "screenshot",
    "screen_reader",
    "scheduler",
)

#: Populated by the last :func:`discover` call: ``{name: error}`` for modules
#: that would not import. Surfaced by ``itsbob scripts`` and the GUI panel.
load_errors: dict[str, str] = {}


def user_scripts_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where a person's own scripts live. Created on demand, never required."""
    env = os.environ if env is None else env
    override = str(env.get("ITSBOB_SCRIPTS_DIR", "")).strip()
    if override:
        return Path(override).expanduser()
    home = str(env.get("ITSBOB_HOME", "")).strip()
    return (Path(home).expanduser() if home else Path.home() / ".itsbob") / "scripts"


def _load_user_module(path: Path) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location(f"itsbob_user_scripts.{path.stem}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover(env: Mapping[str, str] | None = None) -> list[tuple[str, ModuleType, str]]:
    """Every script module that provides tools, built-in then user-supplied.

    Returns ``(name, module, summary)`` triples. Import failures are recorded in
    :data:`load_errors` rather than raised: one broken script must not remove
    every other capability from the agent's prompt.
    """
    load_errors.clear()
    found: dict[str, ModuleType] = {}

    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            load_errors[info.name] = f"{type(exc).__name__}: {exc}"
            continue
        if callable(getattr(module, "tools", None)):
            found[info.name] = module

    directory = user_scripts_dir(env)
    if directory.is_dir():
        for path in sorted(directory.glob("*.py")):
            if path.stem.startswith("_"):
                continue
            try:
                module = _load_user_module(path)
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                load_errors[path.stem] = f"{type(exc).__name__}: {exc}"
                continue
            if module is not None and callable(getattr(module, "tools", None)):
                # A user script shadowing a built-in name wins: it is the more
                # specific thing, and someone who named a file that way meant it.
                found[path.stem] = module

    order = [n for n in CURATED if n in found] + sorted(set(found) - set(CURATED))
    return [(name, found[name], _summary(found[name])) for name in order]


def _summary(module: ModuleType) -> str:
    stated = getattr(module, "SUMMARY", "")
    if stated:
        return str(stated)
    doc = (module.__doc__ or "").strip().splitlines()
    return doc[0].strip() if doc else ""


def script_tools(env: Mapping[str, str] | None = None) -> list[Tool]:
    """Every tool the discovered scripts provide.

    A module whose ``tools()`` raises is skipped, for the same reason a module
    that will not import is: a broken script costs its own capability, nothing
    else.
    """
    found: list[Tool] = []
    for name, module, _ in discover(env):
        try:
            found.extend(module.tools())
        except Exception as exc:  # noqa: BLE001
            load_errors[name] = f"tools(): {type(exc).__name__}: {exc}"
    return found


def describe_scripts(env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """One row per script, for ``itsbob scripts`` and the browser interface."""
    rows: list[dict[str, Any]] = []
    for name, module, summary in discover(env):
        try:
            tools = [
                {"name": tool.name, "risk": tool.risk.value, "description": tool.description}
                for tool in module.tools()
            ]
        except Exception as exc:  # noqa: BLE001
            load_errors[name] = f"tools(): {type(exc).__name__}: {exc}"
            tools = []
        rows.append(
            {
                "name": name,
                "summary": summary,
                "module": module.__name__,
                "source": "built-in" if module.__name__.startswith(__name__) else "user",
                "tools": tools,
            }
        )
    rows.extend(
        {"name": name, "summary": "", "module": "", "source": "broken",
         "tools": [], "error": error}
        for name, error in load_errors.items()
    )
    return rows


def __getattr__(name: str) -> Any:
    """``MODULES`` kept as a lazy alias for the discovered list.

    It used to be a hand-maintained tuple, which is exactly the thing this
    module now exists to stop needing. Keeping the name means existing callers
    and tests do not have to care that it became dynamic.
    """
    if name == "MODULES":
        return tuple(discover())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
