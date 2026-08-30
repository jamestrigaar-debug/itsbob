"""The foundation scripts: what itsbob can do to the machine it runs on.

Each module here is three things at once, deliberately:

* an importable library of plain functions, so it can be tested without an agent;
* a **tool** the agent sees in its prompt, with a schema and a risk level;
* a standalone CLI (``python -m itsbob.scripts.system_monitor``), so a person
  can run it, read it, and change it without going through itsbob at all.

That last one matters: these are meant to be edited. A capability you cannot
inspect is one you have to take on trust, and the whole design of the tool layer
is about not having to.

Risk levels are what the policy gates on, so they are assigned by what a call
can *destroy*, not by how it is usually used:

===================  ==========================================================
``READ``             system_status, list_processes, scan_junk, check_network,
                     list_scheduled_tasks, task_history
``WRITE``            organize_folder, schedule_task, pause_task
``EXECUTE``          start_process
``DESTRUCTIVE``      stop_process, clean_junk, remove_task, repair_network
===================  ==========================================================

In ``guarded`` mode — the default — everything from EXECUTE down needs a human
to say yes, and is refused outright when nobody is there to ask.
"""

from __future__ import annotations

from typing import Any

from ..tools.base import Tool
from . import file_cleaner, network_checker, process_manager, scheduler, system_monitor

__all__ = ["MODULES", "script_tools", "describe_scripts"]

#: Every foundation script, in the order they are described to the agent.
MODULES = (
    ("system_monitor", system_monitor, "Machine health, and whether it is fit for heavy work."),
    ("network_checker", network_checker, "Connectivity, latency, and which layer is broken."),
    ("process_manager", process_manager, "List, start and stop background processes."),
    ("file_cleaner", file_cleaner, "Find and remove disposable files; tidy folders."),
    ("scheduler", scheduler, "Read and write itsbob's own scheduled work."),
)


def script_tools() -> list[Tool]:
    """Every tool the foundation scripts provide."""
    found: list[Tool] = []
    for _, module, _ in MODULES:
        found.extend(module.tools())
    return found


def describe_scripts() -> list[dict[str, Any]]:
    """One row per script, for ``itsbob scripts`` and the browser interface."""
    return [
        {
            "name": name,
            "summary": summary,
            "module": module.__name__,
            "tools": [
                {"name": tool.name, "risk": tool.risk.value, "description": tool.description}
                for tool in module.tools()
            ],
        }
        for name, module, summary in MODULES
    ]
