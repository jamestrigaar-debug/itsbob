"""``itsbob`` — the command line.

Grouped by what you are trying to do:

    itsbob chat                     talk to it, interactively
    itsbob ask "..."                one question, one answer
    itsbob serve                    run the always-on daemon
    itsbob task ...                 the scheduled work it does on its own
    itsbob memory ...               what it remembers
    itsbob doctor                   what is actually configured
    itsbob gui                      the browser interface

Everything writes under ``~/.itsbob`` (``ITSBOB_HOME``): memory, workspace,
task list, audit log. One directory, so backing it up or starting over is a
single command either way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .config import Settings, load_dotenv

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # pragma: no cover - `| head`
        return 0
    except Exception as exc:  # noqa: BLE001 - a traceback is not a user interface
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("ITSBOB_DEBUG"):
            raise
        print("(set ITSBOB_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _home(args: argparse.Namespace) -> Path:
    from .agent import default_home

    return Path(args.home).expanduser() if getattr(args, "home", None) else default_home()


def _confirmer(auto_yes: bool = False):
    """Interactive approval for confirm-gated tools.

    Returns ``None`` when stdin is not a terminal, which makes the policy fail
    closed rather than blocking forever on a prompt nobody can answer.
    """
    if auto_yes:
        return lambda tool, params, call: True
    if not sys.stdin.isatty():
        return None

    def ask(tool, params, call) -> bool:
        detail = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in params.items())
        print(f"\n  ⚠  {tool.name}({detail})")
        if call.reason:
            print(f"     why: {call.reason}")
        print(f"     risk: {tool.risk.value}")
        try:
            return input("     allow? [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    return ask


def _build_agent(args: argparse.Namespace, *, interactive: bool = True):
    from .agent import Persona, build_agent

    persona = Persona()
    instructions = os.environ.get("ITSBOB_INSTRUCTIONS", "").strip()
    persona_file = _home(args) / "persona.md"
    if persona_file.is_file():
        instructions = f"{persona_file.read_text(encoding='utf-8').strip()}\n{instructions}".strip()
    if instructions:
        persona.instructions = instructions

    return build_agent(
        home=_home(args),
        mode=getattr(args, "mode", None),
        confirm=_confirmer(getattr(args, "yes", False)) if interactive else None,
        persona=persona,
        max_steps=getattr(args, "max_steps", 8),
        embeddings=not getattr(args, "no_embeddings", False),
    )


def _print_turn(turn: Any, *, verbose: bool = False) -> None:
    if verbose:
        for step in turn.steps:
            if step.tool:
                mark = "ok" if step.ok else "!!"
                print(f"   {mark} {step.tool}  {_short(step.params)}")
    print(turn.final)
    if turn.remembered:
        for item in turn.remembered:
            print(f"   · remembered: {item}")


def _short(params: dict[str, Any], limit: int = 90) -> str:
    text = ", ".join(f"{k}={v}" for k, v in params.items())
    return text if len(text) <= limit else f"{text[:limit]}…"


def _ago(when: float | None) -> str:
    if not when:
        return "never"
    delta = time.time() - when
    if delta < 0:
        return f"in {_span(-delta)}"
    return f"{_span(delta)} ago"


def _span(seconds: float) -> str:
    for limit, unit, size in ((90, "s", 1), (5400, "m", 60), (172800, "h", 3600)):
        if seconds < limit:
            return f"{int(seconds // size)}{unit}"
    return f"{int(seconds // 86400)}d"


# --------------------------------------------------------------------------
# chat / ask
# --------------------------------------------------------------------------


def _cmd_chat(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    policy = agent.toolbox.policy
    print(f"itsbob — {policy.mode.value} mode, workspace {policy.workspace}")
    print("type your message; /help for commands, /quit to leave\n")

    while True:
        try:
            message = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message.startswith("/"):
            if _repl_command(message, agent, args):
                return 0
            continue

        print("bob › ", end="", flush=True)
        turn = agent.chat(message, on_event=_streamer(args.verbose))
        if args.verbose:
            print()
        _print_turn(turn, verbose=False)
        print()


def _streamer(verbose: bool):
    """Live progress while a turn runs, so it is never a silent wait."""
    state = {"first": True}

    def show(event) -> None:
        data = event.data
        if event.kind == "classified" and verbose:
            print(f"\n   [tier {data['tier']}] {data['decision']['reasoning']}")
        elif event.kind == "tool":
            print(f"\n   → {data['name']}({_short(data['params'])})", flush=True)
            state["first"] = False
        elif event.kind == "observation" and verbose:
            mark = "ok" if data["ok"] else "!!"
            print(f"   {mark} {data['output'].splitlines()[0][:110] if data['output'] else ''}")
        elif event.kind == "final" and not state["first"]:
            print("   ", end="")

    return show


def _repl_command(line: str, agent: Any, args: argparse.Namespace) -> bool:
    """Handle a ``/command``. Returns True to exit the REPL."""
    command, _, rest = line[1:].partition(" ")
    command = command.lower()
    if command in ("quit", "exit", "q"):
        return True
    if command == "help":
        print(
            "  /memory <query>   search what it remembers\n"
            "  /forget <id>      delete a memory\n"
            "  /tools            list tools and the current policy\n"
            "  /audit            recent tool activity\n"
            "  /new              start a fresh conversation (memory is kept)\n"
            "  /quit             leave\n"
        )
    elif command == "memory":
        hits = agent.memory.search(rest or "", limit=10) if agent.memory else []
        for hit in hits:
            print(f"  {hit.record.id[:8]}  {hit.record.content}  ({hit.reason})")
        if not hits:
            print("  (nothing)")
    elif command == "forget":
        print("  forgotten" if agent.memory and agent.memory.forget(rest.strip()) else "  no such id")
    elif command == "tools":
        print(agent.toolbox.render_for_prompt())
        print(f"\n  policy: {json.dumps(agent.toolbox.policy.describe(), indent=2)}")
    elif command == "audit":
        for entry in agent.toolbox.audit.recent(15):
            mark = "!!" if entry["denied"] else ("ok" if entry["ok"] else "xx")
            print(f"  {entry['iso']}  {mark} {entry['tool']}  {entry['error'] or entry['output'][:60]}")
    elif command == "new":
        from .agent.context import Conversation

        agent.conversation = Conversation()
        print("  new conversation (long-term memory kept)")
    else:
        print(f"  unknown command /{command} — try /help")
    return False


def _cmd_ask(args: argparse.Namespace) -> int:
    agent = _build_agent(args, interactive=sys.stdin.isatty())
    turn = agent.chat(args.prompt, on_event=_streamer(args.verbose) if args.verbose else None)
    if args.json:
        print(json.dumps(turn.as_dict(), indent=2, default=str))
    else:
        _print_turn(turn, verbose=args.verbose)
    return 0 if not turn.error else 1


# --------------------------------------------------------------------------
# daemon
# --------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from .daemon import build_daemon

    def show(event) -> None:
        stamp = time.strftime("%H:%M:%S")
        if event.kind == "running":
            print(f"[{stamp}] running {event.data['task']}", flush=True)
        elif event.kind == "finished":
            print(
                f"[{stamp}] {event.data['task']}: {event.data['status']} "
                f"({event.data['duration_ms']:.0f}ms)"
                + ("  → notified" if event.data.get("notified") else ""),
                flush=True,
            )
        elif event.kind in ("error", "started", "stopped"):
            print(f"[{stamp}] {event.kind}: {event.data}", flush=True)

    daemon = build_daemon(
        home=_home(args),
        mode=getattr(args, "mode", None),
        console=False,
        on_event=show,
    )
    state = daemon.describe()
    print(f"itsbob daemon — {state['tasks']} task(s), {state['policy_mode']} mode")
    print(f"  workspace: {state['workspace']}")
    if not state["can_run_commands"]:
        print(
            "  note: nobody is here to approve anything, so tools needing "
            "confirmation will be refused.\n"
            "        Grant specific ones with ITSBOB_AUTO_ALLOW=run_shell,... "
            "or run with --mode trusted."
        )
    if not state["tasks"]:
        print('  no tasks yet — add one with: itsbob task add "name" "what to do" "every 30m"')
    print("  ctrl-c to stop\n")

    if args.once:
        daemon.tick()
        return 0
    daemon.run_forever()
    return 0


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


def _task_store(args: argparse.Namespace):
    from .daemon import TaskStore

    return TaskStore(_home(args) / "tasks.sqlite")


def _cmd_task(args: argparse.Namespace) -> int:
    store = _task_store(args)
    action = args.task_action

    if action == "list":
        tasks = store.all()
        if not tasks:
            print('no tasks. Add one: itsbob task add "name" "what to do" "every 30m"')
            return 0
        for task in tasks:
            when = _ago(task.next_run) if task.next_run else "—"
            state = "on " if task.enabled else "off"
            print(
                f"[{state}] {task.id}  {task.name:<22} {task.schedule:<20} "
                f"next {when:<10} runs={task.run_count} {task.last_status or ''}"
            )
        return 0

    if action == "add":
        task = store.create(
            args.name, args.prompt, args.schedule, notify=not args.quiet, max_runs=args.max_runs
        )
        print(f"added {task.id}  {task.name}  ({task.schedule}), next {_ago(task.next_run)}")
        return 0

    task = store.find(args.name)
    if task is None:
        print(f"no task named or numbered {args.name!r}", file=sys.stderr)
        return 1

    if action == "remove":
        store.remove(task.id)
        print(f"removed {task.name}")
    elif action in ("enable", "disable"):
        store.set_enabled(task.id, action == "enable")
        print(f"{action}d {task.name}")
    elif action == "run":
        from .daemon import build_daemon

        daemon = build_daemon(home=_home(args), mode=getattr(args, "mode", None), console=False,
                              tasks=store)
        run = daemon.run_task(task)
        print(f"{run.status} in {run.duration_ms:.0f}ms\n{run.output}")
        return 0 if run.status == "ok" else 1
    elif action == "runs":
        for entry in store.runs(task.id, limit=args.limit):
            print(
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(entry['started_at']))}  "
                f"{entry['status']:<7} {entry['duration_ms']:>7.0f}ms  {entry['output'][:90]}"
            )
    elif action == "show":
        print(json.dumps(task.as_dict(), indent=2, default=str))
    return 0


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------


def _open_memory(args: argparse.Namespace, *, embeddings: bool = True):
    from .llm.embeddings import default_embedder
    from .memory.long_term import LongTermMemory

    path = Path(args.db).expanduser() if getattr(args, "db", None) else _home(args) / "memory.sqlite"
    return LongTermMemory(path, embedder=default_embedder() if embeddings else None)


def _cmd_memory(args: argparse.Namespace) -> int:
    action = args.memory_action
    # Always attach the embedder, even for commands that never embed. Building
    # it costs nothing (the HTTP client is lazy), and opening the store without
    # one made `memory stats` report "keyword-only recall, set GOOGLE_API_KEY"
    # on a machine where the key was set and semantic recall was working — a
    # diagnostic that describes the diagnostic tool rather than the system.
    store = _open_memory(args)

    if action == "search":
        hits = store.search(args.query, limit=args.limit)
        if not hits:
            print("(nothing found)")
            return 0
        for hit in hits:
            record = hit.record
            print(
                f"{record.id[:8]}  {hit.score:5.2f}  [{record.kind.value}] {record.content}\n"
                f"          {hit.reason}, {_ago(record.created_at)}"
                + (f", tags: {', '.join(record.tags)}" if record.tags else "")
            )
    elif action == "add":
        from .memory.base import MemoryKind, MemoryRecord

        record = store.add(
            MemoryRecord(
                content=args.content,
                kind=MemoryKind(args.kind),
                importance=args.importance,
                tags=tuple(args.tags or ()),
                metadata={"source": "cli"},
            )
        )
        print(f"remembered {record.id[:8]}: {record.content}")
    elif action == "list":
        for record in store.recent(limit=args.limit):
            print(f"{record.id[:8]}  [{record.kind.value:<11}] {_ago(record.created_at):>8}  {record.content}")
    elif action == "forget":
        print("forgotten" if store.forget(args.id) else f"no memory with id {args.id!r}")
        return 0 if store.get(args.id) is None else 1
    elif action == "stats":
        stats = store.stats()
        for key, value in stats.items():
            print(f"  {key:<16} {value}")
        # Branch on the actual cause. "Set GOOGLE_API_KEY" is wrong advice when
        # the key is set and the store is simply empty, and wrong advice in a
        # diagnostic is worse than none — it sends you to fix the one thing
        # that was never broken.
        if stats["offline_embedder"]:
            print(
                "\n  note: using the offline hashing embedder — recall works but "
                "cannot match paraphrases. Set GOOGLE_API_KEY (and unset "
                "ITSBOB_EMBED_OFFLINE), then run `itsbob memory reindex`."
            )
        elif stats["embedder"] is None:
            print(
                "\n  note: recall is keyword-only — no embedding model is configured. "
                "Set GOOGLE_API_KEY, then run `itsbob memory reindex`."
            )
        elif stats["unembedded"]:
            print(
                f"\n  note: {stats['unembedded']} record(s) have no vector for the "
                "current embedding model, so semantic recall will not find them. "
                "Run `itsbob memory reindex`."
            )
        if stats["degraded"]:
            print(f"\n  warning: embeddings degraded — {stats['last_embed_error']}")
    elif action == "reindex":
        print(f"re-embedded {store.reindex()} record(s) at {store.embedder.signature}")
    elif action == "prune":
        print(f"pruned {store.prune(args.keep)} record(s), {len(store)} remain")
    return 0


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .agent.brain import build_brain
    from .llm.base import LLMRequest, user
    from .llm.embeddings import default_embedder
    from .llm.local import default_ollama_config, is_ollama_running, list_ollama_models

    home = _home(args)
    print(f"home: {home}  ({'exists' if home.exists() else 'will be created'})\n")

    print("model tiers:")
    brain = build_brain(Settings.from_env(dotenv=None))
    for tier_value, info in brain.describe()["tiers"].items():
        configured = [row for row in info["providers"] if row["configured"]]
        first = configured[0] if configured else None
        mark = "ok " if first else "-- "
        detail = f"{first['provider']}: {first['models'][0]}" if first else "nothing configured"
        print(f"  {mark}Tier {tier_value} ({info['label']:<16}) {detail}")

    print("\nlocal Back Brain (free, private, preferred for Tier C):")
    if is_ollama_running():
        config = default_ollama_config()
        pulled = list_ollama_models()
        print(f"  ok  ollama reachable at {config.base_url}")
        wanted = list(config.models())
        missing = [m for m in wanted if m not in pulled]
        print(f"      pulled: {pulled or '(none)'}")
        if missing == wanted:
            print(f"      !! none of {wanted} pulled — run: ollama pull {wanted[0]}")
    else:
        print("  --  not reachable — Tier C uses the cheapest cloud model instead")

    print("\nmemory:")
    store = _open_memory(args)
    for key, value in store.stats().items():
        print(f"      {key:<16} {value}")

    print("\nembeddings:")
    for row in default_embedder().describe():
        mark = "ok " if row["configured"] else "-- "
        print(f"  {mark}{row['name']:<14} {row['model']:<24} dims={row['dims']}")

    print("\ntools:")
    from .tools import build_toolbox

    box = build_toolbox(workspace=home / "workspace", mode=getattr(args, "mode", None))
    policy = box.policy.describe()
    print(f"      mode={policy['mode']}  workspace={policy['workspace']}")
    print(f"      {len(box.registry)} tools: {', '.join(box.registry.names())}")
    if box.catalog and len(box.catalog):
        print("\nAPIs:")
        for row in box.catalog.describe():
            mark = "ok " if row["configured"] else "-- "
            print(f"  {mark}{row['name']:<14} {row['base_url']}")

    print("\ntasks:")
    tasks = _task_store(args).all()
    print(f"      {len(tasks)} configured, {sum(1 for t in tasks if t.enabled)} enabled")

    if not args.probe:
        print("\n(pass --probe to send one real request per model)")
        return 0

    print("\nprobing every model on every tier:")
    answered = 0
    seen: set[tuple[str, str]] = set()
    for tier_value, info in brain.describe()["tiers"].items():
        router = brain.router_for(_tier(tier_value))
        for provider in router.providers:
            if not provider.is_configured():
                continue
            for model in provider.models:
                if (provider.name, model) in seen:
                    continue
                seen.add((provider.name, model))
                try:
                    response = provider.complete(
                        LLMRequest(
                            messages=[user("Reply with the single word: ready")],
                            max_tokens=1000,
                            temperature=0.0,
                        ),
                        model=model,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {provider.name:<12} {model:<30} {type(exc).__name__}: {exc}"[:150])
                    continue
                print(
                    f"  ok {provider.name:<12} {model:<30} {response.latency_ms:6.0f}ms "
                    f"{response.text.strip()[:24]!r}"
                )
                answered += 1
    # A partly-degraded set of free providers is the normal state, not a failure.
    print(f"\n{answered} model(s) answered." if answered else "\nNothing answered.")
    return 0 if answered else 1


def _tier(value: str):
    from .router.tiers import Tier

    return Tier(value)


def _cmd_tools(args: argparse.Namespace) -> int:
    from .tools import build_toolbox

    box = build_toolbox(workspace=_home(args) / "workspace", mode=getattr(args, "mode", None))
    print(box.render_for_prompt())
    print("\npolicy:")
    for key, value in box.policy.describe().items():
        print(f"  {key:<18} {value}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .tools.audit import AuditLog

    log = AuditLog(path=_home(args) / "audit.jsonl")
    entries = list(log.read(limit=args.limit))
    if not entries:
        print("(no tool activity recorded yet)")
        return 0
    for entry in entries:
        mark = "denied" if entry["denied"] else ("ok" if entry["ok"] else "failed")
        print(f"{entry['iso']}  {mark:<7} {entry['tool']:<14} {_short(entry.get('params', {}), 60)}")
        if entry.get("error"):
            print(f"    {entry['error'][:150]}")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui.app import run_gui

    run_gui(host=args.host, port=args.port, open_browser=not args.no_browser, home=_home(args))
    return 0


# --------------------------------------------------------------------------
# legacy: the character simulation
# --------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from .factory import build_simulation

    settings = Settings.from_env(dotenv=None)
    if args.db:
        settings = Settings(
            providers=settings.providers,
            memory=settings.memory.with_database(args.db),
            energy=settings.energy,
            allow_offline=True if args.offline else settings.allow_offline,
            max_attempts=settings.max_attempts,
            seed=settings.seed,
        )
    sim = build_simulation(settings, policy=args.policy, seed=args.seed)
    reports = []
    for report in sim.stream(args.ticks):
        reports.append(report)
        if not args.json:
            print(report.line())
    sim.finish()
    if args.json:
        print(json.dumps({"ticks": [r.as_dict() for r in reports], "summary": sim.summary()}, indent=2))
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itsbob",
        description="A memory-backed assistant with a tiered LLM router and a task daemon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  itsbob chat\n"
            '  itsbob ask "what did I say about the deploy?"\n'
            '  itsbob task add inbox "Summarise new files in ~/inbox" "every 30m"\n'
            "  itsbob serve\n"
            "  itsbob doctor --probe\n"
        ),
    )
    parser.add_argument("--home", help="state directory (default ~/.itsbob, or ITSBOB_HOME)")
    subparsers = parser.add_subparsers(dest="command")

    def add(name: str, help_text: str, **kwargs: Any) -> argparse.ArgumentParser:
        return subparsers.add_parser(name, help=help_text, description=help_text, **kwargs)

    def with_mode(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument(
            "--mode",
            choices=["readonly", "guarded", "dry_run", "trusted"],
            help="what tools may do without asking (default guarded, or ITSBOB_TOOL_MODE)",
        )
        return sub

    # chat
    chat = with_mode(add("chat", "Interactive conversation."))
    chat.add_argument("-v", "--verbose", action="store_true", help="show tiers and tool results")
    chat.add_argument("-y", "--yes", action="store_true", help="approve every tool without asking")
    chat.add_argument("--max-steps", type=int, default=8)
    chat.add_argument("--no-embeddings", action="store_true", help="keyword-only recall, no API calls")
    chat.set_defaults(handler=_cmd_chat)

    # ask
    ask = with_mode(add("ask", "One question, one answer."))
    ask.add_argument("prompt")
    ask.add_argument("-v", "--verbose", action="store_true")
    ask.add_argument("-y", "--yes", action="store_true")
    ask.add_argument("--json", action="store_true", help="machine-readable turn record")
    ask.add_argument("--max-steps", type=int, default=8)
    ask.add_argument("--no-embeddings", action="store_true")
    ask.set_defaults(handler=_cmd_ask)

    # serve
    serve = with_mode(add("serve", "Run the always-on daemon."))
    serve.add_argument("--once", action="store_true", help="run whatever is due, then exit")
    serve.set_defaults(handler=_cmd_serve)

    # task
    task = add("task", "Scheduled work it does on its own.")
    task_subs = task.add_subparsers(dest="task_action", required=True)
    task_subs.add_parser("list", help="every task and when it next runs").set_defaults(
        handler=_cmd_task
    )
    add_task = task_subs.add_parser("add", help="create a task")
    add_task.add_argument("name")
    add_task.add_argument("prompt", help="what to do, in plain language")
    add_task.add_argument("schedule", help="'every 30m', 'weekdays at 08:30', 'at 2026-09-01T06:00'")
    add_task.add_argument("--quiet", action="store_true", help="never notify, whatever it finds")
    add_task.add_argument("--max-runs", type=int, help="retire after this many runs")
    add_task.set_defaults(handler=_cmd_task)
    for name, help_text in (
        ("remove", "delete a task"),
        ("enable", "resume a task"),
        ("disable", "pause a task"),
        ("show", "full detail for one task"),
    ):
        sub = task_subs.add_parser(name, help=help_text)
        sub.add_argument("name", help="task id or name")
        sub.set_defaults(handler=_cmd_task)
    run_task = with_mode(task_subs.add_parser("run", help="run a task now, off-schedule"))
    run_task.add_argument("name")
    run_task.set_defaults(handler=_cmd_task)
    runs = task_subs.add_parser("runs", help="run history for a task")
    runs.add_argument("name")
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(handler=_cmd_task)

    # memory
    memory = add("memory", "What it remembers.")
    memory_subs = memory.add_subparsers(dest="memory_action", required=True)
    search = memory_subs.add_parser("search", help="hybrid keyword + semantic recall")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    search.set_defaults(handler=_cmd_memory)
    add_memory = memory_subs.add_parser("add", help="store something by hand")
    add_memory.add_argument("content")
    add_memory.add_argument("--kind", default="fact",
                            choices=["observation", "action", "decision", "reflection", "fact", "dialogue"])
    add_memory.add_argument("--importance", type=float, default=0.6)
    add_memory.add_argument("--tags", nargs="*")
    add_memory.set_defaults(handler=_cmd_memory)
    list_memory = memory_subs.add_parser("list", help="most recent memories")
    list_memory.add_argument("--limit", type=int, default=20)
    list_memory.set_defaults(handler=_cmd_memory)
    forget = memory_subs.add_parser("forget", help="delete one memory by id")
    forget.add_argument("id")
    forget.set_defaults(handler=_cmd_memory)
    memory_subs.add_parser("stats", help="size, and which recall paths are live").set_defaults(
        handler=_cmd_memory
    )
    memory_subs.add_parser(
        "reindex", help="re-embed after changing embedding model or dimensions"
    ).set_defaults(handler=_cmd_memory)
    prune = memory_subs.add_parser("prune", help="keep only the most valuable N")
    prune.add_argument("keep", type=int)
    prune.set_defaults(handler=_cmd_memory)
    for sub in memory_subs.choices.values():
        sub.add_argument("--db", help="a specific memory database")

    # diagnostics
    doctor = with_mode(add("doctor", "What is actually configured, and what works."))
    doctor.add_argument("--probe", action="store_true", help="send one real request per model")
    doctor.set_defaults(handler=_cmd_doctor)

    with_mode(add("tools", "Tools available, and the current policy.")).set_defaults(
        handler=_cmd_tools
    )

    audit = add("audit", "Recent tool activity, including refusals.")
    audit.add_argument("--limit", type=int, default=40)
    audit.set_defaults(handler=_cmd_audit)

    gui = add("gui", "Browser interface.")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true")
    gui.set_defaults(handler=_cmd_gui)

    # legacy
    run = add("run", "The original character simulation (unrelated to the assistant).")
    run.add_argument("--ticks", type=int, default=10)
    run.add_argument("--policy", default="hybrid", choices=["heuristic", "llm", "hybrid"])
    run.add_argument("--seed", type=int)
    run.add_argument("--db")
    run.add_argument("--offline", action="store_true")
    run.add_argument("--json", action="store_true")
    run.set_defaults(handler=_cmd_run)

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
