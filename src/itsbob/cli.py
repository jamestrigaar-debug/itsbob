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
from typing import Any, NamedTuple, Sequence

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


class ProbeResult(NamedTuple):
    """What one real request to one model did."""

    ok: bool
    detail: str
    latency_ms: float
    #: "ok" | "auth" | "model" | "rate" | "network" — carried from the exception
    #: type rather than re-derived from the message, so the summary does not
    #: have to guess at what went wrong from prose.
    kind: str


def _probe_provider(provider, model: str) -> ProbeResult:
    import time

    from .llm.base import (
        BadRequest,
        LLMRequest,
        ProviderNotConfigured,
        RateLimited,
        user,
    )
    from .llm.providers import vendor_message

    started = time.perf_counter()
    try:
        response = provider.complete(
            LLMRequest(
                messages=[user("Reply with the single word: ready")],
                max_tokens=1000,
                temperature=0.0,
            ),
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - the failure is the result here
        kind = (
            "auth" if isinstance(exc, ProviderNotConfigured)
            else "rate" if isinstance(exc, RateLimited)
            else "model" if isinstance(exc, BadRequest)
            else "network"
        )
        elapsed = (time.perf_counter() - started) * 1000
        return ProbeResult(False, vendor_message(exc), elapsed, kind)
    return ProbeResult(True, (response.text or "").strip()[:24], response.latency_ms, "ok")


def _probe_local(config: Any) -> tuple[bool, str]:
    """Make one real call to Ollama and report what came back.

    "Reachable" and "answering" are different claims, and only the second one
    means cheap turns are free. A model that is pulled but wedged, or one whose
    first load takes ninety seconds, both pass a liveness probe and both send
    every turn to a paid API — silently, which is the problem.
    """
    from .llm.base import LLMRequest, user
    from .llm.local import OllamaProvider

    request = LLMRequest(
        messages=[user("Reply with the single word: ready")],
        max_tokens=16,
        temperature=0.0,
        metadata={"timeout": 30.0},
    )
    try:
        response = OllamaProvider(config).complete_with_fallback(request)
    except Exception as exc:  # noqa: BLE001 - the message is the diagnosis
        return False, f"{type(exc).__name__}: {exc}"[:200]
    text = response.text.strip().replace("\n", " ")[:60]
    if not text:
        return False, f"{response.model} returned an empty reply"
    return True, f"{response.model} in {response.latency_ms:.0f}ms — {text!r}"


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .agent.brain import build_brain
    from .llm.embeddings import default_embedder
    from .llm.local import default_ollama_config, is_ollama_running, list_ollama_models
    from .router.tiers import Tier

    home = _home(args)
    print(f"home: {home}  ({'exists' if home.exists() else 'will be created'})\n")

    brain = build_brain(Settings.from_env(load_env_files=False))
    described = brain.describe()["tiers"]
    ollama_up = is_ollama_running()

    # Probe first, so the tier summary can report what actually answers rather
    # than what merely has a key. Reporting "ok" for a configured-but-rejected
    # key is the exact mistake this tool exists to prevent, and it made itself
    # once already.
    results: dict[tuple[str, str], ProbeResult] = {}
    if args.probe:
        print("probing every configured model (one real request each):")
        seen: set[tuple[str, str]] = set()
        for tier_value in described:
            router = brain.router_for(Tier(tier_value))
            if router is None:
                continue
            for provider in router.providers:
                if not provider.is_configured():
                    continue
                for model in provider.models:
                    if (provider.name, model) in seen:
                        continue
                    seen.add((provider.name, model))
                    outcome = _probe_provider(provider, model)
                    results[(provider.name, model)] = outcome
                    mark = "ok" if outcome.ok else "!!"
                    shown = (
                        f"{outcome.latency_ms:5.0f}ms  {outcome.detail!r}"
                        if outcome.ok
                        else f"[{outcome.kind}] {outcome.detail[:110]}"
                    )
                    print(f"  {mark} {provider.name:<11} {model:<28} {shown}")
        print()

    print("model tiers:")
    for tier_value, info in sorted(described.items(), key=lambda kv: Tier(kv[0]).rank):
        if tier_value == Tier.C.value and ollama_up:
            print(f"  ok Tier C ({info['label']:<15}) ollama (local) — preferred when running")
            continue
        answered = None
        configured = [row for row in info["providers"] if row["configured"]]
        if args.probe:
            for row in configured:
                for model in row["models"]:
                    outcome = results.get((row["provider"], model))
                    if outcome is not None and outcome.ok:
                        answered = (row["provider"], model)
                        break
                if answered:
                    break
            if answered:
                print(f"  ok Tier {tier_value} ({info['label']:<15}) {answered[0]}: {answered[1]}")
            elif configured:
                print(f"  !! Tier {tier_value} ({info['label']:<15}) configured, but nothing answered")
            else:
                print(f"  -- Tier {tier_value} ({info['label']:<15}) nothing configured")
        elif configured:
            first = configured[0]
            models = ", ".join(first["models"][:2])
            print(f"  ?  Tier {tier_value} ({info['label']:<15}) {first['provider']}: {models}  (not checked)")
        else:
            print(f"  -- Tier {tier_value} ({info['label']:<15}) nothing configured")

    print("\nkeys as stored (compare these against the provider's console):")
    from .setup_wizard import DEFAULT_KEYS, fingerprint, key_looks_wrong

    any_key = False
    for name, *_rest in DEFAULT_KEYS:
        value = os.environ.get(name, "").strip()
        if not value:
            print(f"  --  {name:<20} not set")
            continue
        any_key = True
        warning = key_looks_wrong(name, value)
        # The fingerprint is the point: input is hidden when a key is pasted,
        # so a character lost in the paste is otherwise invisible and shows up
        # only as an unexplained rejection.
        print(f"  {'!!' if warning else 'ok '} {name:<20} {fingerprint(value)}")
        if warning:
            print(f"      {warning}")
    if not any_key:
        print("      none — run `itsbob setup`")

    if not args.probe:
        print("\n  ? means a key is present but has not been used. A key can be set and\n"
              "    still be rejected — run `itsbob doctor --probe` to actually find out.")

    print("\nlocal model (free, private, first refusal on all cheap work):")
    if ollama_up:
        config = default_ollama_config()
        pulled = list_ollama_models()
        print(f"  ok  ollama reachable at {config.base_url}")
        wanted = list(config.models())
        missing = [m for m in wanted if m not in pulled]
        print(f"      pulled: {pulled or '(none)'}")
        if missing == wanted:
            print(f"      !! none of {wanted} pulled — run: ollama pull {wanted[0]}")
        else:
            # Reachable is not the same as answering. This is the check that
            # decides whether cheap turns actually cost nothing, so it makes a
            # real call rather than trusting the liveness probe.
            answer, detail = _probe_local(config)
            mark = "ok " if answer else "!! "
            print(f"  {mark}answered a real request: {detail}")
    else:
        print("  --  not reachable — cheap work goes to the cheapest cloud model instead")
        print("      install it to stop paying for greetings and bookkeeping:")
        print("      https://ollama.com/download  then: ollama pull qwen2.5:1.5b")

    print("\nmemory:")
    store = _open_memory(args)
    stats = store.stats()
    for key, value in stats.items():
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

    print("\nservices:")
    from .integrations.apis import builtin_status
    from .integrations.discord import is_configured as discord_configured
    from .tools.vision import pillow_available
    from .tools.websearch import available_backend

    for row in builtin_status():
        mark = "ok " if row["configured"] else "-- "
        print(f"  {mark}{row['name']:<10} {row['description']}"
              + ("" if row["configured"] else f"  (set {row['key_env']})"))
    mark = "ok " if discord_configured() else "-- "
    print(f"  {mark}{'discord':<10} proactive posting and two-way chat"
          + ("" if discord_configured() else "  (set DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)"))
    backend = available_backend()
    print(f"  ok  {'search':<10} via {backend}"
          + ("" if backend != "duckduckgo-html" else "  (install ddgr for structured results)"))
    if pillow_available():
        print(f"  ok  {'images':<10} pillow installed — large photos are downscaled before upload")
    else:
        print(f"  --  {'images':<10} no pillow — `pip install -e '.[vision]'` to resize before upload")

    print("\nscripts:")
    from .scripts import describe_scripts, load_errors, user_scripts_dir

    for row in describe_scripts():
        if row["source"] == "broken":
            print(f"  !!  {row['name']:<18} did not load: {row.get('error', '')}")
        else:
            print(f"  ok  {row['name']:<18} {len(row['tools'])} tool(s)  [{row['source']}]")
    print(f"      drop a .py file exposing tools() into {user_scripts_dir()} to add more")
    if load_errors:
        print(f"      {len(load_errors)} script(s) failed to load — see above")

    print("\ntasks:")
    tasks = _task_store(args).all()
    print(f"      {len(tasks)} configured, {sum(1 for t in tasks if t.enabled)} enabled")

    if not args.probe:
        return 0

    working = sum(1 for outcome in results.values() if outcome.ok)
    rejected = sorted({
        name for (name, _), outcome in results.items() if outcome.kind == "auth"
    })
    stale = sorted({
        f"{name}/{model}" for (name, model), outcome in results.items() if outcome.kind == "model"
    })

    print()
    print(f"{working} model(s) answered." if working
          else "Nothing answered — itsbob cannot think until at least one model does.")
    for name in rejected:
        env_var = f"{name.upper()}_API_KEY"
        print(
            f"\n  ! {name} is REJECTING your key, so every model on it fails.\n"
            f"    Fix or remove {env_var} in ~/.itsbob/.env — while it is set and\n"
            f"    invalid, itsbob still tries {name} first on every call."
        )
        if name == "google":
            print(
                "    An AI Studio key looks like `AIza…` (39 characters). Get one at\n"
                "    https://aistudio.google.com/apikey — free, no card."
            )
    if stale and working:
        print(f"\n  · retired/unavailable models (harmless, itsbob walks past them): {', '.join(stale[:6])}")

    # With one provider dead, every tier falls through to the same surviving
    # model. It works, but the cost ladder — the reason the tiers exist — is
    # gone, and that is not obvious from a screen full of ticks.
    answering = {name for (name, _), outcome in results.items() if outcome.ok}
    if working and len(answering) == 1 and rejected:
        only = next(iter(answering))
        print(
            f"\n  · every tier is now answered by {only}, so trivial and hard requests\n"
            f"    cost the same. Fixing the rejected key restores the cheap/premium split."
        )
    return 0 if working else 1


def _tier(value: str):
    from .router.tiers import Tier

    return Tier(value)


def _cmd_scripts(args: argparse.Namespace) -> int:
    """The foundation scripts and the tools each one provides."""
    from .scripts import describe_scripts

    for row in describe_scripts():
        print(f"\n{row['name']}  —  {row['summary']}")
        print(f"  run directly:  python -m {row['module']} --help")
        for tool in row["tools"]:
            print(f"  {tool['risk']:<12} {tool['name']:<22} {tool['description'][:78]}")
    print("\nRisk decides what the policy asks about: in guarded mode (the default)")
    print("execute and destructive need a person to say yes, and are refused outright")
    print("when nobody is there to ask.")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """What each provider actually serves today, versus what itsbob asks for."""
    from .llm.catalog import default_provider_configs, list_models

    configs = {c.name: c for c in default_provider_configs()}
    any_key = False

    for name, config in configs.items():
        if args.provider and args.provider != name:
            continue
        wanted = list(config.models())
        if not config.is_configured():
            print(f"\n{name}: no {config.api_key_env} set")
            continue
        any_key = True
        live = list_models(config)
        print(f"\n{name}:")
        if not live:
            print("  could not read the model list (network, or the key is rejected)")
            print(f"  itsbob will try: {', '.join(wanted)}")
            continue
        missing = [m for m in wanted if m not in live]
        print(f"  itsbob will try: {', '.join(wanted)}")
        if missing:
            print(f"  !! not served any more: {', '.join(missing)}")
            print(f"     pin a live one:  export ITSBOB_{name.upper()}_MODEL=<id>")
        chat_like = [m for m in live if "embed" not in m and "whisper" not in m and "tts" not in m]
        shown = chat_like if args.all else chat_like[:25]
        print(f"  {len(live)} model(s) available:")
        for model in shown:
            mark = "*" if model in wanted else " "
            print(f"   {mark} {model}")
        if len(chat_like) > len(shown):
            print(f"     … {len(chat_like) - len(shown)} more (--all to see them)")

    if not any_key:
        print("\nNo provider keys are set. Run `itsbob setup`.")
        return 1
    return 0


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


def _cmd_setup(args: argparse.Namespace) -> int:
    from .setup_wizard import run_setup

    keys = {}
    for name in ("google", "groq", "openrouter"):
        value = getattr(args, f"{name}_key", None)
        if value:
            keys[f"{name.upper()}_API_KEY"] = value.strip()
    return run_setup(
        home=_home(args),
        keys=keys or None,
        verify=not args.no_verify,
        interactive=sys.stdin.isatty() and not keys,
    )


def _cmd_service(args: argparse.Namespace) -> int:
    from .service import install_service, service_status, uninstall_service, unit_text

    if args.service_action == "status":
        print(f"itsbob daemon service: {service_status()}")
        return 0
    if args.service_action == "print":
        print(unit_text(_home(args), mode=getattr(args, "mode", None)), end="")
        return 0
    if args.service_action == "uninstall":
        ok, message = uninstall_service()
        print(message)
        return 0 if ok else 1

    ok, message = install_service(
        _home(args), mode=getattr(args, "mode", None), start=not args.no_start
    )
    print(("  " if ok else "error: ") + message)
    if ok:
        print("\n  It will now run in the background and survive a reboot.")
        print("  Add work for it with:  itsbob task add <name> <what to do> <schedule>")
    return 0 if ok else 1


def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui.app import run_gui

    host = "0.0.0.0" if getattr(args, "public", False) else args.host  # noqa: S104 - opt-in
    if host != "127.0.0.1":
        print(
            "  ! binding to "
            f"{host} — the interface has no authentication, so anyone who can reach\n"
            "    this port can run tools as you. Only do this on a network you trust.\n"
        )
    run_gui(
        host=host,
        port=args.port,
        open_browser=not args.no_browser,
        home=_home(args),
        mode=getattr(args, "mode", None),
    )
    return 0


# --------------------------------------------------------------------------
# legacy: the character simulation
# --------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from .factory import build_simulation

    settings = Settings.from_env(load_env_files=False)
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
            "  itsbob setup                first run: keys and a working check\n"
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

    # setup
    setup = add("setup", "Get set up: keys, directories, and a check that it all works.")
    setup.add_argument("--google-key", help="set GOOGLE_API_KEY without being prompted")
    setup.add_argument("--groq-key", help="set GROQ_API_KEY without being prompted")
    setup.add_argument("--openrouter-key", help="set OPENROUTER_API_KEY without being prompted")
    setup.add_argument("--no-verify", action="store_true", help="skip the live API check")
    setup.set_defaults(handler=_cmd_setup)

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

    add("scripts", "The foundation scripts itsbob can run, and what each may do.").set_defaults(
        handler=_cmd_scripts
    )

    models = add("models", "What each provider actually serves, versus what itsbob asks for.")
    models.add_argument("--provider", choices=["google", "groq", "openrouter"])
    models.add_argument("--all", action="store_true", help="list every model, not the first 25")
    models.set_defaults(handler=_cmd_models)

    with_mode(add("tools", "Tools available, and the current policy.")).set_defaults(
        handler=_cmd_tools
    )

    audit = add("audit", "Recent tool activity, including refusals.")
    audit.add_argument("--limit", type=int, default=40)
    audit.set_defaults(handler=_cmd_audit)

    # service
    service = add("service", "Run the daemon in the background, across reboots.")
    service_subs = service.add_subparsers(dest="service_action", required=True)
    install = with_mode(service_subs.add_parser("install", help="install and start it"))
    install.add_argument("--no-start", action="store_true", help="write the unit but do not start")
    install.set_defaults(handler=_cmd_service)
    service_subs.add_parser("uninstall", help="stop and remove it").set_defaults(handler=_cmd_service)
    service_subs.add_parser("status", help="is it running?").set_defaults(handler=_cmd_service)
    with_mode(service_subs.add_parser("print", help="show the unit file without installing")).set_defaults(
        handler=_cmd_service
    )

    gui = with_mode(add("gui", "Browser interface."))
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true")
    gui.add_argument("--public", action="store_true",
                     help="bind to 0.0.0.0 — there is NO authentication, so only on a trusted network")
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
