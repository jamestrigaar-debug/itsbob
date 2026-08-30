"""Command line entry point.

    itsbob doctor            # which providers actually answer right now
    itsbob run --ticks 20    # play the simulation
    itsbob ask "question"    # one-shot router call
    itsbob memory            # dump long-term memory from a saved run
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .config import Settings, load_dotenv
from .factory import build_router, build_simulation
from .llm.base import AllProvidersFailed, LLMRequest, user
from .llm.local import is_ollama_running
from .memory.long_term import LongTermMemory
from .router import build_complexity_router

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    load_dotenv(args.env_file)

    handler = {
        "run": _cmd_run,
        "doctor": _cmd_doctor,
        "ask": _cmd_ask,
        "memory": _cmd_memory,
        "classify": _cmd_classify,
        "route": _cmd_route,
        "gui": _cmd_gui,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itsbob", description="Character simulation with metered LLM access."
    )
    parser.add_argument(
        "--env-file", default=".env", help="dotenv file to load (default: .env)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the simulation")
    run.add_argument("--ticks", type=int, default=20)
    run.add_argument("--name", default="Bob")
    run.add_argument(
        "--policy",
        default="hybrid",
        choices=("heuristic", "llm", "hybrid"),
        help="hybrid deliberates only when it can afford to (default)",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--db", default=None, help="persist long-term memory here")
    run.add_argument(
        "--offline",
        action="store_true",
        help="ignore API keys and run on the deterministic echo provider",
    )
    run.add_argument("--json", action="store_true", help="emit JSON instead of narration")
    run.add_argument("--quiet", action="store_true", help="summary only")

    doctor = sub.add_parser("doctor", help="check provider connectivity")
    doctor.add_argument(
        "--probe", action="store_true", help="actually call each provider (uses quota)"
    )

    ask = sub.add_parser("ask", help="send one prompt through the router")
    ask.add_argument("prompt")
    # Reasoning models spend part of this budget thinking before they emit a
    # single visible token, so a small value truncates the answer.
    ask.add_argument("--max-tokens", type=int, default=800)
    ask.add_argument("--temperature", type=float, default=0.7)
    ask.add_argument("--provider", default=None, help="restrict to one provider")

    memory = sub.add_parser("memory", help="inspect a saved long-term memory database")
    memory.add_argument("db")
    memory.add_argument("--query", default=None)
    memory.add_argument("--limit", type=int, default=20)

    classify = sub.add_parser(
        "classify", help="run only the Gatekeeper: tier + fingerprint, no execution"
    )
    classify.add_argument("state", help="JSON game state, or @path/to/file.json")

    route = sub.add_parser(
        "route", help="full complexity-router pipeline: classify, route, execute"
    )
    route.add_argument("state", help="JSON game state, or @path/to/file.json")
    route.add_argument("--goal", default="win the league")

    gui = sub.add_parser("gui", help="launch the browser GUI (requires the 'gui' extra)")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument(
        "--no-browser", action="store_true", help="don't auto-open a browser tab"
    )

    return parser


def _load_state_arg(raw: str) -> str:
    if raw.startswith("@"):
        return open(raw[1:], encoding="utf-8").read()
    return raw


# --------------------------------------------------------------------------


def _settings_for(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(dotenv=None)
    overrides: dict[str, Any] = {}
    if getattr(args, "db", None):
        overrides["memory"] = settings.memory.with_database(args.db)
    if getattr(args, "offline", False):
        overrides["providers"] = ()
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if not overrides:
        return settings
    from dataclasses import replace

    return replace(settings, **overrides)


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _settings_for(args)
    sim = build_simulation(
        settings, name=args.name, policy=args.policy, seed=args.seed
    )

    if not args.quiet:
        providers = ", ".join(sim.router.provider_names()) if sim.router else "none"
        print(f"providers: {providers}   policy: {args.policy}   ticks: {args.ticks}")
        print(f"{sim.character.name}: {sim.character.goal}\n")

    records: list[dict[str, Any]] = []
    for report in sim.stream(args.ticks):
        if args.json:
            records.append(report.as_dict())
        elif not args.quiet:
            print(report.line())
    sim.finish()

    if args.json:
        print(json.dumps({"ticks": records, "summary": sim.summary()}, indent=2))
    else:
        print("\n--- summary ---")
        print(json.dumps(sim.summary(), indent=2))
    sim.character.close()
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = Settings.from_env(dotenv=None)
    router = build_router(settings)

    print("configured providers (in try-order):")
    for row in router.describe():
        mark = "ok " if row["configured"] else "-- "
        print(f"  {mark}{row['provider']:<12} rpm={row['rpm']:<4} models={row['models']}")

    print("\nlocal Back Brain (Tier C, the Gatekeeper's classifier engine):")
    if is_ollama_running():
        print("  ok  ollama       reachable at http://127.0.0.1:11434")
    else:
        print(
            "  --  ollama       not reachable — Tier C/gatekeeper classification "
            "falls back to the rule-based heuristic (see README: Local Back Brain)"
        )

    if not args.probe:
        print("\n(pass --probe to send a real one-token request to each)")
        return 0

    print("\nprobing:")
    answered = 0
    for provider in router.providers:
        if not provider.is_configured():
            print(f"  -- {provider.name:<12} no API key")
            continue
        for model in provider.models:
            request = LLMRequest(
                messages=[user("Reply with the single word: ready")],
                max_tokens=256,
                temperature=0.0,
            )
            try:
                response = provider.complete(request, model=model)
            except Exception as exc:
                print(f"  !! {provider.name:<12} {model:<45} {type(exc).__name__}: {exc}"[:160])
                continue
            print(
                f"  ok {provider.name:<12} {model:<45} "
                f"{response.latency_ms:6.0f}ms  {response.text.strip()[:40]!r}"
            )
            answered += 1
            break  # the first working model per provider is enough

    # Non-zero only if nothing at all answered — a partly-degraded set of free
    # providers is the normal state, not a failure.
    return 0 if answered else 1


def _cmd_ask(args: argparse.Namespace) -> int:
    router = build_router(Settings.from_env(dotenv=None))
    request = LLMRequest(
        messages=[user(args.prompt)],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    try:
        response = router.complete(
            request,
            purpose="cli",
            providers=[args.provider] if args.provider else None,
        )
    except AllProvidersFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(response.text)
    print(
        f"\n[{response.provider}/{response.model} · {response.usage.total_tokens} tokens "
        f"· {response.latency_ms:.0f}ms]",
        file=sys.stderr,
    )
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    store = LongTermMemory(args.db)
    records = (
        store.recall(args.query, limit=args.limit)
        if args.query
        else store.all(limit=args.limit)
    )
    print(f"{len(store)} memories in {args.db}\n")
    for record in records:
        print(f"  [{record.importance:.2f}] {record.render()}")
    store.close()
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from .router import Gatekeeper, compress, default_registry
    from .llm.local import OllamaProvider

    state = compress(_load_state_arg(args.state))
    local = OllamaProvider() if is_ollama_running() else None
    gatekeeper = Gatekeeper(registry=default_registry(), local_provider=local)
    decision = gatekeeper.classify(state)
    print(json.dumps(decision.as_dict(), indent=2))
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    settings = Settings.from_env(dotenv=None)
    router = build_complexity_router(settings, goal=args.goal)
    result = router.route(_load_state_arg(args.state))
    print(json.dumps(result.as_dict(), indent=2))
    return 0 if result.ok else 1


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui.app import run_gui
    except ImportError as exc:
        print(
            "error: the GUI needs Flask — install it with `pip install -e \".[gui]\"`\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 1
    run_gui(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
