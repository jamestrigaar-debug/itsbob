"""``itsbob setup`` — from a fresh clone to a working assistant, once.

The old path was: create a venv, pip install, work out which env vars exist,
create a ``.env``, discover that it is only read from one directory, and find
out whether any of it worked by reading a wall of ``doctor`` output. Every step
was a place to stop.

This does the whole thing and *verifies* it with a real API call, because the
only useful answer to "is it set up?" is one that has actually talked to a
model. A key that is present but rejected is worse than no key at all: it looks
configured and behaves as though it isn't.

Keys are written to ``$ITSBOB_HOME/.env`` rather than the working directory, so
the daemon and the GUI find them no matter where they are started from.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Mapping

from .config import find_dotenv, itsbob_home, load_dotenv

__all__ = ["run_setup", "write_env", "verify_key", "DEFAULT_KEYS"]

#: Recognised providers, in the order they are offered. Google first because
#: one key covers all three tiers, which nothing else does.
DEFAULT_KEYS = (
    ("GOOGLE_API_KEY", "Google AI Studio", "https://aistudio.google.com/apikey", True),
    ("GROQ_API_KEY", "Groq", "https://console.groq.com/keys", False),
    ("OPENROUTER_API_KEY", "OpenRouter", "https://openrouter.ai/keys", False),
)

_TICK, _CROSS, _DOT = "✓", "✗", "·"


def _say(message: str = "") -> None:
    print(message, flush=True)


def _ask(prompt: str, *, secret: bool = False, default: str = "") -> str:
    if not sys.stdin.isatty():
        return default
    try:
        if secret:
            import getpass

            return getpass.getpass(prompt).strip()
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        _say()
        return default


def _confirm(prompt: str, *, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    answer = _ask(f"{prompt} {suffix} ").lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def verify_key(name: str, key: str, *, timeout: float = 30.0) -> tuple[bool, str]:
    """Send one real request. Returns (ok, human-readable detail).

    Deliberately a live call. "The variable is set" and "the key works" are
    different claims, and only the second is worth telling someone.
    """
    from .llm.base import LLMRequest, user
    from .llm.catalog import GOOGLE, GROQ, OPENROUTER
    from .llm.providers import build_provider

    config = {"GOOGLE_API_KEY": GOOGLE, "GROQ_API_KEY": GROQ, "OPENROUTER_API_KEY": OPENROUTER}.get(name)
    if config is None:
        return False, f"unknown provider for {name}"

    provider = build_provider(config, {**os.environ, name: key})
    errors: list[str] = []
    for model in provider.models:
        try:
            response = provider.complete(
                LLMRequest(
                    messages=[user("Reply with the single word: ready")],
                    max_tokens=1000,
                    temperature=0.0,
                ),
                model=model,
            )
        except Exception as exc:  # noqa: BLE001 - try the next model
            errors.append(f"{model}: {type(exc).__name__}")
            continue
        return True, f"{model} answered in {response.latency_ms:.0f}ms"
    return False, "; ".join(errors[:3]) or "no model answered"


def write_env(values: Mapping[str, str], *, path: Path | None = None) -> Path:
    """Merge ``values`` into ``$ITSBOB_HOME/.env``, preserving what is there.

    The file is written 0600. It holds API keys, and a world-readable file of
    API keys in a home directory is the kind of thing nobody looks at twice
    until it matters.
    """
    target = Path(path) if path else itsbob_home() / ".env"
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, str] = {}
    order: list[str] = []
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            existing[key] = value.strip()
            order.append(key)

    for key, value in values.items():
        if key not in existing:
            order.append(key)
        existing[key] = value

    body = ["# Written by `itsbob setup`. Keys are read from here wherever itsbob runs.", ""]
    body += [f"{key}={existing[key]}" for key in dict.fromkeys(order) if existing.get(key)]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")
    try:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - filesystem without permissions
        pass
    return target


def run_setup(
    *,
    home: Path | None = None,
    keys: Mapping[str, str] | None = None,
    verify: bool = True,
    interactive: bool = True,
) -> int:
    """The wizard. Returns a process exit code."""
    root = Path(home).expanduser() if home else itsbob_home()
    load_dotenv()

    _say()
    _say("  itsbob setup")
    _say("  ───────────")
    _say(f"  Everything lives in {root}")
    _say()

    # 1. Directories, before anything can fail — so a partial setup still
    #    leaves a usable home rather than nothing.
    for name in ("", "workspace"):
        (root / name).mkdir(parents=True, exist_ok=True)
    _say(f"  {_TICK} created {root}/ and workspace/")

    for found in find_dotenv():
        _say(f"  {_DOT} reading existing {found}")

    # 2. Keys.
    collected: dict[str, str] = dict(keys or {})
    if interactive and not collected:
        _say()
        _say("  A Google AI Studio key is all you need — one key covers all three")
        _say("  model tiers. It is free and needs no card.")
        _say()
        for name, label, url, primary in DEFAULT_KEYS:
            current = os.environ.get(name, "").strip()
            if current:
                _say(f"  {_TICK} {label}: already set ({current[:6]}…)")
                continue
            if not primary and not _confirm(f"  Add a {label} key as a backup?", default=False):
                continue
            _say(f"     Get one at {url}")
            value = _ask(f"     Paste your {name} (or press Enter to skip): ", secret=True)
            if value:
                collected[name] = value
            elif primary:
                _say(f"     {_DOT} skipped — itsbob will run offline until you add one")

    if collected:
        target = write_env(collected)
        for key, value in collected.items():
            os.environ[key] = value
        _say()
        _say(f"  {_TICK} wrote {len(collected)} key(s) to {target} (mode 600)")

    # 3. Verification — the only step that proves anything.
    configured = [name for name, *_ in DEFAULT_KEYS if os.environ.get(name, "").strip()]
    working = 0
    if verify and configured:
        _say()
        _say("  Checking each key against the real API…")
        for name in configured:
            ok, detail = verify_key(name, os.environ[name])
            mark = _TICK if ok else _CROSS
            _say(f"  {mark} {name:<20} {detail}")
            working += bool(ok)
    elif not configured:
        _say()
        _say(f"  {_DOT} No keys configured. itsbob will run, but it cannot think:")
        _say("    every answer will tell you to set a key. Re-run `itsbob setup` any time.")

    # 4. Embeddings and the local model — optional, so reported not demanded.
    _say()
    from .llm.local import is_ollama_running

    if is_ollama_running():
        _say(f"  {_TICK} Ollama is running — Tier C will use it (free and private)")
    else:
        _say(f"  {_DOT} Ollama not running (optional). Install it to keep cheap work local:")
        _say("    https://ollama.com/download  then: ollama pull qwen2.5:1.5b")

    # 5. What to do next.
    _say()
    _say("  ───────────")
    if working:
        _say(f"  {_TICK} Ready. {working} provider(s) answering.")
    elif configured:
        _say(f"  {_CROSS} Keys are set but none answered. Run `itsbob doctor --probe` for detail.")
    else:
        _say(f"  {_DOT} Set up, but with no model configured.")
    _say()
    _say("  Try:")
    _say("    itsbob chat                 talk to it")
    _say("    itsbob gui                  the browser interface")
    _say('    itsbob task add morning "Summarise my day" "weekdays at 08:30"')
    _say("    itsbob serve                let it work on its own")
    _say()

    if interactive and working and _confirm("  Open the browser interface now?", default=False):
        from .gui.app import run_gui

        run_gui(home=root)
    return 0 if (working or not configured) else 1
