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

__all__ = ["run_setup", "write_env", "verify_key", "key_looks_wrong", "DEFAULT_KEYS"]

#: Recognised providers, in the order they are offered. Google first because
#: one key covers all three tiers, which nothing else does.
DEFAULT_KEYS = (
    ("GOOGLE_API_KEY", "Google AI Studio", "https://aistudio.google.com/apikey", True),
    ("GROQ_API_KEY", "Groq", "https://console.groq.com/keys", False),
    ("OPENROUTER_API_KEY", "OpenRouter", "https://openrouter.ai/keys", False),
)

_TICK, _CROSS, _DOT, _WARN = "✓", "✗", "·", "!"

#: What a real key from each vendor looks like. Checked before spending an API
#: call, because "that is not a key for this service" is a much more useful
#: thing to be told than a 400 three seconds later — and because the shapes are
#: distinctive enough that a mismatch is almost never a false alarm. A warning,
#: never a refusal: vendors change formats and a stale check must not lock
#: anyone out of their own working key.
_KEY_SHAPES = {
    "GOOGLE_API_KEY": (
        # Google issues two API key formats: the long-standing `AIza…` (39
        # chars) and a newer project-scoped `AQ.…` (~53). Both are real keys.
        # `ya29.…` is an OAuth2 access token — a genuinely different thing,
        # short-lived, and rejected.
        lambda k: (k.startswith("AIza") and len(k) >= 35) or (k.startswith("AQ.") and len(k) >= 40),
        "Google API keys look like `AIza…` (39 characters) or `AQ.…` (about 53). "
        "A value starting `ya29.` is a short-lived OAuth access token, not an API key.",
    ),
    "GROQ_API_KEY": (
        lambda k: k.startswith("gsk_"),
        "Groq keys start with `gsk_`.",
    ),
    "OPENROUTER_API_KEY": (
        lambda k: k.startswith("sk-or-"),
        "OpenRouter keys start with `sk-or-`.",
    ),
}


def key_looks_wrong(name: str, key: str) -> str | None:
    """A warning if the key is not shaped like one for this vendor, else None."""
    shape = _KEY_SHAPES.get(name)
    if shape is None or not key.strip():
        return None
    matches, explanation = shape
    return None if matches(key.strip()) else explanation


def _say(message: str = "") -> None:
    print(message, flush=True)


def _ask(prompt: str, *, secret: bool = False, default: str = "") -> str:
    if not sys.stdin.isatty():
        return default
    try:
        if secret:
            import getpass

            return _clean_key(getpass.getpass(prompt))
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        _say()
        return default


def _clean_key(raw: str) -> str:
    """Undo the damage a terminal paste does before it reaches the API.

    Copying from a web console routinely brings along a trailing newline, a
    stray space, wrapping quotes, or a `KEY=` prefix from a copied env line.
    Each of those produces a rejected key that looks perfectly fine, so they
    are stripped here rather than sent and puzzled over.
    """
    key = raw.strip().strip("\"'").strip()
    prefixes = ("export ", "GOOGLE_API_KEY=", "GROQ_API_KEY=", "OPENROUTER_API_KEY=")
    # Repeat until nothing matches: `export GOOGLE_API_KEY=…` needs two passes,
    # and a single pass silently left half the prefix attached.
    for _ in range(4):
        stripped = key
        for prefix in prefixes:
            if stripped.lower().startswith(prefix.lower()):
                stripped = stripped[len(prefix):].strip().strip("\"'").strip()
        if stripped == key:
            break
        key = stripped
    # Internal whitespace is never part of a key; it is a wrapped paste.
    return "".join(key.split())


def fingerprint(key: str) -> str:
    """Enough of a key to check it against the console, not enough to leak it."""
    key = key.strip()
    if len(key) < 12:
        return f"{key[:2]}… ({len(key)} chars)"
    return f"{key[:6]}…{key[-4:]} ({len(key)} chars)"


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
                # Echo a fingerprint. The input is hidden, so a paste that lost
                # a character or gained one is otherwise invisible — and that
                # produces a key the API rejects for no visible reason.
                _say(f"     {_DOT} got {fingerprint(value)} — check that against the console")
                warning = key_looks_wrong(name, value)
                if warning:
                    _say(f"     {_WARN} That does not look like a {label} key.")
                    _say(f"       {warning}")
                    if not _confirm("       Use it anyway?", default=False):
                        _say(f"     {_DOT} skipped — get one at {url}")
                        continue
                collected[name] = value
            elif primary:
                _say(f"     {_DOT} skipped — itsbob will run offline until you add one")

    if collected:
        target = write_env(collected)
        for key, value in collected.items():
            os.environ[key] = value
        _say()
        _say(f"  {_TICK} wrote {len(collected)} key(s) to {target} (mode 600)")
        # Also flagged here, not only on rejection: a non-interactive run
        # (--google-key ...) never saw the prompt, and a key that happens to be
        # accepted today through some proxy is still the wrong kind of key.
        for key, value in collected.items():
            warning = key_looks_wrong(key, value)
            if warning:
                _say(f"  {_WARN} {key} does not look like a key for that service.")
                _say(f"    {warning}")

    # 3. Verification — the only step that proves anything.
    configured = [name for name, *_ in DEFAULT_KEYS if os.environ.get(name, "").strip()]
    working = 0
    rejected: list[str] = []
    if verify and configured:
        _say()
        _say("  Checking each key against the real API…")
        for name in configured:
            for attempt in range(3):
                key = os.environ[name]
                ok, detail = verify_key(name, key)
                _say(f"  {_TICK if ok else _CROSS} {name:<20} {detail}")
                if ok:
                    working += 1
                    break

                _say(f"    the key it tried was {fingerprint(key)}")
                warning = key_looks_wrong(name, key)
                if warning:
                    _say(f"    {_WARN} {warning}")
                elif "auth" in detail.lower() or "valid" in detail.lower():
                    _say(
                        "    The format is right, so this is the key itself: a character lost\n"
                        "    in the paste, a revoked key, or one from a project without the\n"
                        "    Generative Language API enabled."
                    )
                if not (interactive and attempt < 2):
                    rejected.append(name)
                    break
                if not _confirm("    Paste it again?", default=True):
                    rejected.append(name)
                    break
                retyped = _ask(f"     {name}: ", secret=True)
                if not retyped:
                    rejected.append(name)
                    break
                _say(f"     {_DOT} got {fingerprint(retyped)}")
                os.environ[name] = retyped
                write_env({name: retyped})
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
        _say(f"  {_CROSS} Keys are set but none answered — itsbob cannot think yet.")
        for name in rejected:
            url = next((u for n, _, u, _ in DEFAULT_KEYS if n == name), "")
            _say(f"    {name} was rejected. Get a working key at {url}")
        _say("    Then re-run `itsbob setup`, or `itsbob doctor --probe` for the full errors.")
    else:
        _say(f"  {_DOT} Set up, but with no model configured.")
    _say()
    _say("  Try:")
    _say("    itsbob chat                 talk to it")
    _say("    itsbob gui                  the browser interface")
    _say('    itsbob task add morning "Summarise my day" "weekdays at 08:30"')
    _say("    itsbob serve                let it work on its own")
    _say()

    # Deliberately no "open the browser now?" prompt: it lands at the very end
    # of an installer, when people are already typing their next command, and
    # swallows it.
    return 0 if (working or not configured) else 1
