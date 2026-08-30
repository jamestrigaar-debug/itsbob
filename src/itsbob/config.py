"""Environment-driven configuration.

Nothing here reaches the network; it just decides *which* providers are
plausible and how the simulation is sized. Missing API keys are not an error —
the router drops those providers and carries on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, MutableMapping

__all__ = [
    "ProviderConfig",
    "MemorySettings",
    "EnergySettings",
    "Settings",
    "find_dotenv",
    "itsbob_home",
    "load_dotenv",
]


def itsbob_home(env: Mapping[str, str] | None = None) -> Path:
    """Where itsbob keeps its state. ``ITSBOB_HOME``, else ``~/.itsbob``."""
    env = os.environ if env is None else env
    return Path(env.get("ITSBOB_HOME", "").strip() or Path.home() / ".itsbob").expanduser()


def find_dotenv(start: str | Path | None = None) -> list[Path]:
    """Every ``.env`` worth loading, nearest first.

    Looks in the current directory, then each parent up to the filesystem root,
    then ``$ITSBOB_HOME/.env``.

    This exists because loading only ``./.env`` made the whole system look
    broken from anywhere but the source checkout: ``itsbob serve`` started from
    a home directory saw no API keys, silently fell through to the offline
    provider, and answered with plausible nonsense. A daemon is *usually*
    started from somewhere else, so the default was wrong exactly where it
    mattered most.

    Nearest wins, so a project-local ``.env`` overrides the global one — the
    same precedence git and npm use, and the one people already expect.
    """
    found: list[Path] = []
    here = Path(start).expanduser().resolve() if start else Path.cwd()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            found.append(candidate)
    home_env = itsbob_home() / ".env"
    if home_env.is_file() and home_env not in found:
        found.append(home_env)
    return found


def load_dotenv(
    path: str | Path | None = None,
    *,
    override: bool = False,
    env: MutableMapping[str, str] | None = None,
    search: bool = True,
) -> dict[str, str]:
    """Load ``KEY=value`` files into the environment.

    With no ``path``, loads every file :func:`find_dotenv` turns up, nearest
    first. Because a value is only applied when the key is not already set, the
    nearest file wins and the rest fill gaps — so a project ``.env`` overrides
    ``~/.itsbob/.env`` without either having to know about the other.

    Deliberately dependency-free and forgiving: blank lines and ``#`` comments
    are skipped, a leading ``export`` is tolerated, and surrounding quotes are
    stripped. Returns everything it parsed, whether or not it was applied.
    """
    env = os.environ if env is None else env

    if path is None:
        if not search:
            return {}
        merged: dict[str, str] = {}
        for candidate in find_dotenv():
            for key, value in _read_dotenv(candidate).items():
                merged.setdefault(key, value)
                if override or key not in env:
                    env[key] = value
        return merged

    parsed = _read_dotenv(Path(path))
    for key, value in parsed.items():
        if override or key not in env:
            env[key] = value
    return parsed


def _read_dotenv(p: Path) -> dict[str, str]:
    """Parse one file. Missing or unreadable is empty, never an error."""
    try:
        if not p.is_file():
            return {}
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}

    parsed: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        parsed[key] = value
    return parsed


@dataclass(frozen=True)
class ProviderConfig:
    """How to reach one LLM vendor.

    All three supported vendors expose an OpenAI-compatible ``/chat/completions``
    endpoint, so a single provider implementation covers them; only these
    values differ.
    """

    name: str
    base_url: str
    api_key_env: str
    default_model: str
    fallback_models: tuple[str, ...] = ()
    #: Free tiers are rate limited far more often than they are token limited.
    requests_per_minute: int = 20
    #: Whether a 429 exhausts the whole account or just the model that hit it.
    #: Gemini meters per model, so a quota error there leaves siblings usable;
    #: Groq and OpenRouter meter per account, so moving on is the only option.
    rate_limit_scope: str = "account"  # "account" | "model"
    #: Wall-clock timeout for a single request, in seconds.
    timeout: float = 45.0
    #: Extra HTTP headers (OpenRouter uses these for attribution).
    headers: Mapping[str, str] = field(default_factory=dict)
    enabled: bool = True

    def api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if env is None else env
        key = env.get(self.api_key_env, "").strip()
        return key or None

    def is_configured(self, env: Mapping[str, str] | None = None) -> bool:
        return self.enabled and self.api_key(env) is not None

    def models(self) -> tuple[str, ...]:
        """Preferred model first, then per-provider fallbacks, deduplicated."""
        seen: dict[str, None] = {}
        for model in (self.default_model, *self.fallback_models):
            if model:
                seen.setdefault(model, None)
        return tuple(seen)


@dataclass(frozen=True)
class MemorySettings:
    #: How many records stay in the working set before eviction.
    short_term_capacity: int = 12
    #: Fraction of salience a short-term record loses each tick.
    short_term_decay: float = 0.12
    #: Salience below which a record is dropped rather than kept.
    salience_floor: float = 0.15
    #: Evicted records at or above this importance are written to long-term.
    promotion_threshold: float = 0.55
    #: Ticks between LLM-backed reflection passes (0 disables reflection).
    reflection_interval: int = 12
    #: How many memories a single recall returns.
    recall_limit: int = 5
    database: str = ":memory:"

    def with_database(self, database: str | Path) -> "MemorySettings":
        return replace(self, database=str(database))


@dataclass(frozen=True)
class EnergySettings:
    capacity: float = 100.0
    starting: float = 100.0
    regen_per_tick: float = 6.0
    #: Below this the character is "exhausted": deliberation is off the table.
    exhaustion_threshold: float = 15.0
    #: Flat energy price of any LLM round trip, before token cost.
    call_overhead: float = 2.0
    #: How many LLM tokens equate to one point of energy.
    tokens_per_energy: float = 400.0
    #: Energy price of running the LLM decision policy for one tick.
    deliberation_cost: float = 4.0


@dataclass(frozen=True)
class Settings:
    providers: tuple[ProviderConfig, ...]
    memory: MemorySettings = field(default_factory=MemorySettings)
    energy: EnergySettings = field(default_factory=EnergySettings)
    #: Fall back to the offline EchoProvider when no real provider is configured.
    allow_offline: bool = True
    #: Total provider attempts per logical LLM call, across all providers.
    max_attempts: int = 4
    seed: int | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        dotenv: str | Path | None = None,
        load_env_files: bool = True,
    ) -> "Settings":
        if env is None and load_env_files:
            load_dotenv(dotenv)
        env = os.environ if env is None else env

        from .llm.catalog import default_provider_configs

        providers = default_provider_configs(env)
        order = _split_csv(env.get("ITSBOB_PROVIDER_ORDER", ""))
        if order:
            rank = {name: i for i, name in enumerate(order)}
            providers = tuple(
                sorted(providers, key=lambda p: rank.get(p.name, len(rank)))
            )

        memory = MemorySettings(
            short_term_capacity=_int(env, "ITSBOB_SHORT_TERM_CAPACITY", 12),
            reflection_interval=_int(env, "ITSBOB_REFLECTION_INTERVAL", 12),
            database=env.get("ITSBOB_MEMORY_DB", ":memory:"),
        )
        energy = EnergySettings(
            capacity=_float(env, "ITSBOB_ENERGY_CAPACITY", 100.0),
            starting=_float(env, "ITSBOB_ENERGY_START", 100.0),
            regen_per_tick=_float(env, "ITSBOB_ENERGY_REGEN", 6.0),
        )
        seed_raw = env.get("ITSBOB_SEED", "").strip()
        return cls(
            providers=providers,
            memory=memory,
            energy=energy,
            allow_offline=_bool(env, "ITSBOB_ALLOW_OFFLINE", True),
            max_attempts=_int(env, "ITSBOB_MAX_ATTEMPTS", 4),
            seed=int(seed_raw) if seed_raw else None,
        )

    def configured_providers(
        self, env: Mapping[str, str] | None = None
    ) -> tuple[ProviderConfig, ...]:
        return tuple(p for p in self.providers if p.is_configured(env))


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, "").strip() or default)
    except ValueError:
        return default


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, "").strip() or default)
    except ValueError:
        return default


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
