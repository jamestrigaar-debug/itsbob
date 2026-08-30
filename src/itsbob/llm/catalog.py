"""Which free models to reach for, per vendor.

Free tiers churn — models get renamed, retired, or quietly gated to existing
users. So every id here is a *default*, overridable by env var, and each
provider carries fallbacks the router walks on a 404. ``itsbob doctor``
tells you which ones actually answer today.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Mapping

from ..config import ProviderConfig

__all__ = [
    "OPENROUTER",
    "GROQ",
    "GOOGLE",
    "PROVIDER_TEMPLATES",
    "default_provider_configs",
    "discover_openrouter_free_models",
]


# OpenRouter aggregates dozens of vendors; the ``:free`` suffix is the whole
# free tier. Rate limits are per-account, not per-model.
OPENROUTER = ProviderConfig(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    default_model="meta-llama/llama-3.3-70b-instruct:free",
    fallback_models=(
        "deepseek/deepseek-chat-v3-0324:free",
        "google/gemma-3-27b-it:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "mistralai/mistral-small-3.2-24b-instruct:free",
    ),
    requests_per_minute=20,
)

# Groq's free tier is generous and extremely fast — good default for a
# simulation that makes a call every tick.
#
# Groq deprecates models on its own schedule, independent of this repo — the
# 3.3/3.1 Llama ids below are kept as fallbacks because Groq brings models
# back and reintroduces similarly-named ones often enough that dropping them
# outright just repeats this same churn later. gpt-oss-20b and gemma2-9b-it
# are listed first because they're the two that have proven reliably live
# across multiple `itsbob doctor --probe` runs; if they 404 for you too, run
# `itsbob doctor --probe` and pin whatever answers with `ITSBOB_GROQ_MODEL`
# (see README: Optional tuning env vars).
GROQ = ProviderConfig(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    default_model="openai/gpt-oss-20b",
    fallback_models=(
        "gemma2-9b-it",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ),
    requests_per_minute=30,
)

# Gemini exposes an OpenAI-compatible shim at /v1beta/openai/, so it needs no
# special-case client. Flash-class models are the free-tier workhorses.
GOOGLE = ProviderConfig(
    name="google",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key_env="GOOGLE_API_KEY",
    default_model="gemini-3.6-flash",
    fallback_models=(
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-2.0-flash",
    ),
    requests_per_minute=15,
    rate_limit_scope="model",
)

PROVIDER_TEMPLATES: tuple[ProviderConfig, ...] = (GROQ, GOOGLE, OPENROUTER)

#: Env var that pins the model for a given provider.
_MODEL_OVERRIDE = {
    "openrouter": "ITSBOB_OPENROUTER_MODEL",
    "groq": "ITSBOB_GROQ_MODEL",
    "google": "ITSBOB_GOOGLE_MODEL",
}


def default_provider_configs(
    env: Mapping[str, str] | None = None,
) -> tuple[ProviderConfig, ...]:
    """Templates with env overrides applied.

    The default model is only *promoted*, never dropped: an override becomes the
    first choice and the previous default slides into the fallback list, so a
    stale override still degrades into something that works.
    """
    env = os.environ if env is None else env
    configs: list[ProviderConfig] = []
    for template in PROVIDER_TEMPLATES:
        override = env.get(_MODEL_OVERRIDE.get(template.name, ""), "").strip()
        config = template
        if override and override != template.default_model:
            fallbacks = (template.default_model, *template.fallback_models)
            config = ProviderConfig(
                **{
                    **template.__dict__,
                    "default_model": override,
                    "fallback_models": fallbacks,
                }
            )
        if template.name == "openrouter":
            config = _with_openrouter_attribution(config, env)
        configs.append(config)
    return tuple(configs)


def _with_openrouter_attribution(
    config: ProviderConfig, env: Mapping[str, str]
) -> ProviderConfig:
    headers = dict(config.headers)
    referer = env.get("OPENROUTER_APP_URL", "").strip()
    title = env.get("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    if not headers:
        return config
    return ProviderConfig(**{**config.__dict__, "headers": headers})


def discover_openrouter_free_models(
    api_key: str | None = None, *, timeout: float = 20.0
) -> tuple[str, ...]:
    """Ask OpenRouter which models are free *right now*.

    Useful when the hardcoded defaults go stale: feed the result into
    ``ITSBOB_OPENROUTER_MODEL``. Returns ``()`` if the catalog can't be read.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception:  # network, auth, egress policy — all non-fatal here
        return ()

    free: list[str] = []
    for model in payload.get("data", []):
        pricing = model.get("pricing") or {}
        prompt_price = pricing.get("prompt")
        try:
            if prompt_price is not None and float(prompt_price) == 0.0:
                free.append(model["id"])
        except (TypeError, ValueError):
            continue
    return tuple(sorted(free))
