"""The "Back Brain": a small local model reached over Ollama's HTTP API.

This is Tier C/the Gatekeeper's engine — a 1.5B-3B model (Phi-3.5-mini,
Qwen2.5-1.5B, ...) served by `ollama serve` on localhost. It is treated as
just another :class:`~itsbob.llm.base.Provider`, so it can sit in the same
:class:`~itsbob.llm.router.LLMRouter` failover chain as the cloud providers,
or be driven directly by :class:`itsbob.router.gatekeeper.Gatekeeper`.

No local model running? Every call raises :class:`ProviderUnavailable` and
callers fall back — the same graceful-degradation contract as every other
provider in this codebase.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from ..config import ProviderConfig
from .base import LLMRequest, LLMResponse, Provider, ProviderUnavailable, Usage

__all__ = [
    "OllamaProvider",
    "OLLAMA",
    "default_ollama_config",
    "is_ollama_running",
    "list_ollama_models",
]


def default_ollama_config(env: Mapping[str, str] | None = None) -> ProviderConfig:
    """The Back Brain's config, with ``ITSBOB_OLLAMA_URL`` / ``ITSBOB_OLLAMA_MODEL`` applied.

    Mirrors how ``llm/catalog.py`` handles cloud model overrides: an env
    override is promoted to the default and the old default slides into the
    fallback list, so a stale override still degrades into something that
    works.
    """
    env = os.environ if env is None else env
    base_url = env.get("ITSBOB_OLLAMA_URL", "").strip() or "http://127.0.0.1:11434"
    model_override = env.get("ITSBOB_OLLAMA_MODEL", "").strip()
    default_model = "qwen2.5:1.5b"
    fallbacks: tuple[str, ...] = ("phi3.5:3.8b-mini-instruct-q4_K_M", "qwen2.5:0.5b")
    if model_override and model_override != default_model:
        fallbacks = (default_model, *fallbacks)
        default_model = model_override

    return ProviderConfig(
        name="ollama",
        base_url=base_url,
        api_key_env="ITSBOB_OLLAMA_UNUSED",  # local server needs no key
        default_model=default_model,
        fallback_models=fallbacks,
        requests_per_minute=10_000,  # local, no vendor quota
        # Classification must stay fast, but *answering* on a 1.5B model on a
        # cold cache genuinely takes ten to twenty seconds, and a 5s ceiling
        # meant every local answer timed out and fell through to a paid API
        # call — the exact opposite of the point. Callers that need the tight
        # budget pass `metadata={"timeout": 5}` on the request; the gatekeeper
        # does.
        timeout=float(env.get("ITSBOB_OLLAMA_TIMEOUT", "").strip() or 45.0),
    )


#: The default config, for callers that don't need env overrides applied.
OLLAMA = default_ollama_config({})


def is_ollama_running(base_url: str | None = None, *, timeout: float = 0.5) -> bool:
    """Cheap liveness probe, used by ``itsbob doctor`` and the GUI status panel.

    Resolves ``ITSBOB_OLLAMA_URL`` at call time (rather than a module-load-time
    default) so a `.env` loaded after import still takes effect.
    """
    base_url = base_url or default_ollama_config().base_url
    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def list_ollama_models(base_url: str | None = None, *, timeout: float = 2.0) -> list[str]:
    """The model tags actually pulled locally, per Ollama's own ``/api/tags``.

    Used by ``itsbob doctor`` to catch the "configured a model that was
    never pulled" mistake *before* a call 404s — returns ``[]`` if Ollama
    isn't reachable rather than raising, since this is a diagnostic, not a
    call that needs to succeed.
    """
    base_url = base_url or default_ollama_config().base_url
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/tags", timeout=timeout
        ) as resp:
            data = json.load(resp)
        return [m.get("name") or m.get("model") for m in data.get("models", [])]
    except Exception:
        return []


class OllamaProvider(Provider):
    """Talks to a local `ollama serve` instance via its native `/api/chat`.

    Deliberately not routed through the OpenAI-compatible client: Ollama's
    native endpoint reports load/eval timings we want for the <800ms latency
    budget, and needs no `openai` package to be installed.
    """

    def __init__(
        self, config: ProviderConfig | None = None, env: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(config or default_ollama_config(env))
        self._env = env

    def is_configured(self, env: Mapping[str, str] | None = None) -> bool:
        # "Configured" for a local provider means "reachable", not "has a key".
        return is_ollama_running(self.config.base_url)

    def _complete(self, request: LLMRequest, model: str) -> LLMResponse:
        # A per-request override, so one provider serves both the gatekeeper's
        # sub-second budget and a full local answer.
        timeout = self.config.timeout
        requested = request.metadata.get("timeout")
        if requested:
            try:
                timeout = max(0.5, float(requested))
            except (TypeError, ValueError):
                pass
        payload = {
            "model": model,
            "messages": request.payload(),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.json_mode:
            payload["format"] = "json"

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        http_request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as resp:
                data: dict[str, Any] = json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ProviderUnavailable(
                    f"ollama: model {model!r} is not pulled locally "
                    f"(HTTP 404) — run `ollama pull {model}`, or unset "
                    "ITSBOB_OLLAMA_MODEL to use whatever this repo's default "
                    "(qwen2.5:1.5b) or its fallbacks resolve to instead"
                ) from exc
            raise ProviderUnavailable(f"ollama: {exc}") from exc
        except urllib.error.URLError as exc:
            raise ProviderUnavailable(f"ollama: {exc}") from exc
        except TimeoutError as exc:  # pragma: no cover - platform dependent
            raise ProviderUnavailable(f"ollama: timed out after {timeout}s") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        text = (data.get("message") or {}).get("content", "")
        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count") or request.approx_prompt_tokens()),
            completion_tokens=int(data.get("eval_count") or max(1, len(text) // 4)),
        )
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason="stop" if data.get("done") else None,
            raw=data,
        )
