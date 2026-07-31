"""Concrete providers.

OpenRouter, Groq and Gemini all speak OpenAI's ``/chat/completions``, so one
client implementation covers all three — the differences are a base URL, an env
var, and a model catalog (see :mod:`itsbob.llm.catalog`). :class:`EchoProvider`
is the offline stand-in that keeps tests and no-key runs honest.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from ..config import ProviderConfig
from .base import (
    BadRequest,
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderNotConfigured,
    ProviderUnavailable,
    RateLimited,
    Usage,
)
from .catalog import GOOGLE, GROQ, OPENROUTER

__all__ = [
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "GroqProvider",
    "GoogleProvider",
    "EchoProvider",
    "build_provider",
]


class OpenAICompatibleProvider(Provider):
    """Any endpoint that implements OpenAI's chat completions API.

    The ``openai`` package is imported lazily so the rest of the framework —
    including the whole simulation running on :class:`EchoProvider` — works
    without it installed.
    """

    def __init__(
        self, config: ProviderConfig, env: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(config)
        self._env = env
        self._client: Any | None = None

    def _make_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ProviderNotConfigured(
                f"{self.name}: the 'openai' package is required (pip install openai)"
            ) from exc

        api_key = self.config.api_key(self._env)
        if not api_key:
            raise ProviderNotConfigured(
                f"{self.name}: set {self.config.api_key_env} to enable this provider"
            )
        return OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,  # the router owns retry policy
            default_headers=dict(self.config.headers) or None,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _build_kwargs(self, request: LLMRequest, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": request.payload(),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            kwargs["stop"] = list(request.stop)
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _complete(self, request: LLMRequest, model: str) -> LLMResponse:
        try:
            completion = self.client.chat.completions.create(
                **self._build_kwargs(request, model)
            )
        except Exception as exc:
            raise _translate_error(self.name, exc) from exc

        choice = completion.choices[0] if completion.choices else None
        text = ""
        finish_reason = None
        if choice is not None:
            text = getattr(choice.message, "content", None) or ""
            finish_reason = choice.finish_reason

        usage = Usage()
        if getattr(completion, "usage", None):
            usage = Usage(
                prompt_tokens=completion.usage.prompt_tokens or 0,
                completion_tokens=completion.usage.completion_tokens or 0,
            )

        if not text.strip() and finish_reason == "length":
            # Reasoning models burn max_tokens on hidden thinking and return an
            # empty body. Surfacing it as retryable lets the router move on.
            raise ProviderUnavailable(
                f"{self.name}/{model}: token budget exhausted before any output "
                f"(raise LLMRequest.max_tokens, currently {request.max_tokens})"
            )

        return LLMResponse(
            text=text,
            model=getattr(completion, "model", model) or model,
            provider=self.name,
            usage=usage,
            finish_reason=finish_reason,
            raw=completion,
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, config: ProviderConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config or OPENROUTER, **kwargs)


class GroqProvider(OpenAICompatibleProvider):
    def __init__(self, config: ProviderConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config or GROQ, **kwargs)


class GoogleProvider(OpenAICompatibleProvider):
    def __init__(self, config: ProviderConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config or GOOGLE, **kwargs)


_STATUS_RE = re.compile(r"\b(4\d\d|5\d\d)\b")


def _translate_error(provider: str, exc: Exception) -> Exception:
    """Map an SDK exception onto our error vocabulary.

    Matches on duck-typed attributes rather than ``openai`` exception classes so
    this keeps working if the SDK reshuffles its hierarchy.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is None:
        match = _STATUS_RE.search(str(exc))
        status = int(match.group(1)) if match else None

    message = f"{provider}: {type(exc).__name__}: {exc}"[:500]

    if status == 429:
        return RateLimited(message, retry_after=_retry_after(exc))
    if status in (401, 403):
        return ProviderNotConfigured(message)
    if status == 404:
        return BadRequest(message)
    if status is not None and 500 <= status < 600:
        return ProviderUnavailable(message)
    if status is not None and 400 <= status < 500:
        return BadRequest(message)
    # No status at all: connection reset, DNS, timeout, proxy refusal.
    return ProviderUnavailable(message)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value:
            try:
                return float(str(value).rstrip("s"))
            except ValueError:
                continue
    return None


class EchoProvider(Provider):
    """Deterministic offline provider.

    Not a mock in the test-double sense — it is a real fallback. With no keys
    the simulation still runs end to end, and because its output is a pure
    function of the prompt, seeded runs are reproducible. When asked for JSON it
    emits a well-formed decision object so the LLM decision policy exercises its
    real parsing path.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(
            config
            or ProviderConfig(
                name="echo",
                base_url="",
                api_key_env="ITSBOB_ECHO",  # never read; always configured
                default_model="echo-1",
                requests_per_minute=10_000,
            )
        )

    def is_configured(self, env: Mapping[str, str] | None = None) -> bool:
        return True

    def _complete(self, request: LLMRequest, model: str) -> LLMResponse:
        prompt = "\n".join(m.content for m in request.messages)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if request.json_mode:
            text = json.dumps(self._decide(prompt, digest))
        else:
            text = self._narrate(request, digest)

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=request.approx_prompt_tokens(),
                completion_tokens=max(1, len(text) // 4),
            ),
            finish_reason="stop",
        )

    def _decide(self, prompt: str, digest: str) -> dict[str, Any]:
        options = re.findall(r"^\s*-\s*([a-z_][a-z0-9_]*)\b", prompt, re.MULTILINE)
        options = list(dict.fromkeys(options))
        if options:
            action = options[int(digest[:8], 16) % len(options)]
        else:
            action = "observe"
        return {
            "action": action,
            "rationale": "offline heuristic: deterministic choice from the prompt",
            "confidence": 0.4,
        }

    def _narrate(self, request: LLMRequest, digest: str) -> str:
        last = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        subject = " ".join(last.split()[:12]) or "the silence"
        return f"[echo:{digest[:6]}] Nothing answers, so Bob thinks it through alone: {subject}"


_BUILDERS = {
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
    "google": GoogleProvider,
    "echo": EchoProvider,
}


def build_provider(
    config: ProviderConfig, env: Mapping[str, str] | None = None
) -> Provider:
    """Instantiate the provider class matching ``config.name``.

    Unknown names get the generic OpenAI-compatible client, so adding a vendor
    is a config entry rather than a code change.
    """
    builder = _BUILDERS.get(config.name)
    if builder is EchoProvider:
        return EchoProvider(config)
    if builder is not None:
        return builder(config, env=env)
    return OpenAICompatibleProvider(config, env=env)
