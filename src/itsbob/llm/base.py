"""Provider-neutral request/response types and the Provider contract."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..config import ProviderConfig

__all__ = [
    "Message",
    "LLMRequest",
    "LLMResponse",
    "Usage",
    "Provider",
    "LLMError",
    "ProviderNotConfigured",
    "ProviderUnavailable",
    "RateLimited",
    "BadRequest",
    "AllProvidersFailed",
    "system",
    "user",
    "assistant",
]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for anything the LLM layer raises."""


class ProviderNotConfigured(LLMError):
    """No API key, or the provider was explicitly disabled."""


class ProviderUnavailable(LLMError):
    """Transient: network trouble, 5xx, timeout, or an open circuit breaker."""


class RateLimited(ProviderUnavailable):
    """429 or a locally-enforced rate budget. Carries a retry hint when known."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BadRequest(LLMError):
    """Permanent for this provider: bad model id, 4xx that retrying won't fix."""


class AllProvidersFailed(LLMError):
    """Every candidate provider was exhausted."""

    def __init__(self, errors: Mapping[str, BaseException]) -> None:
        detail = "; ".join(f"{name}: {err}" for name, err in errors.items()) or "none tried"
        super().__init__(f"all providers failed ({detail})")
        self.errors = dict(errors)


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class LLMRequest:
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.7
    #: Reasoning models spend part of this budget thinking, so keep headroom —
    #: too small a value yields an empty `content` with a `stop` finish reason.
    max_tokens: int = 800
    stop: Sequence[str] | None = None
    #: Ask the provider for a JSON object. Ignored by providers that lack it.
    json_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> "LLMRequest":
        messages: list[Message] = []
        if system_prompt:
            messages.append(system(system_prompt))
        messages.append(user(prompt))
        return cls(messages=messages, **kwargs)

    def payload(self) -> list[dict[str, str]]:
        return [m.as_dict() for m in self.messages]

    def approx_prompt_tokens(self) -> int:
        """Rough pre-flight estimate (~4 chars/token) for budgeting."""
        chars = sum(len(m.content) for m in self.messages)
        return max(1, chars // 4)


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    raw: Any | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


# --------------------------------------------------------------------------
# Provider contract
# --------------------------------------------------------------------------


class Provider(ABC):
    """One vendor endpoint.

    Subclasses implement :meth:`_complete` and may raise anything; :meth:`complete`
    normalizes the result and stamps latency so the router sees one error
    vocabulary regardless of which SDK blew up.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def models(self) -> tuple[str, ...]:
        return self.config.models()

    def is_configured(self, env: Mapping[str, str] | None = None) -> bool:
        return self.config.is_configured(env)

    @abstractmethod
    def _complete(self, request: LLMRequest, model: str) -> LLMResponse:
        """Perform one round trip. Raise on failure."""

    def complete(self, request: LLMRequest, model: str | None = None) -> LLMResponse:
        chosen = model or request.model or self.config.default_model
        if not chosen:
            raise ProviderNotConfigured(f"{self.name}: no model configured")
        started = time.perf_counter()
        response = self._complete(request, chosen)
        if not response.latency_ms:
            response.latency_ms = (time.perf_counter() - started) * 1000
        return response

    def candidate_models(self, request: LLMRequest) -> Iterable[str]:
        """Models to try for this request, best first."""
        if request.model:
            return (request.model,)
        return self.models

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<{type(self).__name__} {self.name} models={list(self.models)}>"
