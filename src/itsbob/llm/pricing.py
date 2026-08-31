"""What a call actually cost, so the token panel shows money and not just counts.

Token counts alone do not answer the question people have, which is "am I
spending too much". A million Tier C tokens and a million Tier S tokens differ
by roughly forty times in price, so a single "tokens used" number is worse than
no number: it invites exactly the wrong conclusion.

Prices are per million tokens, in US dollars, and they are **estimates**. They
move, they vary by region and tier, and a free allowance makes the real bill
zero regardless. So everything here is labelled an estimate, an unknown model
costs nothing rather than guessing, and the local model is pinned at zero
because it genuinely is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

__all__ = ["Price", "PRICES", "price_for", "estimate", "Ledger"]


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens."""

    prompt: float
    completion: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt + completion_tokens * self.completion
        ) / 1_000_000


#: Keyed by a substring of the model id, longest match first, so
#: `gemini-3.5-flash-lite` does not match the `gemini-3.5-flash` entry.
PRICES: dict[str, Price] = {
    "gemini-3.5-flash-lite": Price(0.10, 0.40),
    "gemini-flash-lite": Price(0.10, 0.40),
    "gemini-3.1-flash-lite": Price(0.10, 0.40),
    "gemini-3.6-flash": Price(0.30, 2.50),
    "gemini-3.5-flash": Price(0.30, 2.50),
    "gemini-pro": Price(1.25, 10.00),
    "gemini-embedding": Price(0.15, 0.0),
    # Local. Not "cheap" — actually free, which is the whole point of it.
    "qwen": Price(0.0, 0.0),
    "phi3": Price(0.0, 0.0),
    "llama": Price(0.0, 0.0),
}


def price_for(model: str, provider: str = "") -> Price | None:
    """The price for a model id, or ``None`` when it is genuinely unknown.

    ``None`` rather than a guess: a made-up number shown next to real ones is
    indistinguishable from a real one, and the panel says "unpriced" instead.
    """
    if provider == "ollama":
        return Price(0.0, 0.0)
    name = (model or "").lower()
    for key in sorted(PRICES, key=len, reverse=True):
        if key in name:
            return PRICES[key]
    return None


def estimate(records: Iterable[Any]) -> dict[str, Any]:
    """Roll a set of usage records into a spend summary."""
    total = 0.0
    unpriced = 0
    prompt_tokens = completion_tokens = 0
    for record in records:
        usage = getattr(record, "usage", None)
        p = getattr(usage, "prompt_tokens", 0) or 0
        c = getattr(usage, "completion_tokens", 0) or 0
        prompt_tokens += p
        completion_tokens += c
        price = price_for(getattr(record, "model", ""), getattr(record, "provider", ""))
        if price is None:
            unpriced += 1
            continue
        total += price.cost(p, c)
    return {
        "usd": round(total, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens": prompt_tokens + completion_tokens,
        "unpriced_calls": unpriced,
    }


class Ledger:
    """A read-only view over a :class:`~itsbob.llm.router.UsageTracker`.

    Lives apart from the tracker because the tracker is a hot path — it appends
    on every call and must stay trivial — while this is only ever read by a
    person looking at a panel.
    """

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker

    def records(self) -> list[Any]:
        return list(getattr(self.tracker, "records", []))

    def summary(self, *, since: float | None = None) -> dict[str, Any]:
        rows = self.records()
        if since is not None:
            rows = [r for r in rows if getattr(r, "timestamp", 0) >= since]
        totals = estimate(rows)

        by_model: dict[str, dict[str, Any]] = {}
        by_purpose: dict[str, dict[str, Any]] = {}
        local_tokens = 0
        for record in rows:
            usage = getattr(record, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) or 0
            provider = getattr(record, "provider", "?")
            if provider == "ollama":
                local_tokens += tokens
            one = estimate([record])
            model_row = by_model.setdefault(
                f"{provider}/{getattr(record, 'model', '?')}",
                {"calls": 0, "tokens": 0, "usd": 0.0, "local": provider == "ollama"},
            )
            model_row["calls"] += 1
            model_row["tokens"] += tokens
            model_row["usd"] = round(model_row["usd"] + one["usd"], 4)

            # Purpose is the honest answer to "what am I paying for": answering,
            # classifying, extracting memories, or screening a turn.
            purpose = str(getattr(record, "purpose", "unspecified")).split(".")[0]
            purpose_row = by_purpose.setdefault(
                purpose, {"calls": 0, "tokens": 0, "usd": 0.0}
            )
            purpose_row["calls"] += 1
            purpose_row["tokens"] += tokens
            purpose_row["usd"] = round(purpose_row["usd"] + one["usd"], 4)

        total_tokens = totals["tokens"]
        return {
            **totals,
            "calls": len(rows),
            "local_tokens": local_tokens,
            #: The number that says whether the local model is earning its keep.
            "local_share": round(local_tokens / total_tokens, 3) if total_tokens else 0.0,
            "by_model": dict(
                sorted(by_model.items(), key=lambda kv: -kv[1]["tokens"])
            ),
            "by_purpose": dict(
                sorted(by_purpose.items(), key=lambda kv: -kv[1]["tokens"])
            ),
        }

    def recent(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.records()[-limit:]
        out = []
        for record in rows:
            usage = getattr(record, "usage", None)
            out.append(
                {
                    "at": getattr(record, "timestamp", 0.0),
                    "provider": getattr(record, "provider", "?"),
                    "model": getattr(record, "model", "?"),
                    "purpose": getattr(record, "purpose", ""),
                    "ok": bool(getattr(record, "ok", True)),
                    "tokens": getattr(usage, "total_tokens", 0) or 0,
                    "latency_ms": round(getattr(record, "latency_ms", 0.0), 1),
                    "usd": estimate([record])["usd"],
                }
            )
        out.reverse()
        return out
