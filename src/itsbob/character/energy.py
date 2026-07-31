"""The energy economy.

Energy is the only real constraint in the simulation. Acting costs it, thinking
costs more, and calling an LLM costs most of all — priced by the tokens the call
actually burned. That is the tension the game runs on: the smartest move is
usually the one you can least afford, and a character who deliberates about
everything runs itself into exhaustion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import EnergySettings
from ..llm.base import Usage

__all__ = [
    "InsufficientEnergy",
    "EnergyTransaction",
    "EnergyLedger",
    "TokenCostModel",
]


class InsufficientEnergy(RuntimeError):
    """Raised by :meth:`EnergyLedger.spend` when the balance won't cover a cost."""

    def __init__(self, requested: float, available: float, reason: str) -> None:
        super().__init__(
            f"cannot spend {requested:.1f} energy on {reason!r}: only {available:.1f} left"
        )
        self.requested = requested
        self.available = available
        self.reason = reason


@dataclass(frozen=True)
class EnergyTransaction:
    """One movement in the ledger. Negative ``amount`` is a spend."""

    tick: int
    amount: float
    reason: str
    balance: float

    @property
    def is_spend(self) -> bool:
        return self.amount < 0

    def __str__(self) -> str:  # pragma: no cover - convenience
        sign = "-" if self.is_spend else "+"
        return f"t{self.tick} {sign}{abs(self.amount):.1f} {self.reason} → {self.balance:.1f}"


@dataclass
class EnergyLedger:
    """Balance plus an audit trail of everything that moved it."""

    capacity: float = 100.0
    current: float = 100.0
    regen_per_tick: float = 6.0
    exhaustion_threshold: float = 15.0
    history: list[EnergyTransaction] = field(default_factory=list, repr=False)

    @classmethod
    def from_settings(cls, settings: EnergySettings) -> "EnergyLedger":
        return cls(
            capacity=settings.capacity,
            current=min(settings.starting, settings.capacity),
            regen_per_tick=settings.regen_per_tick,
            exhaustion_threshold=settings.exhaustion_threshold,
        )

    # -- state -------------------------------------------------------------

    @property
    def fraction(self) -> float:
        return self.current / self.capacity if self.capacity else 0.0

    @property
    def is_exhausted(self) -> bool:
        return self.current <= self.exhaustion_threshold

    def can_afford(self, amount: float) -> bool:
        return amount <= self.current + 1e-9

    def describe(self) -> str:
        state = "exhausted" if self.is_exhausted else "ok"
        return f"{self.current:.0f}/{self.capacity:.0f} energy ({state})"

    # -- movements ---------------------------------------------------------

    def spend(self, amount: float, reason: str, *, tick: int = 0) -> EnergyTransaction:
        amount = max(0.0, float(amount))
        if not self.can_afford(amount):
            raise InsufficientEnergy(amount, self.current, reason)
        self.current = max(0.0, self.current - amount)
        return self._log(tick, -amount, reason)

    def try_spend(
        self, amount: float, reason: str, *, tick: int = 0
    ) -> EnergyTransaction | None:
        """Spend if affordable, else ``None``. For optional, skippable actions."""
        try:
            return self.spend(amount, reason, tick=tick)
        except InsufficientEnergy:
            return None

    def gain(self, amount: float, reason: str, *, tick: int = 0) -> EnergyTransaction:
        amount = max(0.0, float(amount))
        self.current = min(self.capacity, self.current + amount)
        return self._log(tick, amount, reason)

    def regenerate(self, *, tick: int = 0, multiplier: float = 1.0) -> EnergyTransaction:
        """Passive per-tick recovery. Resting raises ``multiplier``."""
        return self.gain(self.regen_per_tick * multiplier, "regen", tick=tick)

    def _log(self, tick: int, amount: float, reason: str) -> EnergyTransaction:
        transaction = EnergyTransaction(
            tick=tick, amount=amount, reason=reason, balance=self.current
        )
        self.history.append(transaction)
        return transaction

    # -- reporting ---------------------------------------------------------

    def spent_total(self) -> float:
        return sum(-t.amount for t in self.history if t.is_spend)

    def spent_by_reason(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for transaction in self.history:
            if transaction.is_spend:
                totals[transaction.reason] = (
                    totals.get(transaction.reason, 0.0) - transaction.amount
                )
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


@dataclass(frozen=True)
class TokenCostModel:
    """Converts LLM token usage into energy.

    Charged after the fact, because you don't know what a call cost until it
    returns. :meth:`estimate` is the pre-flight guess used to decide whether
    deliberation is affordable at all.
    """

    call_overhead: float = 2.0
    tokens_per_energy: float = 400.0
    #: Completion tokens are the expensive half — generating is the work.
    completion_weight: float = 2.0

    @classmethod
    def from_settings(cls, settings: EnergySettings) -> "TokenCostModel":
        return cls(
            call_overhead=settings.call_overhead,
            tokens_per_energy=settings.tokens_per_energy,
        )

    def energy_for(self, usage: Usage) -> float:
        weighted = usage.prompt_tokens + self.completion_weight * usage.completion_tokens
        return self.call_overhead + weighted / max(1.0, self.tokens_per_energy)

    def estimate(self, prompt_tokens: int, max_tokens: int) -> float:
        """Worst-case cost, assuming the model uses its whole output budget."""
        return self.energy_for(
            Usage(prompt_tokens=prompt_tokens, completion_tokens=max_tokens)
        )

    def affordable_max_tokens(self, ledger: EnergyLedger, prompt_tokens: int) -> int:
        """Largest output budget the ledger can cover right now.

        Lets a tired character still ask a short question instead of being
        locked out of thinking entirely.
        """
        budget = ledger.current - self.call_overhead
        if budget <= 0:
            return 0
        remaining = budget * self.tokens_per_energy - prompt_tokens
        if remaining <= 0:
            return 0
        return max(0, int(math.floor(remaining / self.completion_weight)))
