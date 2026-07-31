"""The character: traits, needs, energy, memory.

Traits are fixed and bias *how* choices get made. Needs drift upward every tick
and bias *what* gets chosen. Together they give a heuristic policy enough to
work with, and give an LLM policy something concrete to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from ..config import EnergySettings, MemorySettings
from ..memory.bank import MemoryBank
from .energy import EnergyLedger

__all__ = ["Traits", "Needs", "Character"]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class Traits:
    """Stable dispositions in ``[0, 1]``."""

    curiosity: float = 0.6  # pull toward new information (and toward the LLM)
    diligence: float = 0.5  # willingness to do unrewarding work
    sociability: float = 0.5  # pull toward other agents
    caution: float = 0.5  # reluctance to act on low confidence

    def __post_init__(self) -> None:
        for name in ("curiosity", "diligence", "sociability", "caution"):
            setattr(self, name, _clamp(getattr(self, name)))

    def as_dict(self) -> dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "diligence": self.diligence,
            "sociability": self.sociability,
            "caution": self.caution,
        }

    def render(self) -> str:
        return ", ".join(f"{k} {v:.2f}" for k, v in self.as_dict().items())


@dataclass
class Needs:
    """Drives in ``[0, 1]``, where 1 is maximally unmet and uncomfortable."""

    levels: dict[str, float] = field(
        default_factory=lambda: {
            "rest": 0.2,
            "sustenance": 0.3,
            "social": 0.3,
            "curiosity": 0.4,
            "purpose": 0.3,
        }
    )
    drift: dict[str, float] = field(
        default_factory=lambda: {
            "rest": 0.05,
            "sustenance": 0.06,
            "social": 0.03,
            "curiosity": 0.04,
            "purpose": 0.02,
        }
    )

    def __post_init__(self) -> None:
        self.levels = {k: _clamp(v) for k, v in self.levels.items()}

    def tick(self, *, multiplier: float = 1.0) -> None:
        for name, rate in self.drift.items():
            self.levels[name] = _clamp(self.levels.get(name, 0.0) + rate * multiplier)

    def satisfy(self, name: str, amount: float) -> float:
        """Reduce a need. Returns how much was actually relieved."""
        before = self.levels.get(name, 0.0)
        self.levels[name] = _clamp(before - abs(amount))
        return before - self.levels[name]

    def aggravate(self, name: str, amount: float) -> None:
        self.levels[name] = _clamp(self.levels.get(name, 0.0) + abs(amount))

    def most_pressing(self) -> tuple[str, float]:
        if not self.levels:
            return ("none", 0.0)
        name = max(self.levels, key=lambda k: self.levels[k])
        return (name, self.levels[name])

    @property
    def pressure(self) -> float:
        """Mean unmet need — a single number for 'how badly is it going'."""
        return sum(self.levels.values()) / len(self.levels) if self.levels else 0.0

    def render(self) -> str:
        ordered = sorted(self.levels.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{k} {v:.2f}" for k, v in ordered)

    def __getitem__(self, name: str) -> float:
        return self.levels.get(name, 0.0)

    def __iter__(self) -> Iterator[str]:
        return iter(self.levels)


@dataclass
class Character:
    """Everything that is true about the agent right now."""

    name: str = "Bob"
    backstory: str = (
        "A cautious generalist who keeps a notebook, distrusts easy answers, "
        "and has recently gained the ability to consult an oracle."
    )
    goal: str = "Understand the place well enough to be useful in it."
    traits: Traits = field(default_factory=Traits)
    needs: Needs = field(default_factory=Needs)
    energy: EnergyLedger = field(default_factory=EnergyLedger)
    memory: MemoryBank = field(default_factory=MemoryBank)
    mood: float = 0.0  # -1 miserable .. +1 delighted
    location: str = "the workshop"

    @classmethod
    def create(
        cls,
        name: str = "Bob",
        *,
        traits: Traits | None = None,
        energy_settings: EnergySettings | None = None,
        memory_settings: MemorySettings | None = None,
        router: Any | None = None,
        **kwargs: Any,
    ) -> "Character":
        """Build a character wired to the given settings and router."""
        energy_settings = energy_settings or EnergySettings()
        return cls(
            name=name,
            traits=traits or Traits(),
            energy=EnergyLedger.from_settings(energy_settings),
            memory=MemoryBank(memory_settings or MemorySettings(), router=router),
            **kwargs,
        )

    # -- state -------------------------------------------------------------

    def adjust_mood(self, delta: float) -> float:
        self.mood = _clamp(self.mood + delta, -1.0, 1.0)
        return self.mood

    @property
    def mood_word(self) -> str:
        if self.mood > 0.35:
            return "buoyant"
        if self.mood > 0.1:
            return "content"
        if self.mood < -0.35:
            return "wretched"
        if self.mood < -0.1:
            return "irritable"
        return "level"

    def can_deliberate(self, cost: float) -> bool:
        """Deliberation needs both the energy and the wakefulness for it."""
        return not self.energy.is_exhausted and self.energy.can_afford(cost)

    # -- prompting ---------------------------------------------------------

    def sheet(self) -> str:
        """Character block for LLM prompts."""
        return "\n".join(
            [
                f"Name: {self.name}",
                f"Goal: {self.goal}",
                f"Backstory: {self.backstory}",
                f"Location: {self.location}",
                f"Traits: {self.traits.render()}",
                f"Needs (1.0 = desperate): {self.needs.render()}",
                f"Energy: {self.energy.describe()}",
                f"Mood: {self.mood_word} ({self.mood:+.2f})",
            ]
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "mood": round(self.mood, 3),
            "energy": round(self.energy.current, 2),
            "energy_fraction": round(self.energy.fraction, 3),
            "exhausted": self.energy.is_exhausted,
            "needs": {k: round(v, 3) for k, v in self.needs.levels.items()},
            "traits": self.traits.as_dict(),
            "memory": self.memory.stats(),
        }

    def close(self) -> None:
        self.memory.close()
