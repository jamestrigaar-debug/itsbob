"""The world: a clock, some places, and a little weather.

Deliberately thin. Its job is to give the character something to perceive and
somewhere to be, and to give the framework a clear place to grow into — swap
this class for a real environment and nothing above it changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = ["World", "PHASES"]

PHASES: tuple[str, ...] = ("dawn", "morning", "afternoon", "evening", "night")

_WEATHER = ("clear", "overcast", "raining", "windy", "still and cold")
_HAPPENINGS = (
    "a door stands open that was shut before",
    "the machine in the corner is warm to the touch",
    "someone has left a half-finished note on the bench",
    "the light has changed and nothing else has",
    "there are footprints that do not match anyone here",
    "the shelf is one item emptier than it was",
)


@dataclass
class World:
    """Tick counter plus enough texture to make observation worth doing."""

    tick: int = 0
    #: Ticks per day; also controls how fast the phase advances.
    ticks_per_day: int = 10
    locations: tuple[str, ...] = ("the workshop", "the yard", "the long room", "the road")
    inhabitants: tuple[str, ...] = ("Mira", "the quiet neighbour", "a passing trader")
    weather: str = "clear"
    project_progress: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)

    # -- time --------------------------------------------------------------

    @property
    def day(self) -> int:
        return self.tick // self.ticks_per_day

    @property
    def phase(self) -> str:
        slot = (self.tick % self.ticks_per_day) * len(PHASES) // self.ticks_per_day
        return PHASES[slot]

    @property
    def is_night(self) -> bool:
        return self.phase == "night"

    def advance(self, rng: random.Random | None = None) -> int:
        """Move one tick forward. Weather drifts occasionally."""
        self.tick += 1
        rng = rng or random
        if rng.random() < 0.15:
            self.weather = rng.choice(_WEATHER)
        return self.tick

    # -- perception --------------------------------------------------------

    def describe(self) -> str:
        return (
            f"Day {self.day}, {self.phase}, {self.weather}. "
            f"Project progress {self.project_progress:.2f}."
        )

    def observe(self, rng: random.Random | None = None) -> str:
        """One noticed detail. Consumed by the ``observe`` action."""
        rng = rng or random
        return rng.choice(_HAPPENINGS)

    def random_neighbour(self, rng: random.Random | None = None) -> str | None:
        if not self.inhabitants:
            return None
        rng = rng or random
        return rng.choice(self.inhabitants)

    def random_location(self, rng: random.Random | None = None) -> str:
        rng = rng or random
        return rng.choice(self.locations)

    # -- mutation ----------------------------------------------------------

    def advance_project(self, amount: float) -> float:
        self.project_progress = min(1.0, self.project_progress + max(0.0, amount))
        return self.project_progress

    def snapshot(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "day": self.day,
            "phase": self.phase,
            "weather": self.weather,
            "project_progress": round(self.project_progress, 3),
        }
