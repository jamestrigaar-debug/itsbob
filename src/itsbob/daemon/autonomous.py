"""Built-in weighted work for autonomous serve mode."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Sequence
from ..router.tiers import Tier

@dataclass(frozen=True)
class AutonomousTask:
    name: str
    prompt: str
    tier: Tier
    weight: float

AUTONOMOUS_TASKS: tuple[AutonomousTask, ...] = (
    AutonomousTask("memory filter", "Review long-term and short-term memories, remove redundancy, and condense important information into compact understandable summaries.", Tier.A, 1.0),
    AutonomousTask("check in with user", "Send the user a brief thoughtful question on Discord that invites a useful response.", Tier.C, 4.0),
    AutonomousTask("news headline", "Find one genuinely relevant current news item and send a concise sourced headline and summary.", Tier.B, 3.0),
    AutonomousTask("web digest", "Scrape and condense useful current information from configured Reddit, Twitter, 4chan, or news sources into a readable digest.", Tier.A, 1.5),
    AutonomousTask("create script", "Identify a useful small script to create for this workspace, implement it, and report what it does.", Tier.S, 0.35),
    AutonomousTask("create task", "Identify one useful recurring task and create it in the task store with a clear schedule and prompt.", Tier.A, 0.8),
    AutonomousTask("observe screens", "Take a screenshot of the current screens and observe anything useful or requiring attention.", Tier.B, 2.0),
)

def choose(pool: Sequence[AutonomousTask] = AUTONOMOUS_TASKS, *, rng: random.Random | None = None, pressure: float = 1.0) -> AutonomousTask:
    rng = rng or random
    weights = [max(0.01, item.weight * (pressure if item.tier >= Tier.A else 1.0)) for item in pool]
    return rng.choices(list(pool), weights=weights, k=1)[0]
