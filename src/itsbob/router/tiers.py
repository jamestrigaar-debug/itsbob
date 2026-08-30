"""The Complexity Tier taxonomy: S, A, B, C, D.

Named after the "itsbob" complexity-based hierarchical router design —
Classify First, Execute Cheapest, Fallback Gracefully. Deliberately its own
enum rather than reusing anything action/policy-shaped from ``character/``:
a tier is a *routing* decision, made before anything about "what action to
take" is even asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["Tier", "GateDecision"]


class Tier(str, Enum):
    """Cheapest first. Ordering matters for display and for escalation.

    Comparison is by :attr:`rank`, explicitly. Inheriting from ``str`` for
    JSON-friendliness means the default comparison is *alphabetical*, which
    silently reverses the thing this enum exists to express: ``Tier.A <
    Tier.B`` was True and ``max([Tier.C, Tier.A])`` returned the cheapest tier,
    not the most capable. Any code that sorted or compared tiers got the wrong
    answer without erroring.

    All four comparisons are written out rather than derived with
    ``functools.total_ordering``, which does nothing useful on a ``str`` mixin:
    it only fills in operators the class does not already have, and ``str``
    supplies every one of them. Decorating this class with it left ``>``
    comparing alphabetically while ``<`` compared correctly — a worse state
    than before, because the two disagreed.
    """

    D = "D"  #: Direct — a registered routine, no model at all
    C = "C"  #: Cheapest — the local model if one is running, else the cheapest cloud one
    B = "B"  #: Standard — the everyday workhorse
    A = "A"  #: Premium — judgement, ambiguity, anything hard to undo
    S = "S"  #: Halt — nothing could answer; a person has to

    @property
    def label(self) -> str:
        # Named for what the tier *is*, not for where it runs: Tier C prefers a
        # local model but falls back to the cheapest cloud one, and labelling
        # that "Local Back Brain" in `doctor` output claims something untrue.
        return {
            Tier.D: "Direct routine",
            Tier.C: "Cheapest model",
            Tier.B: "Standard model",
            Tier.A: "Premium model",
            Tier.S: "Halt — ask a person",
        }[self]

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Tier):
            return self.rank < other.rank
        return NotImplemented

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Tier):
            return self.rank <= other.rank
        return NotImplemented

    def __gt__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Tier):
            return self.rank > other.rank
        return NotImplemented

    def __ge__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Tier):
            return self.rank >= other.rank
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tier):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def rank(self) -> int:
        """Cost/capability order, cheapest first. S sits above A: reaching it
        means nothing else could answer, which is the most expensive outcome
        there is — a person now has to."""
        return {Tier.D: 0, Tier.C: 1, Tier.B: 2, Tier.A: 3, Tier.S: 4}[self]

    @property
    def uses_llm(self) -> bool:
        return self is not Tier.D

    @property
    def is_cloud(self) -> bool:
        return self in (Tier.B, Tier.A)


#: The tags the Gatekeeper prompt is instructed to output, and their tier.
GATEKEEPER_TAGS: dict[str, Tier] = {
    "SCRIPT": Tier.D,
    "LOCAL_SUM": Tier.C,
    "CLOUD_B": Tier.B,
    "CLOUD_A": Tier.A,
}


@dataclass
class GateDecision:
    """What the Gatekeeper decided, and why — the unit the pipeline routes on."""

    tier: Tier
    fingerprint: str
    #: "gatekeeper" (local model tagged it) | "heuristic" (model unavailable,
    #: rule-based classifier stood in) | "escalation" (a lower tier failed and
    #: this is where it landed) | "user" (Tier S, a human will decide)
    source: str = "gatekeeper"
    reasoning: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "tier_label": self.tier.label,
            "fingerprint": self.fingerprint,
            "source": self.source,
            "reasoning": self.reasoning,
            "latency_ms": round(self.latency_ms, 1),
            "metadata": self.metadata,
        }
