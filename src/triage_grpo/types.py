"""Shared types for TRIAGE-GRPO (see docs/SPEC.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence


class Cell(str, Enum):
    """2x2 process-outcome consistency grid."""

    C_HIGH = "C_HIGH"  # correct + high process
    C_LOW = "C_LOW"  # correct + low process
    W_LOW = "W_LOW"  # wrong + low process
    W_HIGH = "W_HIGH"  # wrong + high process (hack signature)


@dataclass(frozen=True)
class GroupInput:
    """One prompt group of G rollouts."""

    outcome_rewards: Sequence[float]
    process_scores: Sequence[float]
    confidences: Sequence[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.outcome_rewards)
        if n == 0:
            raise ValueError("GroupInput must be non-empty")
        if len(self.process_scores) != n:
            raise ValueError("process_scores length must match outcome_rewards")
        if self.confidences is not None and len(self.confidences) != n:
            raise ValueError("confidences length must match outcome_rewards")


@dataclass
class AdvantageOutput:
    """Per-response advantages and diagnostics for one group."""

    advantages: List[float]
    cells: List[Cell]
    masks: List[bool]
    a_outcome: List[float] = field(default_factory=list)
    a_process: List[float] = field(default_factory=list)
    a_neg: List[float] = field(default_factory=list)
    a_hack: List[float] = field(default_factory=list)
    method: str = ""
