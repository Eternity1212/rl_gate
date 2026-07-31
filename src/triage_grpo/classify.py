"""2x2 cell assignment (docs/SPEC.md §3)."""

from __future__ import annotations

from typing import List, Sequence

from triage_grpo.types import Cell


def assign_cell(outcome: float, process: float, tau: float) -> Cell:
    """Assign one rollout to a consistency cell."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    correct = float(outcome) >= 0.5
    high_process = float(process) >= tau
    if correct and high_process:
        return Cell.C_HIGH
    if correct and not high_process:
        return Cell.C_LOW
    if (not correct) and (not high_process):
        return Cell.W_LOW
    return Cell.W_HIGH


def assign_cells(
    outcome_rewards: Sequence[float],
    process_scores: Sequence[float],
    tau: float,
) -> List[Cell]:
    if len(outcome_rewards) != len(process_scores):
        raise ValueError("length mismatch")
    return [
        assign_cell(o, p, tau) for o, p in zip(outcome_rewards, process_scores)
    ]
