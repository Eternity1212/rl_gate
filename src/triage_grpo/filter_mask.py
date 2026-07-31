"""Sample masks for W_HIGH handling (docs/SPEC.md §4.4)."""

from __future__ import annotations

from typing import List, Sequence

from triage_grpo.types import Cell


def build_sample_mask(
    cells: Sequence[Cell],
    w_high_mode: str = "filter",
) -> List[bool]:
    """Return per-sample keep mask.

    - filter: W_HIGH -> False (excluded from gradient)
    - downweight: all True (penalty applied in advantage instead)
    """
    if w_high_mode == "downweight":
        return [True] * len(cells)
    if w_high_mode == "filter":
        return [c != Cell.W_HIGH for c in cells]
    raise ValueError(f"unknown w_high_mode: {w_high_mode}")
