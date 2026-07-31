"""Training/eval diagnostics (docs/SPEC.md §7)."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from triage_grpo.types import Cell


def cell_stats(cells: Sequence[Cell]) -> Dict[str, float]:
    n = len(cells)
    if n == 0:
        return {
            "pct_c_high": 0.0,
            "pct_c_low": 0.0,
            "pct_w_low": 0.0,
            "pct_w_high": 0.0,
            "n": 0.0,
        }
    cnt = Counter(cells)
    return {
        "pct_c_high": cnt[Cell.C_HIGH] / n,
        "pct_c_low": cnt[Cell.C_LOW] / n,
        "pct_w_low": cnt[Cell.W_LOW] / n,
        "pct_w_high": cnt[Cell.W_HIGH] / n,
        "n": float(n),
    }


def hack_rate(cells: Sequence[Cell]) -> float:
    """Proxy hack rate = fraction of W_HIGH."""
    if not cells:
        return 0.0
    return sum(1 for c in cells if c == Cell.W_HIGH) / len(cells)


def mean_process_given_wrong(
    outcome_rewards: Sequence[float],
    process_scores: Sequence[float],
) -> float:
    vals: List[float] = [
        float(p)
        for o, p in zip(outcome_rewards, process_scores)
        if float(o) < 0.5
    ]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def frac_masked(masks: Sequence[bool]) -> float:
    if not masks:
        return 0.0
    return sum(1 for m in masks if not m) / len(masks)
