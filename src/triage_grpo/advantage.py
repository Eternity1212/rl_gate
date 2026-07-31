"""Advantage computation for TRIAGE and baselines (docs/SPEC.md §4–5)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from triage_grpo.baselines import MethodName, normalize_method
from triage_grpo.classify import assign_cells
from triage_grpo.filter_mask import build_sample_mask
from triage_grpo.types import AdvantageOutput, Cell, GroupInput


def _as_array(xs: Sequence[float]) -> np.ndarray:
    return np.asarray(list(xs), dtype=np.float64)


def group_normalize(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standard GRPO-style mean/std normalize over a 1D group."""
    if values.size == 0:
        return values
    mu = float(values.mean())
    sigma = float(values.std(ddof=0))
    if sigma < eps:
        return np.zeros_like(values)
    return (values - mu) / (sigma + eps)


def _normalize_subset(
    values: np.ndarray, mask: np.ndarray, eps: float
) -> np.ndarray:
    """Normalize values on True-mask indices; zeros elsewhere."""
    out = np.zeros_like(values)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return out
    out[idx] = group_normalize(values[idx], eps=eps)
    return out


def _default_confidences(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64)


def compute_group_advantages(
    group: GroupInput,
    method: str | MethodName = MethodName.TRIAGE,
    *,
    tau: float = 0.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.1,
    w_high_mode: str = "filter",
    lens_normalize: bool = True,
    hybrid_lambda: float = 0.5,
    eps: float = 1e-6,
) -> AdvantageOutput:
    """Compute per-response advantages for one prompt group.

    Parameters follow docs/SPEC.md and docs/API.md.
    """
    method_s = normalize_method(method)
    r_o = _as_array(group.outcome_rewards)
    r_p = np.clip(_as_array(group.process_scores), 0.0, 1.0)
    n = r_o.size
    if group.confidences is None:
        conf = _default_confidences(n)
    else:
        conf = np.clip(_as_array(group.confidences), 1e-8, 1.0)

    cells = assign_cells(r_o.tolist(), r_p.tolist(), tau=tau)
    a_o = np.zeros(n)
    a_p = np.zeros(n)
    a_n = np.zeros(n)
    a_h = np.zeros(n)
    masks = [True] * n

    if method_s == MethodName.ORM_GRPO:
        a_o = group_normalize(r_o, eps=eps)
        advantages = a_o.copy()

    elif method_s == MethodName.PRM_DIRECT:
        advantages = group_normalize(r_p, eps=eps)

    elif method_s == MethodName.NAIVE_HYBRID:
        mixed = hybrid_lambda * r_o + (1.0 - hybrid_lambda) * r_p
        advantages = group_normalize(mixed, eps=eps)

    elif method_s == MethodName.P_GRPO:
        # Posterior gate: R = r_o + r_o * r_p
        gated = r_o + r_o * r_p
        advantages = group_normalize(gated, eps=eps)

    elif method_s == MethodName.PAPO:
        a_o = group_normalize(r_o, eps=eps)
        correct = r_o >= 0.5
        a_p = _normalize_subset(r_p, correct, eps=eps)
        advantages = a_o + alpha * a_p

    elif method_s == MethodName.PROF:
        # Filter C_LOW and W_HIGH (conflict types), ORM on remaining.
        masks = [
            c not in (Cell.C_LOW, Cell.W_HIGH) for c in cells
        ]
        keep = np.array(masks, dtype=bool)
        a_o = np.zeros(n)
        if keep.sum() >= 2:
            a_o[keep] = group_normalize(r_o[keep], eps=eps)
        elif keep.sum() == 1:
            # Single kept sample: zero advantage (no relative signal).
            pass
        advantages = a_o.copy()

    elif method_s == MethodName.LENS:
        a_o = group_normalize(r_o, eps=eps)
        wrong = r_o < 0.5
        c_term = group_normalize(conf, eps=eps) if lens_normalize else conf
        a_n = np.where(wrong, -gamma * c_term, 0.0)
        advantages = a_o + a_n

    elif method_s == MethodName.TRIAGE:
        a_o = group_normalize(r_o, eps=eps)
        correct = r_o >= 0.5
        a_p = _normalize_subset(r_p, correct, eps=eps)

        # LENS only on W_LOW
        c_term = group_normalize(conf, eps=eps) if lens_normalize else conf
        for i, cell in enumerate(cells):
            if cell == Cell.W_LOW:
                a_n[i] = -gamma * float(c_term[i])

        masks = build_sample_mask(cells, w_high_mode=w_high_mode)
        if w_high_mode == "downweight":
            for i, cell in enumerate(cells):
                if cell == Cell.W_HIGH:
                    a_h[i] = -beta * float(r_p[i])
        elif w_high_mode == "filter":
            # Masked samples get zero advantage contribution.
            for i, m in enumerate(masks):
                if not m:
                    a_o[i] = 0.0
                    a_p[i] = 0.0
                    a_n[i] = 0.0
                    a_h[i] = 0.0
        else:
            raise ValueError(
                f"w_high_mode must be 'filter' or 'downweight', got {w_high_mode}"
            )

        advantages = a_o + alpha * a_p + a_n + a_h
        # Ensure filtered rows stay at 0
        for i, m in enumerate(masks):
            if not m:
                advantages[i] = 0.0

    else:
        raise ValueError(f"Unknown method: {method_s}")

    return AdvantageOutput(
        advantages=[float(x) for x in advantages],
        cells=cells,
        masks=masks,
        a_outcome=[float(x) for x in a_o],
        a_process=[float(x) for x in a_p],
        a_neg=[float(x) for x in a_n],
        a_hack=[float(x) for x in a_h],
        method=method_s.value if isinstance(method_s, MethodName) else str(method_s),
    )


def advantages_to_list(out: AdvantageOutput) -> List[float]:
    return list(out.advantages)
