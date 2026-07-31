"""Thin veRL integration surface (Phase B wiring).

This module does NOT import veRL at import time. It exposes pure helpers that a
veRL reward manager / advantage estimator can call once tensors are reduced to
Python lists per prompt group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from triage_grpo.advantage import compute_group_advantages
from triage_grpo.metrics import cell_stats, frac_masked, hack_rate
from triage_grpo.types import AdvantageOutput, GroupInput


@dataclass
class TriageConfig:
    method: str = "triage"
    tau: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1
    w_high_mode: str = "filter"
    lens_normalize: bool = True
    hybrid_lambda: float = 0.5
    eps: float = 1e-6

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "TriageConfig":
        cfg = cfg or {}
        return cls(
            method=str(cfg.get("method", "triage")),
            tau=float(cfg.get("tau", 0.5)),
            alpha=float(cfg.get("alpha", 1.0)),
            beta=float(cfg.get("beta", 1.0)),
            gamma=float(cfg.get("gamma", 0.1)),
            w_high_mode=str(cfg.get("w_high_mode", "filter")),
            lens_normalize=bool(cfg.get("lens_normalize", True)),
            hybrid_lambda=float(cfg.get("hybrid_lambda", 0.5)),
            eps=float(cfg.get("eps", 1e-6)),
        )


def compute_advantages_for_verl_group(
    outcome_rewards: Sequence[float],
    process_scores: Sequence[float],
    confidences: Optional[Sequence[float]] = None,
    triage_cfg: Optional[Mapping[str, Any]] = None,
) -> AdvantageOutput:
    """Main hook: replace default GRPO advantage with TRIAGE/baselines."""
    cfg = TriageConfig.from_mapping(triage_cfg)
    return compute_group_advantages(
        GroupInput(
            outcome_rewards=outcome_rewards,
            process_scores=process_scores,
            confidences=confidences,
        ),
        method=cfg.method,
        tau=cfg.tau,
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        w_high_mode=cfg.w_high_mode,
        lens_normalize=cfg.lens_normalize,
        hybrid_lambda=cfg.hybrid_lambda,
        eps=cfg.eps,
    )


def attach_diagnostics(
    batch_metrics: MutableMapping[str, float],
    out: AdvantageOutput,
    prefix: str = "triage/",
) -> Dict[str, float]:
    """Merge cell/hack diagnostics into a logger dict."""
    stats = cell_stats(out.cells)
    payload = {
        f"{prefix}hack_rate": hack_rate(out.cells),
        f"{prefix}frac_masked": frac_masked(out.masks),
        f"{prefix}pct_c_high": stats["pct_c_high"],
        f"{prefix}pct_c_low": stats["pct_c_low"],
        f"{prefix}pct_w_low": stats["pct_w_low"],
        f"{prefix}pct_w_high": stats["pct_w_high"],
    }
    batch_metrics.update(payload)
    return payload


# ---------------------------------------------------------------------------
# Suggested veRL wiring (pseudo-integration checklist for Phase B)
# ---------------------------------------------------------------------------
#
# 1) Reward manager (per response):
#       non_tensor_batch["r_o"][i] = score_outcome_exact(resp, gold)
#       non_tensor_batch["r_p"][i] = process_scorer.score(prompt, resp)
#       non_tensor_batch["confidence"][i] = mean_token_prob(resp)
#
# 2) After gathering G responses for the same uid/prompt:
#       out = compute_advantages_for_verl_group(r_o, r_p, conf, cfg.triage)
#       advantages[group] = out.advantages
#       loss_weights[group] = out.masks  # False -> 0 weight
#
# 3) Logging each step:
#       attach_diagnostics(metrics, out)
#
# Do not import verl here until Phase B pins a version.
