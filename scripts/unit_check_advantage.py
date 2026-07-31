#!/usr/bin/env python3
"""Hand-check TRIAGE advantages on a tiny synthetic group."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage_grpo.advantage import compute_group_advantages
from triage_grpo.metrics import cell_stats, hack_rate
from triage_grpo.types import GroupInput


def main() -> None:
    group = GroupInput(
        outcome_rewards=[1, 1, 0, 0],
        process_scores=[0.90, 0.20, 0.10, 0.85],
        confidences=[0.80, 0.70, 0.30, 0.90],
    )
    methods = ["orm_grpo", "p_grpo", "papo", "prof", "lens", "triage"]
    report = {}
    for m in methods:
        out = compute_group_advantages(
            group,
            method=m,
            tau=0.5,
            alpha=1.0,
            beta=1.0,
            gamma=0.1,
            w_high_mode="filter",
        )
        report[m] = {
            "cells": [c.value for c in out.cells],
            "masks": out.masks,
            "advantages": [round(a, 6) for a in out.advantages],
            "a_process": [round(a, 6) for a in out.a_process],
            "a_neg": [round(a, 6) for a in out.a_neg],
            "hack_rate": hack_rate(out.cells),
            "cell_stats": cell_stats(out.cells),
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    triage = report["triage"]
    assert triage["cells"] == ["C_HIGH", "C_LOW", "W_LOW", "W_HIGH"]
    assert triage["masks"][-1] is False
    print("\nOK: TRIAGE four-cell + W_HIGH filter looks correct.")


if __name__ == "__main__":
    main()
