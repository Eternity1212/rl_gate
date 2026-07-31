#!/usr/bin/env python3
"""Smoke: classify synthetic rollouts and dump cell stats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage_grpo.advantage import compute_group_advantages
from triage_grpo.metrics import cell_stats, hack_rate
from triage_grpo.types import GroupInput


def main() -> int:
    out_dir = ROOT / "outputs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    group = GroupInput(
        outcome_rewards=[1, 1, 1, 0, 0, 0, 0, 1],
        process_scores=[0.9, 0.8, 0.2, 0.1, 0.15, 0.85, 0.7, 0.55],
        confidences=[0.9, 0.8, 0.5, 0.2, 0.4, 0.95, 0.7, 0.6],
    )
    out = compute_group_advantages(group, method="triage", tau=0.5, w_high_mode="filter")
    payload = {
        "cells": [c.value for c in out.cells],
        "masks": out.masks,
        "advantages": out.advantages,
        "cell_stats": cell_stats(out.cells),
        "hack_rate": hack_rate(out.cells),
    }
    path = out_dir / "cell_stats.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["cell_stats"], indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
