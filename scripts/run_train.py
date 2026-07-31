#!/usr/bin/env python3
"""Training launcher.

Phase B: wires into veRL when available.
Until then, creates a dry-run job card under outputs/<run_id>/ so the matrix
orchestration is usable end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_registry() -> dict:
    path = ROOT / "configs" / "experiment_registry.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_run(registry: dict, run_id: str) -> Optional[Dict]:
    for stage, runs in registry.get("stages", {}).items():
        for r in runs:
            if r["id"] == run_id:
                out = dict(r)
                out["stage"] = stage
                return out
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run_id from experiment_registry.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="force dry-run even if veRL exists")
    args = parser.parse_args()

    registry = load_registry()
    run = find_run(registry, args.run)
    if run is None:
        print(f"Unknown run_id: {args.run}")
        print("List with: ./run.sh list")
        return 2

    steps = args.steps
    if steps is None:
        steps = run.get("steps", registry.get("defaults", {}).get("full_steps", 1000))

    out_dir = ROOT / "outputs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "run_id": args.run,
        "stage": run.get("stage"),
        "method": run.get("method"),
        "config": run.get("config"),
        "seed": run.get("seed"),
        "steps": steps,
        "overrides": run.get("overrides", {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run",
    }

    # If TRIAGE_FORCE_DRYRUN=1 or --dry-run or veRL missing → dry run
    verl_ok = False
    try:
        import verl  # noqa: F401

        verl_ok = True
    except Exception:
        verl_ok = False

    force_dry = args.dry_run or os.environ.get("TRIAGE_FORCE_DRYRUN") == "1"
    job_path = out_dir / "job.json"

    if (not verl_ok) or force_dry or run.get("method") == "base_eval":
        job["status"] = "dry_run"
        job["message"] = (
            "veRL not installed or dry-run requested. "
            "Install veRL and re-run without --dry-run for real GPU training. "
            "See docs/HOW_TO_RUN.md Phase B."
        )
        job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(job, indent=2, ensure_ascii=False))
        print(f"Wrote {job_path}")
        # dry-run is success for orchestration bring-up
        return 0

    # Placeholder for real veRL entry (Phase B implementation)
    job["status"] = "error"
    job["message"] = (
        "veRL detected but trainer entry not yet wired. "
        "Implement scripts/verl_train_entry.py in Phase B."
    )
    job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(job, indent=2, ensure_ascii=False))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
