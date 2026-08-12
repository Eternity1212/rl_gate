#!/usr/bin/env python3
"""Training launcher.

Prefers the TRL GRPO entry (``scripts/trl_train_entry.py``). Falls back to a
dry-run job card when TRL/torch are unavailable or ``--dry-run`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    parser.add_argument("--dry-run", action="store_true", help="force dry-run job card only")
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
        "backend": "trl.GRPOTrainer",
    }
    job_path = out_dir / "job.json"

    force_dry = args.dry_run or os.environ.get("TRIAGE_FORCE_DRYRUN") == "1"
    if force_dry or run.get("method") == "base_eval":
        job["status"] = "dry_run"
        job["message"] = (
            "dry-run or base_eval: no GPU train. "
            "For real train: omit --dry-run and ensure torch/trl/peft installed."
        )
        job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(job, indent=2, ensure_ascii=False))
        print(f"Wrote {job_path}")
        return 0

    # Need torch + transformers + peft. TRL is optional (3.9 uses grpo39).
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        job["status"] = "error"
        job["message"] = (
            f"torch/transformers/peft not importable ({e}). "
            "pip install -r requirements.txt && pip install -e ."
        )
        job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(job, indent=2, ensure_ascii=False))
        return 3

    smoke = str(run.get("stage")) == "smoke" or int(steps) <= 200
    cfg = run.get("config") or "configs/triage_grpo_1.5b.yaml"
    # On Python < 3.10, force grpo39 to avoid trl `|` annotation crash
    env = os.environ.copy()
    if sys.version_info < (3, 10):
        env["TRIAGE_FORCE_GRPO39"] = "1"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "trl_train_entry.py"),
        "--run-id",
        args.run,
        "--config",
        str(cfg),
        "--method",
        str(run.get("method") or "triage"),
        "--seed",
        str(run.get("seed", 0)),
        "--steps",
        str(steps),
        "--overrides-json",
        json.dumps(run.get("overrides") or {}),
    ]
    if smoke:
        cmd.append("--smoke")

    job["status"] = "running"
    job["command"] = cmd
    job["backend"] = "grpo39_or_trl_auto"
    job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Launching:", " ".join(cmd), flush=True)
    print("Python:", sys.version, "TRIAGE_FORCE_GRPO39=", env.get("TRIAGE_FORCE_GRPO39"), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    job["status"] = "finished" if proc.returncode == 0 else "error"
    job["returncode"] = proc.returncode
    job["finished_at"] = datetime.now(timezone.utc).isoformat()
    job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
