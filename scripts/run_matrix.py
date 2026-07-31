#!/usr/bin/env python3
"""Run a stage of experiments from experiment_registry.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["smoke", "main", "ablation", "sensitivity", "extend"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    reg = yaml.safe_load((ROOT / "configs" / "experiment_registry.yaml").read_text())
    runs = reg.get("stages", {}).get(args.stage, [])
    if not runs:
        print(f"No runs for stage={args.stage}")
        return 1

    print(f"=== matrix stage={args.stage} n={len(runs)} ===")
    failed = []
    for r in runs:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_train.py"), "--run", r["id"]]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.steps is not None:
            cmd.extend(["--steps", str(args.steps)])
        elif "steps" in r:
            cmd.extend(["--steps", str(r["steps"])])
        print("\n>>", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append(r["id"])

    print("\n=== done ===")
    print("failed:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
