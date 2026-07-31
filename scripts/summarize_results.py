#!/usr/bin/env python3
"""Summarize job cards under outputs/ into CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES = OUT / "tables"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in OUT.glob("*/job.json"):
        data = json.loads(job.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": data.get("run_id"),
                "stage": data.get("stage"),
                "method": data.get("method"),
                "seed": data.get("seed"),
                "steps": data.get("steps"),
                "status": data.get("status"),
                "config": data.get("config"),
            }
        )
    rows.sort(key=lambda r: (str(r["stage"]), str(r["run_id"])))
    path = TABLES / "runs.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run_id", "stage", "method", "seed", "steps", "status", "config"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
    # placeholders for paper tables
    for name in ("main.csv", "ablation.csv", "hack_rate.csv"):
        p = TABLES / name
        if not p.exists():
            p.write_text(
                "run_id,metric,value,note\n"
                "# fill after real eval metrics are available\n",
                encoding="utf-8",
            )
            print(f"Created placeholder {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
