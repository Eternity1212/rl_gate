#!/usr/bin/env python3
"""Load a YAML experiment config and print triage section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        default=Path("configs/triage_grpo_1.5b.yaml"),
        nargs="?",
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(json.dumps(cfg.get("triage", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
