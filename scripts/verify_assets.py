#!/usr/bin/env python3
"""Verify downloaded assets exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MANIFEST = ASSETS / "manifest.json"


def main() -> int:
    errors = []
    if not MANIFEST.exists():
        errors.append(f"missing {MANIFEST} — run: ./run.sh download")
    else:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        model = ASSETS / "models" / "Qwen2.5-Math-1.5B"
        if not model.exists():
            errors.append(f"missing model dir {model}")
        train = ASSETS / "data" / "DAPO-Math-17k"
        if not train.exists():
            errors.append(f"missing train data {train}")
        if not manifest.get("train", {}).get("ok", False):
            errors.append("manifest.train.ok is false")

    if errors:
        print("VERIFY FAIL:")
        for e in errors:
            print(" -", e)
        return 1
    print("VERIFY OK")
    print(f"manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
