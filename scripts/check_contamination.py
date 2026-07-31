#!/usr/bin/env python3
"""Lightweight train/eval prompt overlap check.

Uses local assets if present. If eval download failed, writes a TODO report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ROOT / "outputs" / "contamination_report.json"


def _load_preview_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    preview = path / "preview.jsonl"
    if not preview.exists():
        return prompts
    for line in preview.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        for key in ("prompt", "question", "problem", "input"):
            if key in obj and isinstance(obj[key], str):
                prompts.add(obj[key].strip())
                break
    return prompts


def main() -> int:
    train_dir = ASSETS / "data" / "DAPO-Math-17k"
    eval_root = ASSETS / "data" / "eval"
    train_prompts = _load_preview_prompts(train_dir)

    report = {
        "train_preview_n": len(train_prompts),
        "overlaps": {},
        "note": "Uses preview.jsonl (first 20 rows). Full check needs full disk datasets.",
    }
    if not train_prompts:
        report["status"] = "SKIP"
        report["reason"] = "train preview missing — run ./run.sh download first"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    total_overlap = 0
    if eval_root.exists():
        for bench in sorted(p.name for p in eval_root.iterdir() if p.is_dir()):
            ep = _load_preview_prompts(eval_root / bench)
            inter = sorted(train_prompts & ep)
            report["overlaps"][bench] = {"n": len(inter), "examples": inter[:3]}
            total_overlap += len(inter)

    report["total_overlap_preview"] = total_overlap
    report["status"] = "OK" if total_overlap == 0 else "WARN"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
