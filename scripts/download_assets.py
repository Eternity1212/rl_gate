#!/usr/bin/env python3
"""One-click download for models + train/eval datasets.

Usage:
  python scripts/download_assets.py
  python scripts/download_assets.py --with-7b
  python scripts/download_assets.py --skip-model --skip-eval
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MANIFEST = ASSETS / "manifest.json"


def _ensure_deps() -> None:
    missing = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    if missing:
        print("Installing:", ", ".join(missing))
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )


def download_model(repo_id: str, local_dir: Path) -> dict:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] {repo_id} -> {local_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return {"repo_id": repo_id, "path": path, "ok": True}


def download_dataset(repo_id: str, local_dir: Path) -> dict:
    from datasets import load_dataset

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] {repo_id} -> {local_dir}")
    try:
        ds = load_dataset(repo_id)
        out = local_dir / "hf_dataset"
        out.mkdir(parents=True, exist_ok=True)
        # save_to_disk for offline reuse
        ds.save_to_disk(str(out))
        # also dump a small jsonl preview
        split = "train" if "train" in ds else list(ds.keys())[0]
        preview_path = local_dir / "preview.jsonl"
        n = min(20, len(ds[split]))
        with preview_path.open("w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps(ds[split][i], ensure_ascii=False) + "\n")
        return {
            "repo_id": repo_id,
            "path": str(out),
            "split": split,
            "num_rows": len(ds[split]),
            "ok": True,
        }
    except Exception as e:  # noqa: BLE001
        note = local_dir / "DOWNLOAD_FAILED.txt"
        note.write_text(
            f"Failed to download {repo_id}: {e}\n"
            "Replace with a working HF dataset id or place files manually.\n",
            encoding="utf-8",
        )
        print(f"[warn] {repo_id}: {e}")
        return {"repo_id": repo_id, "ok": False, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-7b", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    _ensure_deps()
    ASSETS.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "train": {},
        "eval": {},
    }

    if not args.skip_model:
        manifest["models"]["1.5b"] = download_model(
            "Qwen/Qwen2.5-Math-1.5B",
            ASSETS / "models" / "Qwen2.5-Math-1.5B",
        )
        if args.with_7b:
            manifest["models"]["7b"] = download_model(
                "Qwen/Qwen2.5-Math-7B",
                ASSETS / "models" / "Qwen2.5-Math-7B",
            )

    if not args.skip_train:
        # Prefer official DAPO; fall back to processed
        train = download_dataset(
            "BytedTsinghua-SIA/DAPO-Math-17k",
            ASSETS / "data" / "DAPO-Math-17k",
        )
        if not train.get("ok"):
            print("[info] falling back to open-r1/DAPO-Math-17k-Processed")
            train = download_dataset(
                "open-r1/DAPO-Math-17k-Processed",
                ASSETS / "data" / "DAPO-Math-17k",
            )
        manifest["train"] = train

    if not args.skip_eval:
        eval_ids = {
            "MATH-500": "HuggingFaceH4/MATH-500",
            "AIME24": "HuggingFaceH4/aime_2024",
            "AIME25": "MathArena/aime_2025",
            "AMC23": "math-ai/amc23",
            "OlympiadBench": "Hothan/OlympiadBench",
            "MinervaMath": "math-ai/minervamath",
        }
        for name, repo in eval_ids.items():
            manifest["eval"][name] = download_dataset(
                repo, ASSETS / "data" / "eval" / name
            )

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {MANIFEST}")
    ok_train = manifest.get("train", {}).get("ok", args.skip_train)
    ok_model = True
    if not args.skip_model:
        ok_model = manifest["models"].get("1.5b", {}).get("ok", False)
    if ok_train and ok_model:
        print("DOWNLOAD OK (core assets ready)")
        return 0
    print("DOWNLOAD PARTIAL — check manifest / DOWNLOAD_FAILED.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
