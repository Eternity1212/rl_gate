#!/usr/bin/env bash
# TRIAGE-GRPO one-entry launcher
# Usage: ./run.sh <command> [args...]
#
# Uses system python/pip by default (no .venv required).
# Optional: TRIAGE_USE_VENV=1  → prefer ./.venv if it exists and is complete
# Optional: TRIAGE_SKIP_ENSURE=1 → do not auto pip install on each command
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

resolve_python() {
  if [[ "${TRIAGE_USE_VENV:-0}" == "1" && -x "${ROOT}/.venv/bin/python" ]]; then
    echo "${ROOT}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "ERROR: python3/python not found in PATH" >&2
  exit 1
}

PY="$(resolve_python)"

pip_install() {
  # Always go through "python -m pip" (works without .venv/bin/pip)
  "${PY}" -m pip install "$@"
}

ensure_dev() {
  if [[ "${TRIAGE_SKIP_ENSURE:-0}" == "1" ]]; then
    return 0
  fi
  # Quiet reinstall of package extras; safe if user already pip-installed deps
  pip_install -e ".[dev,download]" -q
}

case "${1:-help}" in
  help|-h|--help)
    shift || true
    cat <<'EOF'
TRIAGE-GRPO runner (system python OK — no .venv required)

Env:
  TRIAGE_SKIP_ENSURE=1   skip auto pip on each command (if you already pip installed)
  TRIAGE_USE_VENV=1      prefer ./.venv/bin/python when present

Setup / checks:
  ./run.sh setup                 # pip install -e ".[dev,download]" into CURRENT python
  ./run.sh unit                  # pytest
  ./run.sh check-advantage
  ./run.sh smoke-cells
  ./run.sh list

Download:
  ./run.sh download
  ./run.sh download --with-7b
  ./run.sh verify-assets
  ./run.sh check-contamination

Train / matrix:
  ./run.sh train --run S2.triage --steps 200
  ./run.sh train --run Ours.triage.s0 --dry-run
  ./run.sh matrix --stage smoke
  ./run.sh matrix --stage main --dry-run
  ./run.sh matrix --stage ablation
  ./run.sh summarize

Full pipeline:
  ./run.sh all
  ./run.sh all --dry-run
  ./run.sh all --with-sensitivity
  ./run.sh all --with-extend
EOF
    ;;

  setup)
    shift || true
    "${PY}" -m pip install -U pip setuptools wheel -q
    pip_install -e ".[dev,download]" -q
    echo "Setup OK."
    echo "Python: ${PY}"
    echo "No venv activation needed if you install into this interpreter."
    ;;

  unit)
    shift || true
    ensure_dev
    "${PY}" -m pytest -q
    ;;

  check-advantage)
    shift || true
    ensure_dev
    "${PY}" scripts/unit_check_advantage.py
    ;;

  smoke-cells)
    shift || true
    ensure_dev
    "${PY}" scripts/smoke_cells.py
    ;;

  download)
    shift || true
    ensure_dev
    pip_install -q huggingface_hub datasets
    "${PY}" scripts/download_assets.py "$@"
    ;;

  verify-assets)
    shift || true
    "${PY}" scripts/verify_assets.py
    ;;

  check-contamination)
    shift || true
    "${PY}" scripts/check_contamination.py
    ;;

  list)
    shift || true
    "${PY}" - <<'PY'
import yaml
from pathlib import Path
reg = yaml.safe_load(Path("configs/experiment_registry.yaml").read_text())
for stage, runs in reg.get("stages", {}).items():
    print(f"[{stage}] {len(runs)} runs")
    for r in runs:
        print(f"  {r['id']:20s}  method={r.get('method')}  seed={r.get('seed')}")
PY
    ;;

  train)
    shift || true
    ensure_dev
    "${PY}" scripts/run_train.py "$@"
    ;;

  matrix)
    shift || true
    ensure_dev
    "${PY}" scripts/run_matrix.py "$@"
    ;;

  summarize)
    shift || true
    ensure_dev
    "${PY}" scripts/summarize_results.py
    ;;

  all)
    shift || true
    DRY=()
    EXTRA_SENS=0
    EXTRA_EXT=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --dry-run) DRY=(--dry-run); shift ;;
        --with-sensitivity) EXTRA_SENS=1; shift ;;
        --with-extend) EXTRA_EXT=1; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
      esac
    done
    ./run.sh setup
    ./run.sh unit
    ./run.sh check-advantage
    ./run.sh smoke-cells
    ./run.sh download || true
    ./run.sh verify-assets || true
    ./run.sh check-contamination || true
    ./run.sh matrix --stage smoke "${DRY[@]}"
    ./run.sh matrix --stage main "${DRY[@]}"
    ./run.sh matrix --stage ablation "${DRY[@]}"
    if [[ "${EXTRA_SENS}" == "1" ]]; then
      ./run.sh matrix --stage sensitivity "${DRY[@]}"
    fi
    if [[ "${EXTRA_EXT}" == "1" ]]; then
      ./run.sh matrix --stage extend "${DRY[@]}"
    fi
    ./run.sh summarize
    echo "ALL PIPELINE FINISHED (see outputs/)"
    ;;

  *)
    echo "Unknown command: ${1:-}"
    echo "Run: ./run.sh help"
    exit 2
    ;;
esac
