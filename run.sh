#!/usr/bin/env bash
# TRIAGE-GRPO one-entry launcher
# Usage: ./run.sh <command> [args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3)"
fi

cmd="${1:-help}"
shift || true

ensure_dev() {
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    "${ROOT}/.venv/bin/pip" install -e ".[dev]" -q
  else
    "${PY}" -m pip install -e ".[dev]" -q
  fi
}

case "${cmd}" in
  help|-h|--help)
    cat <<'EOF'
TRIAGE-GRPO runner

Setup / checks (no GPU):
  ./run.sh setup                 # create venv + install package
  ./run.sh unit                  # pytest
  ./run.sh check-advantage       # hand-check advantages
  ./run.sh smoke-cells           # synthetic 2x2 cell stats
  ./run.sh list                  # list all experiment run_ids

Download:
  ./run.sh download              # model 1.5B + DAPO-Math-17k + eval sets
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
  ./run.sh all                   # unit + download + verify + smoke + main + ablation + summarize
  ./run.sh all --dry-run         # same orchestration without real GPU train
  ./run.sh all --with-sensitivity
  ./run.sh all --with-extend

Docs:
  docs/HOW_TO_RUN.md
  docs/EXPERIMENT_MATRIX.md
EOF
    ;;

  setup)
    if [[ ! -d .venv ]]; then
      python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip setuptools wheel -q
    pip install -e ".[dev,download]" -q
    echo "Setup OK. Activate with: source .venv/bin/activate"
    ;;

  unit)
    ensure_dev
    "${PY}" -m pytest -q
    ;;

  check-advantage)
    ensure_dev
    "${PY}" scripts/unit_check_advantage.py
    ;;

  smoke-cells)
    ensure_dev
    "${PY}" scripts/smoke_cells.py
    ;;

  download)
    ensure_dev
    "${PY}" -m pip install -q huggingface_hub datasets
    "${PY}" scripts/download_assets.py "$@"
    ;;

  verify-assets)
    "${PY}" scripts/verify_assets.py
    ;;

  check-contamination)
    "${PY}" scripts/check_contamination.py
    ;;

  list)
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
    ensure_dev
    "${PY}" scripts/run_train.py "$@"
    ;;

  matrix)
    ensure_dev
    "${PY}" scripts/run_matrix.py "$@"
    ;;

  summarize)
    ensure_dev
    "${PY}" scripts/summarize_results.py
    ;;

  all)
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
    # shellcheck disable=SC1091
    source .venv/bin/activate
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
    echo "Unknown command: ${cmd}"
    echo "Run: ./run.sh help"
    exit 2
    ;;
esac
