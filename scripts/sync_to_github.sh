#!/usr/bin/env bash
# Manually sync current branch (and optionally main) to GitHub.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "ERROR: origin not set. Expected https://github.com/Eternity1212/rl_gate.git"
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
echo "Pushing ${branch}..."
git push -u origin "HEAD:${branch}"

if [[ "${1:-}" == "--also-main" && "${branch}" != "main" ]]; then
  echo "Also updating origin/main from ${branch}..."
  git push origin "HEAD:main"
fi

echo "Done. Repo: https://github.com/Eternity1212/rl_gate"
git status -sb
