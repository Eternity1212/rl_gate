#!/usr/bin/env bash
# Install a local post-commit hook that pushes the current branch to origin.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK="$ROOT/.git/hooks/post-commit"

mkdir -p "$ROOT/.git/hooks"
cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Auto-sync: push current branch to origin after each successful commit.
set -euo pipefail

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
  exit 0
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "[autosync] no origin remote; skip push"
  exit 0
fi

echo "[autosync] pushing ${branch} -> origin/${branch}"
# Prefer upstream if set; otherwise create/update same-named remote branch.
if git rev-parse --abbrev-ref "@{u}" >/dev/null 2>&1; then
  git push origin "HEAD"
else
  git push -u origin "HEAD:${branch}"
fi
EOF

chmod +x "$HOOK"
echo "Installed post-commit autosync hook at ${HOOK}"
