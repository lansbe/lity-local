#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
  UV=(python3 -m uv)
else
  echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

"${UV[@]}" sync --extra desktop --extra web --extra packaging

# Build the web UI so PyInstaller can bundle it (web_dist). Skipped with a
# warning if npm is missing; the Qt UI still packages without it.
if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install && npm run build)
else
  echo "WARNING: npm not found — skipping web UI build."
  echo "         The packaged app's '--ui web' mode needs frontend/ built first."
fi

"${UV[@]}" run pyinstaller packaging/pyinstaller/lity.spec --noconfirm

echo "Build complete: dist/Lity"
