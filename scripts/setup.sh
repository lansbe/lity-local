#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
  UV=(python3 -m uv)
else
  echo "uv is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

"${UV[@]}" sync --extra desktop --extra web --extra dev --extra packaging
echo "Setup complete. Run: uv run lity"
