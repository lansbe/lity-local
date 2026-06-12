from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_macos_dev_script_bootstraps_frontend_and_python_dev() -> None:
    script = ROOT / "scripts" / "dev_macos.sh"

    assert script.exists()
    assert os.access(script, os.X_OK)

    text = script.read_text(encoding="utf-8")
    assert "uv sync --extra desktop --extra web --extra dev --extra packaging" in text
    assert "frontend/node_modules/vite/dist/node/cli.js" in text
    assert "rm -rf" in text
    assert "npm ci" in text
    assert "npm run dev" in text
    assert "free_vite_port" in text
    assert 'lsof -nP -iTCP:"${VITE_PORT}" -sTCP:LISTEN -t' in text
    assert "kill -TERM" in text
    assert "kill -KILL" in text
    assert "APP_PID" in text
    assert "trap '' INT" in text
    assert "wait_for_desktop_app" in text
    assert "uv run lity --ui web --dev" in text
    assert "trap cleanup EXIT INT TERM" in text

    assert "install_frontend_deps\n  free_vite_port\n  start_vite" in text
    assert "wait_for_vite\n  run_desktop_app" in text
    assert 'run_desktop_app "$@"\n  wait_for_desktop_app' in text
