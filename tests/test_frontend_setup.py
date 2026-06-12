from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def test_vite_dev_mock_is_opt_in() -> None:
    main = (FRONTEND / "src" / "main.tsx").read_text(encoding="utf-8")

    assert "lity_mock" in main
    assert "const shouldInstallDevMock" in main
    assert "if (import.meta.env.DEV && shouldInstallDevMock)" in main
    assert "installDevBridge()" in main


def test_frontend_uses_vite_line_without_known_audit_findings() -> None:
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))

    assert package["devDependencies"]["vite"].startswith("^6.4.")


def test_frontend_approves_expected_native_install_scripts() -> None:
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    allow_scripts = package["allowScripts"]

    assert any(
        name.startswith("esbuild@") and allowed is True for name, allowed in allow_scripts.items()
    )
    assert any(
        name.startswith("fsevents@") and allowed is True for name, allowed in allow_scripts.items()
    )
