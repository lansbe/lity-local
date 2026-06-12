from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def candidate_dirs(settings: Any) -> list[Path]:
    candidates: list[Path] = []
    configured = getattr(settings, "install_dir", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    for env_name in ("STABLE_DIFFUSION_WEBUI_HOME", "SD_WEBUI_HOME"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            candidates.append(Path(env_value).expanduser())

    home = Path.home()
    candidates.extend(
        [
            home / "Documents" / "stable-diffusion-webui-forge",
            home / "Desktop" / "stable-diffusion-webui-forge",
            home / "stable-diffusion-webui-forge",
            home / "Applications" / "stable-diffusion-webui-forge",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        if resolved not in seen and candidate.exists() and candidate.is_dir():
            unique.append(candidate)
            seen.add(resolved)
    return unique
