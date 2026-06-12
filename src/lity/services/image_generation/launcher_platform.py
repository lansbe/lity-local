from __future__ import annotations

import sys
from pathlib import Path


def launch_command(install_dir: Path, platform_name: str = sys.platform) -> list[str] | None:
    if platform_name.startswith("win"):
        for script_name in ("webui-user.bat", "webui.bat"):
            script = install_dir / script_name
            if script.exists():
                return ["cmd", "/c", str(script), "--api"]
        return None

    for script_name in ("webui.sh", "webui-user.sh"):
        script = install_dir / script_name
        if script.exists():
            return ["bash", str(script), "--api"]
    return None
