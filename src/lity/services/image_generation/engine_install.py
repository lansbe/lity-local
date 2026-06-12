"""One-click install of the local image engine (torch + diffusers).

These are large, optional dependencies, so the app ships without them and pulls
them in on demand the first time the user opens image mode. We install into the
*running* interpreter (``sys.executable``) so the freshly installed packages are
importable in-process right after — no restart.

``uv`` is preferred (it created these venvs and is much faster); plain ``pip``
is the fallback, bootstrapped via ``ensurepip`` for the pip-less venvs ``uv``
produces.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from lity.services.image_generation.local_engine import ENGINE_PACKAGES
from lity.services.image_generation.mlx_engine import MLX_ENGINE_PACKAGES

logger = logging.getLogger(__name__)

# on_progress(percent_0_100, human_readable_line)
ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
        str(Path.home() / ".local" / "bin" / "uv"),
        str(Path.home() / ".cargo" / "bin" / "uv"),
    ):
        if Path(candidate).exists():
            return candidate
    return None


def _install_command(packages: tuple[str, ...]) -> list[str]:
    """``uv pip install`` if available, else ``python -m pip install``."""
    uv = _find_uv()
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, *packages]
    return [sys.executable, "-m", "pip", "install", *packages]


def _ensure_pip() -> None:
    """uv-made venvs ship no pip; bootstrap it so the pip fallback can run."""
    if importlib.util.find_spec("pip") is not None:
        return
    with subprocess.Popen(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as proc:
        proc.wait()


def _progress_for(line: str, current: int) -> int:
    """Coarse, monotonic percent from pip/uv stdout phase markers.

    There is no reliable global percentage in the tools' output, so we map
    phases to a rising number that never goes backwards and stays < 100 until
    the install actually finishes.
    """
    low = line.lower()
    if "resolved" in low:
        return max(current, 12)
    if "downloading" in low or "download" in low:
        return min(85, max(current + 6, 20))
    if "preparing" in low or "building" in low or "compiling" in low:
        return min(90, max(current + 4, 30))
    if "installed" in low or "installing" in low:
        return max(current, 95)
    return current


def install_mlx_engine(
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, object]:
    """Install the MLX engine (mflux) the same way as the diffusers engine."""
    return install_engine(on_progress, should_cancel, packages=MLX_ENGINE_PACKAGES)


def install_engine(
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    packages: tuple[str, ...] = ENGINE_PACKAGES,
) -> dict[str, object]:
    """Install an engine's packages, streaming progress. ``{"ok", "message"}``."""
    if _find_uv() is None:
        _ensure_pip()
    command = _install_command(packages)
    logger.info("Installing image engine: %s", " ".join(command))

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    percent = 2
    if on_progress:
        on_progress(percent, "Préparation de l'installation…")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:  # pragma: no cover - launcher missing
        return {"ok": False, "message": f"Impossible de lancer l'installation : {exc}"}

    last_line = ""
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if not line:
            continue
        last_line = line
        percent = _progress_for(line, percent)
        if on_progress:
            on_progress(percent, line)
        if should_cancel and should_cancel():
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
            return {"ok": False, "message": "Installation annulée."}

    process.wait()
    if process.returncode != 0:
        return {
            "ok": False,
            "message": f"Échec de l'installation du moteur image.\n{last_line}",
        }

    # Make the just-installed packages importable in this process.
    importlib.invalidate_caches()
    if on_progress:
        on_progress(100, "Moteur image installé.")
    return {"ok": True, "message": "Moteur image installé."}
