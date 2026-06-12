"""One-click install of local video engines.

These are large, optional dependencies, so the app ships without them and pulls
them in on demand the first time the user opens video mode. We install into the
*running* interpreter (``sys.executable``) so the freshly installed packages are
importable in-process right after — no restart. Mirrors the image engine
installer for diffusers.

The MLX backend is different: ``ltx-2-mlx`` is a standalone runtime. Lity clones
it into the app cache and runs ``uv sync --all-extras`` there, then invokes its
private CLI in a subprocess for generation.
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

from lity.infrastructure.paths import AppPaths
from lity.services.video_generation.local_engine import ENGINE_PACKAGES
from lity.services.video_generation.mlx_engine import (
    mlx_video_runtime_dir,
    mlx_video_supported_platform,
)

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


def _install_command() -> list[str]:
    """``uv pip install`` if available, else ``python -m pip install``."""
    packages = list(ENGINE_PACKAGES)
    uv = _find_uv()
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, *packages]
    return [sys.executable, "-m", "pip", "install", *packages]


def _git_command() -> str | None:
    found = shutil.which("git")
    if found:
        return found
    return "/usr/bin/git" if Path("/usr/bin/git").exists() else None


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
    """Coarse, monotonic percent from pip/uv stdout phase markers."""
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


def install_video_engine(
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, object]:
    """Install the engine, streaming progress. Returns ``{"ok", "message"}``."""
    if _find_uv() is None:
        _ensure_pip()
    command = _install_command()
    logger.info("Installing video engine: %s", " ".join(command))

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
            "message": f"Échec de l'installation du moteur vidéo.\n{last_line}",
        }

    # Make the just-installed packages importable in this process.
    importlib.invalidate_caches()
    if on_progress:
        on_progress(100, "Moteur vidéo installé.")
    return {"ok": True, "message": "Moteur vidéo installé."}


def install_mlx_video_engine(
    paths: AppPaths,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, object]:
    """Install/update the external ltx-2-mlx runtime in Lity's cache."""
    if not mlx_video_supported_platform():
        return {
            "ok": False,
            "message": "Le runtime vidéo MLX nécessite macOS Apple Silicon et Python 3.11+.",
        }
    uv = _find_uv()
    if uv is None:
        return {"ok": False, "message": "Le runtime vidéo MLX nécessite uv pour s'installer."}
    git = _git_command()
    if git is None:
        return {"ok": False, "message": "Le runtime vidéo MLX nécessite git pour cloner ltx-2-mlx."}

    runtime = mlx_video_runtime_dir(paths)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(2, "Préparation du runtime ltx-2-mlx…")

    if not (runtime / ".git").is_dir():
        result = _run_step(
            [
                git,
                "clone",
                "--depth",
                "1",
                "https://github.com/dgrauet/ltx-2-mlx.git",
                str(runtime),
            ],
            cwd=runtime.parent,
            start=5,
            end=35,
            on_progress=on_progress,
            should_cancel=should_cancel,
            label="Téléchargement du runtime ltx-2-mlx…",
        )
        if not result["ok"]:
            return result
    else:
        result = _run_step(
            [git, "pull", "--ff-only"],
            cwd=runtime,
            start=5,
            end=25,
            on_progress=on_progress,
            should_cancel=should_cancel,
            label="Mise à jour du runtime ltx-2-mlx…",
        )
        if not result["ok"]:
            return result

    result = _run_step(
        [uv, "sync", "--all-extras"],
        cwd=runtime,
        start=35,
        end=98,
        on_progress=on_progress,
        should_cancel=should_cancel,
        label="Installation des dépendances MLX vidéo…",
    )
    if not result["ok"]:
        return result

    importlib.invalidate_caches()
    if on_progress:
        on_progress(100, "Runtime vidéo MLX installé.")
    return {"ok": True, "message": "Runtime vidéo MLX installé."}


def _run_step(
    command: list[str],
    *,
    cwd: Path,
    start: int,
    end: int,
    on_progress: ProgressCallback | None,
    should_cancel: CancelCallback | None,
    label: str,
) -> dict[str, object]:
    logger.info("Installing video MLX runtime: %s", " ".join(command))
    if on_progress:
        on_progress(start, label)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:  # pragma: no cover - launcher missing
        return {"ok": False, "message": f"Impossible de lancer l'installation : {exc}"}

    percent = start
    last_line = ""
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if not line:
            continue
        last_line = line
        mapped = _progress_for(line, percent)
        if mapped == percent:
            mapped = percent + 3
        percent = min(end, max(start, mapped))
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
        return {"ok": False, "message": f"Échec de l'installation vidéo MLX.\n{last_line}"}
    return {"ok": True, "message": last_line or label}
