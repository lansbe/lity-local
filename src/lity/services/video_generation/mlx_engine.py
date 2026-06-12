"""Local MLX video generation through the ltx-2-mlx CLI.

``ltx-2-mlx`` is a separate Apple-Silicon runtime with its own dependency graph.
Keep it outside the app's Python process and render through a short-lived CLI
subprocess, mirroring the image ``mflux`` integration. This avoids mixing MLX,
torch, diffusers, and their compiled wheels in one interpreter while still
making the flow one-click from the app.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lity.infrastructure.paths import AppPaths

logger = logging.getLogger(__name__)

_DEFAULT_COMMAND = "ltx-2-mlx"


def mlx_video_runtime_dir(paths: AppPaths) -> Path:
    """Private clone/sync location for the external ltx-2-mlx runtime."""
    return paths.cache_dir / "engines" / "ltx-2-mlx"


def mlx_video_supported_platform() -> bool:
    """ltx-2-mlx/MLX run only on Apple Silicon with Python 3.11+."""
    return (
        sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info >= (3, 11)
    )


def mlx_video_dependencies_available(paths: AppPaths, command: str = _DEFAULT_COMMAND) -> bool:
    """True when the CLI needed by the selected MLX model is resolvable."""
    return resolve_ltx_mlx_cli(paths, command or _DEFAULT_COMMAND) is not None


def resolve_ltx_mlx_cli(paths: AppPaths, command: str = _DEFAULT_COMMAND) -> str | None:
    """Locate the ltx-2-mlx console script.

    The one-click installer creates a dedicated uv environment under
    ``cache/engines/ltx-2-mlx/.venv``. GUI apps often have a sparse PATH, so
    check that private venv before falling back to PATH.
    """
    runtime = mlx_video_runtime_dir(paths)
    exe = "Scripts" if sys.platform == "win32" else "bin"
    candidate = runtime / ".venv" / exe / command
    if candidate.exists():
        return str(candidate)
    found = shutil.which(command)
    return found if found else None


def _snap16(value: int) -> int:
    return max(256, int(round(value / 16)) * 16)


def _snap_ltx_frames(value: int) -> int:
    """LTX temporal VAE requires frame counts of the form ``8k + 1``."""
    frames = max(9, int(value))
    return ((frames - 1) // 8) * 8 + 1


class MlxVideoEngine:
    """Render video from MLX-format LTX weights via ``ltx-2-mlx generate``."""

    def __init__(self, paths: AppPaths):
        self.paths = paths

    def unload(self) -> None:
        """No resident state — each render is its own subprocess."""
        return None

    def build_command(
        self,
        model_dir: Path,
        params: dict[str, Any],
        seed: int,
        mlx: dict[str, Any],
        output: Path,
    ) -> list[str]:
        command = str((mlx or {}).get("command") or _DEFAULT_COMMAND)
        cli = resolve_ltx_mlx_cli(self.paths, command)
        if cli is None:
            raise RuntimeError(
                f"Commande ltx-2-mlx introuvable : « {command} ». "
                "Installe le runtime vidéo MLX depuis l'onglet Vidéos."
            )

        args = [
            cli,
            "generate",
            "--prompt",
            str(params.get("prompt", "")),
            "--model",
            str(model_dir),
            "--output",
            str(output),
            "--seed",
            str(int(seed)),
            "--height",
            str(_snap16(int(params.get("height", 512)))),
            "--width",
            str(_snap16(int(params.get("width", 768)))),
            "--frames",
            str(_snap_ltx_frames(int(params.get("num_frames", 49)))),
        ]

        mode = str((mlx or {}).get("mode") or "distilled").strip().lower()
        if mode == "two-stage":
            args.append("--two-stage")
        elif mode == "two-stages-hq":
            args.append("--two-stages-hq")
        elif mode == "one-stage":
            args.append("--one-stage")
        else:
            args.append("--distilled")

        if bool((mlx or {}).get("low_ram", True)):
            args.append("--low-ram")

        cfg = params.get("cfg_scale")
        try:
            if cfg not in (None, "") and float(cfg) > 0:
                args += ["--cfg-scale", str(float(cfg))]
        except (TypeError, ValueError):
            pass

        stage1_steps = (mlx or {}).get("stage1_steps")
        if stage1_steps:
            args += ["--stage1-steps", str(int(stage1_steps))]
        stage2_steps = (mlx or {}).get("stage2_steps")
        if stage2_steps:
            args += ["--stage2-steps", str(int(stage2_steps))]

        extra = (mlx or {}).get("extra_args")
        if isinstance(extra, (list, tuple)):
            args += [str(token) for token in extra]
        return args

    def generate(
        self,
        model_dir: Path,
        params: dict[str, Any],
        seed: int,
        *,
        mlx: dict[str, Any],
    ) -> Path:
        output = self._output_path(seed)
        args = self.build_command(model_dir, params, seed, mlx, output)
        logger.info("MLX video generate: %s", " ".join(args))
        try:
            result = subprocess.run(args, capture_output=True, text=True)  # noqa: S603
        except FileNotFoundError as exc:  # pragma: no cover - resolved above
            raise RuntimeError(f"Lancement ltx-2-mlx impossible : {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            tail = "\n".join(detail.splitlines()[-8:]) or "code de sortie non nul"
            raise RuntimeError(f"Échec de la génération ltx-2-mlx :\n{tail}")
        if not output.exists():
            raise RuntimeError("ltx-2-mlx s'est terminé sans produire de vidéo.")
        return output

    def _output_path(self, seed: int) -> Path:
        self.paths.output_videos_dir.mkdir(parents=True, exist_ok=True)
        return self.paths.output_videos_dir / f"video_{int(time.time())}_{seed}.mp4"
