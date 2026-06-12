"""Local MLX (mflux) image generation, isolated in a subprocess.

The in-process diffusers engine (:mod:`local_engine`) cannot load MLX-format
weights, and running mflux inside the same interpreter as an already-imported
``torch``/``diffusers`` risks ``mlx``/``numpy`` ABI conflicts. So MLX models
render by shelling out to the ``mflux`` CLI: one clean process per render that
loads the pre-quantized weights the user downloaded and writes a PNG.

``mflux`` is a heavy, Apple-Silicon-only dependency, installed on first use like
torch/diffusers. Every import/probe is guarded so the rest of the app keeps
working when it is absent.

The exact CLI command + flags differ per model family (FLUX.1 vs FLUX.2 vs
Z-Image) and evolve across mflux releases, so the invocation is **catalog-data
driven**: each model's ``mlx`` block (persisted into the download marker)
provides ``command`` (the executable, e.g. ``mflux-generate``), ``model_arg``
(``model`` for current mflux CLIs, or ``path`` for older saved-model CLIs), an
optional ``model`` literal, ``base_model``, ``quantize``, and optional
``extra_args``. Tweaking a model's invocation is a one-line catalog change, not
an engine change.
"""

from __future__ import annotations

import importlib.util
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

# pip distribution(s) the MLX engine needs. ``mflux`` pulls ``mlx`` itself.
# FLUX.2/Z-Image Turbo require the newer model-specific CLI entry points.
MLX_ENGINE_PACKAGES: tuple[str, ...] = ("mflux>=0.18",)

# Default few-step CLI; individual models override via their ``mlx.command``.
_DEFAULT_COMMAND = "mflux-generate"


def mlx_supported_platform() -> bool:
    """mflux/MLX run only on Apple Silicon (arm64 macOS)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_dependencies_available(command: str = _DEFAULT_COMMAND) -> bool:
    """True when the mflux CLI command this model needs is resolvable."""
    if resolve_mflux_cli(command or _DEFAULT_COMMAND) is not None:
        return True
    # Importability alone is not enough to render (we shell out), but it keeps
    # health checks informative for unusual installs where entry points are
    # generated lazily or exposed by a launcher wrapper.
    return bool(command == _DEFAULT_COMMAND and importlib.util.find_spec("mflux") is not None)


def resolve_mflux_cli(command: str) -> str | None:
    """Locate an mflux console script (PATH first, then next to the interpreter).

    uv/venv installs the ``mflux-*`` entry points beside ``sys.executable``,
    which isn't always on the GUI app's PATH — check there too.
    """
    found = shutil.which(command)
    if found:
        return found
    candidate = Path(sys.executable).parent / command
    return str(candidate) if candidate.exists() else None


class MlxImageEngine:
    """Render images from MLX (mflux) models via a per-render CLI subprocess."""

    def __init__(self, paths: AppPaths):
        self.paths = paths

    def unload(self) -> None:
        """No resident state — each render is its own short-lived process."""
        return None

    def build_command(
        self, model_dir: Path, params: dict[str, Any], seed: int, mlx: dict[str, Any], output: Path
    ) -> list[str]:
        """Assemble the mflux CLI argv for one render (pure, for testability)."""
        command = str((mlx or {}).get("command") or _DEFAULT_COMMAND)
        cli = resolve_mflux_cli(command)
        if cli is None:
            raise RuntimeError(
                f"Commande mflux introuvable : « {command} ». Le moteur MLX est-il installé ?"
            )
        args = [
            cli,
            "--prompt",
            str(params.get("prompt", "")),
            "--seed",
            str(int(seed)),
            "--steps",
            str(int(params.get("steps", 4))),
            "--height",
            str(int(params.get("height", 768))),
            "--width",
            str(int(params.get("width", 768))),
            "--output",
            str(output),
        ]
        # Load the weights the user already downloaded (no re-fetch). Current
        # mflux generation CLIs use ``--model <local-dir>``; older saved-model
        # flows used ``--path <local-dir>``. Keep it catalog-driven so model
        # entries can track mflux changes without editing this engine.
        model_source_arg = str(
            (mlx or {}).get("model_arg") or (mlx or {}).get("path_arg") or "path"
        ).strip()
        if model_source_arg:
            args += [f"--{model_source_arg.replace('_', '-')}", str(model_dir)]
        model_arg = str((mlx or {}).get("model") or "").strip()
        if model_arg:
            args += ["--model", model_arg]
        base_model = str((mlx or {}).get("base_model") or "").strip()
        if base_model:
            args += ["--base-model", base_model]
        quantize = (mlx or {}).get("quantize")
        if quantize:
            args += ["--quantize", str(int(quantize))]
        # Distilled few-step checkpoints (turbo/schnell) are guidance-free; only
        # pass guidance when the params carry a meaningful (>0) value.
        guidance = params.get("cfg_scale")
        try:
            if guidance not in (None, "") and float(guidance) > 0:
                args += ["--guidance", str(float(guidance))]
        except (TypeError, ValueError):
            pass
        extra = (mlx or {}).get("extra_args")
        if isinstance(extra, (list, tuple)):
            args += [str(token) for token in extra]
        return args

    def generate(
        self, model_dir: Path, params: dict[str, Any], seed: int, *, mlx: dict[str, Any]
    ) -> Path:
        """Render one image with mflux and return the saved PNG path."""
        output = self._output_path(seed)
        args = self.build_command(model_dir, params, seed, mlx, output)
        logger.info("MLX generate: %s", " ".join(args))
        try:
            result = subprocess.run(args, capture_output=True, text=True)  # noqa: S603
        except FileNotFoundError as exc:  # pragma: no cover - resolved above
            raise RuntimeError(f"Lancement mflux impossible : {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            tail = "\n".join(detail.splitlines()[-8:]) or "code de sortie non nul"
            raise RuntimeError(f"Échec de la génération mflux :\n{tail}")
        if not output.exists():
            raise RuntimeError("mflux s'est terminé sans produire d'image.")
        return output

    def _output_path(self, seed: int) -> Path:
        self.paths.output_images_dir.mkdir(parents=True, exist_ok=True)
        return self.paths.output_images_dir / f"image_{int(time.time())}_{seed}.png"
