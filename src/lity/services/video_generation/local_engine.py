"""In-process text-to-video generation from a downloaded model.

The local, server-less counterpart to the image engine. Instead of a single
``.safetensors``, a video model is a multi-file diffusers repository the user
downloaded under ``…/Documents/Lity/Models/Videos/<name>/``. We load it with
``diffusers.DiffusionPipeline.from_pretrained`` (which auto-selects ``WanPipeline``
from ``model_index.json``), render a short clip on the local GPU (Metal on Apple
Silicon, CUDA on NVIDIA, CPU otherwise), and export it to MP4 via imageio/ffmpeg.

``torch``/``diffusers``/``imageio`` are heavy, optional dependencies, so every
import is lazy and guarded: the rest of the app keeps working — and the video
tab can offer a one-click install — when they are absent.

The ``mlx`` backend (LTX-2 / mlx-video) is catalogued and downloadable but its
dedicated runtime is not embedded yet; generating with it raises a clear error
(the manager already guards against reaching here).
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any

from lity.infrastructure.paths import AppPaths

logger = logging.getLogger(__name__)

# Packages the local video engine needs, in install order (torch first so the
# rest resolve against the already-present build). torch/transformers/accelerate
# are shared with the image engine; diffusers is bumped to the version exposing
# WanPipeline, and imageio[-ffmpeg] writes the MP4. transformers stays <5 to
# match the image engine's single-file loader constraint (same venv).
ENGINE_PACKAGES: tuple[str, ...] = (
    "torch",
    "safetensors",
    "transformers>=4.40,<5",
    "accelerate",
    "diffusers>=0.32",
    "imageio",
    "imageio-ffmpeg",
    "ftfy",
    "Pillow",
)


def dependencies_available() -> bool:
    """True when ``torch``, ``diffusers`` and ``imageio`` can be imported."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "imageio")
    )


def torch_device() -> str:
    """Best local accelerator for generation: ``cuda`` → ``mps`` → ``cpu``."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing/broken
        pass
    return "cpu"


def _dtype_for(device: str) -> Any:
    """float16 on GPUs (CUDA *and* MPS), float32 on CPU.

    Half precision is essential on 16 GB unified memory: it roughly halves the
    pipeline footprint, the difference between a slow render and swapping the
    whole Mac. ``LITY_VIDEO_DTYPE=float16|bfloat16|float32`` overrides.
    """
    import torch

    override = os.environ.get("LITY_VIDEO_DTYPE", "").strip().lower()
    if override in ("float16", "fp16", "half"):
        return torch.float16
    if override in ("bfloat16", "bf16"):
        return torch.bfloat16
    if override in ("float32", "fp32", "full"):
        return torch.float32
    return torch.float32 if device == "cpu" else torch.float16


def _snap16(value: int) -> int:
    """Video VAEs need spatial sizes that are multiples of 16."""
    return max(256, int(round(value / 16)) * 16)


def _snap_frames(value: int) -> int:
    """Wan-family temporal VAE expects num_frames of the form 4·k + 1."""
    frames = max(5, int(value))
    return ((frames - 1) // 4) * 4 + 1


class LocalVideoEngine:
    """Loads a downloaded video model once and renders clips from it in-process."""

    def __init__(self, paths: AppPaths):
        self.paths = paths
        self._pipe: Any = None
        self._pipe_key: str | None = None

    # ---------------------------------------------------------------- loading
    def _load_pipeline(self, model_dir: Path, dtype: Any, device: str) -> Any:
        key = f"{model_dir.resolve()}|{dtype}"
        if self._pipe is not None and self._pipe_key == key:
            return self._pipe

        # Switching models: release the old pipeline before loading the new one
        # so two models never sit in (V)RAM at once.
        self._release_pipeline()

        from diffusers import DiffusionPipeline

        logger.info("Loading video model %s on %s (%s)", model_dir.name, device, dtype)
        pipe = DiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to(device)
        # Memory savers — best-effort; a pipeline may lack any of these. VAE
        # tiling and attention slicing are what keep a 480p clip inside 16 GB.
        for enable in ("enable_vae_tiling", "enable_vae_slicing", "enable_attention_slicing"):
            with _suppress():
                getattr(pipe, enable, lambda: None)()

        self._pipe = pipe
        self._pipe_key = key
        return pipe

    def _release_pipeline(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        self._pipe_key = None
        gc.collect()
        with _suppress():
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload(self) -> None:
        """Free the loaded model (called when video mode is turned off)."""
        self._release_pipeline()

    # ------------------------------------------------------------- generation
    def generate(
        self,
        model_dir: Path,
        params: dict[str, Any],
        seed: int,
        backend: str = "diffusers",
        config_hint: str = "",
    ) -> Path:
        """Render one short clip and save it as MP4. Returns the saved path."""
        if backend == "mlx":
            raise RuntimeError(
                "Ce modèle doit être généré par le moteur MLX dédié, pas par le moteur diffusers."
            )

        import torch
        from diffusers.utils import export_to_video

        device = torch_device()
        dtype = _dtype_for(device)
        pipe = self._load_pipeline(model_dir, dtype, device)

        steps = int(params.get("steps", 25))
        num_frames = _snap_frames(int(params.get("num_frames", 49)))
        fps = int(params.get("fps", 15))
        width = _snap16(int(params.get("width", 832)))
        height = _snap16(int(params.get("height", 480)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        result = pipe(
            prompt=str(params.get("prompt", "")),
            negative_prompt=str(params.get("negative_prompt", "")),
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=float(params.get("cfg_scale", 5.0)),
            width=width,
            height=height,
            generator=generator,
        )
        frames = result.frames[0]
        path = self._save(frames, fps, seed, export_to_video)
        # Release cached allocator blocks right away: on unified-memory Macs the
        # cache from a render otherwise stacks onto the next one and swaps.
        with _suppress():
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return path

    def _save(self, frames: Any, fps: int, seed: int, export_to_video: Any) -> Path:
        self.paths.output_videos_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.output_videos_dir / f"video_{int(time.time())}_{seed}.mp4"
        exported = export_to_video(frames, output_video_path=str(path), fps=fps)
        return Path(exported or path)


class _suppress:
    """Swallow optional-feature errors (a pipeline may lack a given saver)."""

    def __enter__(self) -> _suppress:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return True
