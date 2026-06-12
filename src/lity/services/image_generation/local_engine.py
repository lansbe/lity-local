"""In-process text-to-image generation from a downloaded checkpoint.

This is the local, server-less alternative to the AUTOMATIC1111 HTTP backend:
instead of launching an external WebUI and talking to ``/sdapi``, Lity loads the
``.safetensors`` checkpoint the user already downloaded (under
``…/Documents/Lity/Models/Images/<name>/``) straight into the process with
Hugging Face ``diffusers`` and renders the image on the local GPU (Metal on
Apple Silicon, CUDA on NVIDIA, CPU otherwise).

``torch``/``diffusers`` are heavy, optional dependencies, so every import is
lazy and guarded: the rest of the app keeps working — and the image tab can
offer a one-click install — when they are absent.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lity.infrastructure.paths import AppPaths

logger = logging.getLogger(__name__)

# Per-step progress: (step_done, step_total).
StepCallback = Callable[[int, int], None]

# AUTOMATIC1111 sampler label (what the prompt builder proposes) → the diffusers
# scheduler class + the flags that reproduce that sampler. Anything unknown
# falls back to the checkpoint's native scheduler.
_SCHEDULERS: dict[str, tuple[str, dict[str, Any]]] = {
    "euler a": ("EulerAncestralDiscreteScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    # Few-step distilled checkpoints (SDXL-Lightning) are trained against
    # trailing timestep spacing; without it their 2-8 step renders degrade.
    "euler trailing": ("EulerDiscreteScheduler", {"timestep_spacing": "trailing"}),
    "heun": ("HeunDiscreteScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "lms karras": ("LMSDiscreteScheduler", {"use_karras_sigmas": True}),
    "ddim": ("DDIMScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "dpm++ 2m": ("DPMSolverMultistepScheduler", {}),
    "dpm++ 2m karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpm++ 2m sde": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    "dpm++ 2m sde karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpm++ sde": ("DPMSolverSDEScheduler", {}),
    "dpm++ sde karras": ("DPMSolverSDEScheduler", {"use_karras_sigmas": True}),
}

# Packages the local engine needs, in install order (torch first so the rest
# resolve against the already-present build). transformers is capped below 5.0:
# diffusers 0.38's single-file CLIP loader reads ``model.text_model`` internals
# that transformers 5.x removed, which crashes checkpoint loading.
ENGINE_PACKAGES: tuple[str, ...] = (
    "torch",
    "safetensors",
    "transformers>=4.40,<5",
    "accelerate",
    "diffusers>=0.27",
    "Pillow",
)


def dependencies_available() -> bool:
    """True when ``torch`` and ``diffusers`` can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers"))


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

    Half precision matters doubly on Apple unified memory: it halves the
    pipeline's footprint (SD 1.5 ~9 GB → ~4 GB), which is the difference
    between generating smoothly and swapping the whole Mac on 16 GB machines.
    fp16 renders correctly on MPS as long as attention slicing stays off (see
    ``_load_pipeline``); the black-frame guard catches any residual failure
    and retries that render in fp32.

    ``LITY_IMAGE_DTYPE=float16|bfloat16|float32`` overrides for troubleshooting.
    """
    import torch

    override = os.environ.get("LITY_IMAGE_DTYPE", "").strip().lower()
    if override in ("float16", "fp16", "half"):
        return torch.float16
    if override in ("bfloat16", "bf16"):
        return torch.bfloat16
    if override in ("float32", "fp32", "full"):
        return torch.float32
    return torch.float32 if device == "cpu" else torch.float16


def _prefer_tiny_vae() -> bool:
    """Swap the full VAE decoder for TAESD on memory-constrained machines.

    TAESD (~10 MB) decodes latents visually indistinguishably from the full
    VAE while cutting ~1 GB off the render peak AND ~30% off the render time
    (measured on M5: 2.78 s → 1.98 s per sd-turbo image). The full VAE stays
    the default on big machines; ``LITY_IMAGE_VAE=full|tiny`` forces either.
    """
    override = os.environ.get("LITY_IMAGE_VAE", "").strip().lower()
    if override == "full":
        return False
    if override == "tiny":
        return True
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3 < 24.0
    except Exception:  # pragma: no cover - psutil unavailable
        return False


def _is_sdxl_checkpoint(path: Path) -> bool:
    """Tell an SDXL checkpoint from an SD 1.x/2.x one without loading weights.

    SDXL carries a *second* text encoder, so its state dict has
    ``conditioner.embedders.1`` keys (and, once converted, ``add_embedding``).
    We only read the safetensors header (cheap); ``.ckpt`` pickles fall back to
    the filename hint.
    """
    name = path.name.lower()
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safetensors handle, not a dict
                    if "conditioner.embedders.1" in key or "add_embedding" in key:
                        return True
            return False
        except Exception as exc:  # pragma: no cover - corrupt/odd file
            logger.info("SDXL detection fell back to filename for %s: %s", path.name, exc)
    return "xl" in name


def _snap8(value: int) -> int:
    """Diffusion UNets need spatial sizes that are multiples of 8."""
    return max(256, int(round(value / 8)) * 8)


class LocalImageEngine:
    """Loads a checkpoint once and renders images from it in-process."""

    def __init__(self, paths: AppPaths):
        self.paths = paths
        self._pipe: Any = None
        self._pipe_key: str | None = None
        self._is_sdxl = False
        # Set after an all-black fp16 render (historical MPS quirk): reload the
        # pipeline in fp32 and retry once.
        self._force_fp32 = False

    # ---------------------------------------------------------------- loading
    def _load_pipeline(self, checkpoint: Path, config_hint: str = "") -> Any:
        import torch

        device = torch_device()
        dtype = torch.float32 if self._force_fp32 else _dtype_for(device)
        key = f"{checkpoint.resolve()}|{dtype}|{config_hint}"
        if self._pipe is not None and self._pipe_key == key:
            return self._pipe

        # Switching models: release the old pipeline before loading the new one
        # so two checkpoints never sit in (V)RAM at once.
        self._release_pipeline()

        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        self._is_sdxl = _is_sdxl_checkpoint(checkpoint)
        pipeline_cls = StableDiffusionXLPipeline if self._is_sdxl else StableDiffusionPipeline

        logger.info(
            "Loading %s checkpoint %s on %s (%s)",
            "SDXL" if self._is_sdxl else "SD",
            checkpoint.name,
            device,
            dtype,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": checkpoint.suffix.lower() == ".safetensors",
        }
        # Some checkpoints (SD2.x arch) need an explicit diffusers config repo
        # because the default inference points at a gated/removed repo.
        if config_hint:
            load_kwargs["config"] = config_hint
        pipe = pipeline_cls.from_single_file(str(checkpoint), **load_kwargs)
        # The NSFW checker blanks flagged images and pulls an extra model; a
        # local-first tool the user controls doesn't want either.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to(device)
        # Attention slicing only in fp32: it halves activation memory there,
        # but combined with fp16/bf16 on MPS (torch 2.12) it corrupts the
        # attention output into all-NaN (black) frames — measured on M5.
        # Half precision already halves activations, so skipping it is fine.
        if dtype == torch.float32:
            with _suppress():
                pipe.enable_attention_slicing()
        if _prefer_tiny_vae():
            # First use downloads ~10 MB from the Hub (cached after); offline
            # or any failure silently keeps the full VAE.
            with _suppress():
                from diffusers import AutoencoderTiny

                repo = "madebyollin/taesdxl" if self._is_sdxl else "madebyollin/taesd"
                pipe.vae = AutoencoderTiny.from_pretrained(repo, torch_dtype=dtype).to(device)
        if self._is_sdxl:
            with _suppress():
                pipe.enable_vae_tiling()

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
        """Free the loaded checkpoint (called when image mode is turned off)."""
        self._release_pipeline()

    def _apply_scheduler(self, pipe: Any, sampler: str) -> None:
        mapping = _SCHEDULERS.get((sampler or "").strip().lower())
        if mapping is None:
            return
        class_name, kwargs = mapping
        try:
            import diffusers

            scheduler_cls = getattr(diffusers, class_name)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
        except Exception as exc:  # pragma: no cover - keep the native scheduler
            logger.info("Sampler %s unavailable (%s); keeping default scheduler", sampler, exc)

    # ------------------------------------------------------------- generation
    def generate(
        self,
        checkpoint: Path,
        params: dict[str, Any],
        seed: int,
        on_step: StepCallback | None = None,
        config_hint: str = "",
    ) -> Path:
        """Render one image and save it as PNG. Returns the saved path."""
        import torch

        pipe = self._load_pipeline(checkpoint, config_hint)
        self._apply_scheduler(pipe, str(params.get("sampler", "")))

        steps = int(params.get("steps", 25))
        width = _snap8(int(params.get("width", 512)))
        height = _snap8(int(params.get("height", 512)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        callback_kwargs: dict[str, Any] = {}
        if on_step is not None:

            def _step_cb(_pipe: Any, step_index: int, _timestep: Any, kwargs: dict[str, Any]):
                with contextlib.suppress(Exception):  # progress is best-effort
                    on_step(step_index + 1, steps)
                return kwargs

            callback_kwargs["callback_on_step_end"] = _step_cb

        result = pipe(
            prompt=str(params.get("prompt", "")),
            negative_prompt=str(params.get("negative_prompt", "")),
            num_inference_steps=steps,
            guidance_scale=float(params.get("cfg_scale", 7.5)),
            width=width,
            height=height,
            generator=generator,
            **callback_kwargs,
        )
        image = result.images[0]
        if _is_black_frame(image) and not self._force_fp32:
            # Historical MPS fp16 failure mode: NaNs decode to a black frame.
            # Reload this checkpoint in fp32 and retry the same render once.
            logger.warning("fp16 render came back black; retrying %s in fp32", checkpoint.name)
            self._force_fp32 = True
            return self.generate(checkpoint, params, seed, on_step, config_hint)
        path = self._save(image, seed)
        # Release the allocator's cached blocks right away: on unified-memory
        # Macs the cache from a 1024px SDXL render otherwise stacks onto the
        # next render's working set and pushes the whole machine into swap.
        with _suppress():
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return path

    def _save(self, image: Any, seed: int) -> Path:
        self.paths.output_images_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.output_images_dir / f"image_{int(time.time())}_{seed}.png"
        image.save(path)
        return path


def _is_black_frame(image: Any) -> bool:
    """True when every pixel is (near) zero — the fp16-NaN failure signature."""
    try:
        extrema = image.getextrema()  # per-band (min, max) tuples
        if isinstance(extrema[0], (int, float)):
            extrema = (extrema,)
        return all(band[1] <= 1 for band in extrema)
    except Exception:  # pragma: no cover - never block saving on the check
        return False


class _suppress:
    """Swallow optional-feature errors (a pipeline may lack a given saver)."""

    def __enter__(self) -> _suppress:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return True
