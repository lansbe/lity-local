from __future__ import annotations

from typing import Any

from lity.core.compatibility import (
    GRADES,
    evaluate_model_complete,
    profile_from_hardware,
)
from lity.core.model_advisor import _verdict

IMAGE_MODEL_CATALOG: tuple[dict[str, Any], ...] = (
    # Optional fields consumed by the in-process engine and downloader:
    #   model_file — exact file to download from the repo (skips heuristics).
    #   gen        — generation defaults this checkpoint was distilled for
    #                (few-step models break with classic steps=25/cfg=7.5).
    {
        "name": "sd-turbo",
        "display_name": "SD Turbo (1-4 étapes)",
        "provider": "Stability AI",
        "params_b": 0.9,
        "vram_gb": 4.0,  # mesuré : 4,2 Go (fp16, M5)
        "disk_gb": 5.2,
        "backend": "automatic1111",
        "license": "Stability AI Community (non commercial)",
        "model_url": "https://huggingface.co/stabilityai/sd-turbo",
        "model_file": "sd_turbo.safetensors",
        # SD2.x-arch single files need an explicit config repo: diffusers'
        # default inference points at stabilityai/stable-diffusion-2-1, which
        # Stability gated (401) — the sd-turbo repo itself stayed public.
        "hf_config": "stabilityai/sd-turbo",
        "gen": {"steps": 4, "cfg_scale": 0.0, "sampler": "Euler a", "width": 512, "height": 512},
        "install_hint": "Distillé pour 1-4 étapes : quasi instantané, idéal 8-16 Go.",
    },
    {
        "name": "sdxl-lightning",
        "display_name": "SDXL Lightning (4 étapes)",
        "provider": "ByteDance",
        "params_b": 3.5,
        "vram_gb": 8.5,  # mesuré : 8,6 Go à 768px (fp16, M5) ; 18,7 Go à 1024px
        "disk_gb": 6.9,
        "backend": "automatic1111",
        "license": "OpenRAIL++ (SDXL)",
        "model_url": "https://huggingface.co/ByteDance/SDXL-Lightning",
        "model_file": "sdxl_lightning_4step.safetensors",
        # 768 par défaut : le 1024 natif alloue ~19 Go et fait swapper les
        # machines 16 Go ; à 768 le rendu reste très net (demande "1024" dans
        # le dialogue de paramètres sur une machine plus grosse).
        "gen": {
            "steps": 4,
            "cfg_scale": 0.0,
            "sampler": "Euler trailing",
            "width": 768,
            "height": 768,
        },
        "install_hint": "Qualité SDXL en 4 étapes (768px par défaut) — le meilleur ratio qualité/vitesse local.",
    },
    {
        "name": "sd15",
        "display_name": "Stable Diffusion 1.5",
        "provider": "Runway / Stability AI",
        "params_b": 0.9,
        "vram_gb": 4.0,
        "disk_gb": 4.3,
        "backend": "automatic1111",
        "license": "CreativeML Open RAIL-M",
        "model_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5",
        "install_hint": "Checkpoint .safetensors/.ckpt a placer dans models/Stable-diffusion.",
    },
    {
        "name": "realistic-vision-v6",
        "display_name": "Realistic Vision V6",
        "provider": "Community",
        "params_b": 0.9,
        "vram_gb": 4.5,
        "disk_gb": 4.3,
        "backend": "automatic1111",
        "license": "Community checkpoint",
        "model_url": "https://civitai.com/models/4201/realistic-vision-v60-b1",
        "install_hint": "Checkpoint SD 1.5 compatible AUTOMATIC1111.",
    },
    {
        "name": "dreamshaper-xl",
        "display_name": "DreamShaper XL",
        "provider": "Lykon / Community",
        "params_b": 3.5,
        "vram_gb": 7.5,
        "disk_gb": 6.9,
        "backend": "automatic1111",
        "license": "Community checkpoint",
        "model_url": "https://civitai.com/models/112902/dreamshaper-xl",
        "install_hint": "Checkpoint SDXL compatible AUTOMATIC1111.",
    },
    {
        "name": "sdxl-base",
        "display_name": "Stable Diffusion XL Base 1.0",
        "provider": "Stability AI",
        "params_b": 3.5,
        "vram_gb": 8.0,
        "disk_gb": 6.9,
        "backend": "automatic1111",
        "license": "CreativeML Open RAIL++",
        "model_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "install_hint": "Checkpoint SDXL a placer dans models/Stable-diffusion.",
    },
    # --- MLX / mflux one-click models (Apple Silicon, rendered out-of-process
    # by mlx_engine). The ``mlx`` block drives the CLI invocation; ``model_arg:
    # "model"`` means Lity passes the downloaded model directory as ``--model``.
    # ``backend: "mlx"`` keeps the
    # diffusers engine from ever trying to load these weights.
    {
        "name": "z-image-turbo",
        "display_name": "Z-Image Turbo 6B (MLX 4-bit)",
        "provider": "Tongyi-MAI",
        "params_b": 6.0,
        "vram_gb": 10.0,  # pic run ~9-10 Go en 4-bit (poids+activations+VAE+encodeur)
        "disk_gb": 6.0,
        "backend": "mlx",
        "license": "Apache 2.0",
        "model_url": "https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit",
        "mlx": {
            "command": "mflux-generate-z-image-turbo",
            "model_arg": "model",
            "base_model": "z-image-turbo",
            "quantize": 4,
        },
        "gen": {"steps": 9, "cfg_scale": 0.0, "sampler": "", "width": 768, "height": 768},
        "install_hint": "MLX/mflux 4-bit — #1 open-weights, ~8 étapes, 768px. La meilleure qualité tenable sur 16 Go.",
    },
    {
        "name": "flux2-klein-4b",
        "display_name": "FLUX.2 klein 4B (MLX 4-bit)",
        "provider": "Black Forest Labs",
        "params_b": 4.0,
        "vram_gb": 9.0,  # pic run ~8-9 Go en 4-bit (mflux)
        "disk_gb": 4.3,
        "backend": "mlx",
        "license": "Apache 2.0",
        "model_url": "https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit",
        "mlx": {
            "command": "mflux-generate-flux2",
            "model_arg": "model",
            "base_model": "flux2-klein-4b",
            "quantize": 4,
        },
        "gen": {"steps": 4, "cfg_scale": 0.0, "sampler": "", "width": 768, "height": 768},
        "install_hint": "MLX/mflux 4-bit — édition rapide en 4 étapes, ~9 Go en pic. Idéal M-series 16 Go.",
    },
    {
        "name": "flux1-schnell-mlx",
        "display_name": "FLUX.1 schnell (MLX 4-bit)",
        "provider": "Black Forest Labs",
        "params_b": 12.0,
        "vram_gb": 12.0,  # 512px tient en 4-bit ; le 1024px ne passe pas sur 16 Go
        "disk_gb": 6.7,
        "backend": "mlx",
        "license": "Apache 2.0",
        "model_url": "https://huggingface.co/dhairyashil/FLUX.1-schnell-mflux-4bit",
        "mlx": {
            "command": "mflux-generate",
            "model_arg": "model",
            "base_model": "schnell",
            "quantize": 4,
        },
        "gen": {"steps": 4, "cfg_scale": 0.0, "sampler": "", "width": 512, "height": 512},
        "install_hint": "MLX/mflux 4-bit — 2-4 étapes, 512px sur 16 Go (le 1024 ne passe pas).",
    },
    {
        "name": "qwen-image",
        "display_name": "Qwen-Image",
        "provider": "Alibaba",
        "params_b": 20.0,
        "vram_gb": 24.0,
        "disk_gb": 40.0,
        "backend": "comfyui",
        "license": "Apache 2.0",
        "model_url": "https://huggingface.co/Qwen/Qwen-Image",
        "install_hint": "Excellent texte/edition; demande ComfyUI, Diffusers ou stable-diffusion.cpp.",
    },
    {
        "name": "flux2-dev",
        "display_name": "FLUX.2 dev",
        "provider": "Black Forest Labs",
        "params_b": 32.0,
        "vram_gb": 48.0,
        "disk_gb": 64.0,
        "backend": "comfyui",
        "license": "FLUX Non-Commercial",
        "model_url": "https://huggingface.co/black-forest-labs/FLUX.2-dev",
        "install_hint": "Qualite maximale locale, licence non-commerciale et GPU tres costaud.",
    },
)

# MLX renders in-app (out-of-process) like the diffusers engine, so it ranks
# alongside automatic1111 ahead of the external-runtime backends.
_BACKEND_PRIORITY = {"automatic1111": 0, "mlx": 0, "comfyui": 1, "stable-diffusion.cpp": 1}


def _apple_silicon() -> bool:
    """MLX/mflux only run on arm64 macOS — recommend them only there."""
    import platform
    import sys

    return sys.platform == "darwin" and platform.machine() == "arm64"


def rank_image_models(
    hardware: dict[str, Any], installed: list[str] | None = None
) -> list[dict[str, Any]]:
    """Rank local image-generation models for this device.

    Unlike Ollama LLMs, image checkpoints target several runtimes. Rows carry the
    backend so the UI can explain whether Lity can use the model through the
    current AUTOMATIC1111 integration or whether another local runtime is needed.
    """
    profile = profile_from_hardware(hardware)
    installed_names = {name.lower() for name in installed or [] if name}
    rows: list[dict[str, Any]] = []
    for model in IMAGE_MODEL_CATALOG:
        evaluation = evaluate_model_complete(
            float(model["vram_gb"]), profile, float(model["params_b"])
        )
        row = {
            **model,
            "kind": "image",
            "installed": _is_installed(model, installed_names),
            "verdict": _verdict(evaluation),
            "speed": _image_speed_label(evaluation),
            "grade": evaluation["grade"],
            "grade_label": GRADES[evaluation["grade"]]["label"],
            "score": evaluation["score"],
            "status": evaluation["status"],
            "tokens_per_sec": evaluation["tokens_per_sec"],
            "mem_pct": evaluation["mem_pct"],
        }
        rows.append(row)

    rows.sort(key=_sort_key)
    _flag_recommended(rows)
    return rows


def _is_installed(model: dict[str, Any], installed_names: set[str]) -> bool:
    if not installed_names:
        return False
    needles = {
        str(model["name"]).lower(),
        str(model["display_name"]).lower(),
    }
    return any(any(needle in installed for needle in needles) for installed in installed_names)


def _image_speed_label(evaluation: dict[str, Any]) -> str:
    status = evaluation["status"]
    if status == "can-run":
        return "pret localement"
    if status == "tight":
        return "serre mais possible"
    if status == "can-run-slow":
        return "lent avec offload RAM"
    if status == "unknown":
        return "materiel inconnu"
    return "trop lourd"


def _sort_key(row: dict[str, Any]) -> tuple[int, int, float, float]:
    status_tier = {
        "can-run": 0,
        "tight": 1,
        "can-run-slow": 2,
        "unknown": 3,
        "cannot-run": 4,
    }.get(str(row["status"]), 4)
    backend_tier = _BACKEND_PRIORITY.get(str(row["backend"]), 2)
    if status_tier == 4:
        return (status_tier, backend_tier, 0.0, float(row["vram_gb"]))
    return (status_tier, backend_tier, -float(row["score"]), -float(row["params_b"]))


def _flag_recommended(rows: list[dict[str, Any]]) -> None:
    # In-app backends Lity can actually run: diffusers everywhere, MLX on Apple
    # Silicon (where it's the best local path on this hardware).
    native_backends = {"automatic1111", "mlx"} if _apple_silicon() else {"automatic1111"}
    app_native = [
        row
        for row in rows
        if row["backend"] in native_backends and row["status"] in ("can-run", "tight")
    ]
    pool = app_native or [row for row in rows if row["status"] in ("can-run", "tight")]
    if not pool:
        pool = [row for row in rows if row["status"] == "can-run-slow"]
    if pool:
        best = max(pool, key=lambda row: (float(row["score"]), float(row["params_b"])))
        best["recommended"] = True
