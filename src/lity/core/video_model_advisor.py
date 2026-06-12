from __future__ import annotations

from typing import Any

from lity.core.compatibility import (
    GRADES,
    evaluate_model_complete,
    profile_from_hardware,
)
from lity.core.model_advisor import _verdict

# Local text/image-to-video models, mirroring IMAGE_MODEL_CATALOG. Extra fields
# beyond the image catalog:
#   input_type — "text" (T2V) or "image" (I2V); shown in the UI.
#   hf_config  — the diffusers repo id (also the from_pretrained source); only
#                meaningful for backend == "diffusers".
# The ``backend`` field decides the runtime:
#   diffusers — in-process WanPipeline on torch/MPS, the integrated engine
#               (tier 0, like automatic1111 is for images).
#   mlx       — external Apple-Silicon MLX runtime (ltx-2-mlx), installed and
#               invoked by Lity in one click.
#
# vram_gb/disk_gb are conservative provisional estimates for an M5 / 16 GB at
# fp16; they gate the can-run/tight/cannot-run verdict and must be re-measured
# before a model is promoted from "tight" to "can-run".
VIDEO_MODEL_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "name": "wan21-t2v-1.3b",
        "display_name": "Wan 2.1 T2V 1.3B",
        "provider": "Alibaba / Wan-AI",
        "params_b": 1.3,
        "vram_gb": 9.0,  # provisoire : fp16 sur MPS, à mesurer sur M5
        "disk_gb": 17.0,  # repo diffusers multi-fichiers (DiT + VAE + text encoder)
        "backend": "diffusers",
        "input_type": "text",
        "license": "Apache 2.0",
        "model_url": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "hf_config": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        # Few-step generation defaults the manager applies before rendering.
        "gen": {
            "num_frames": 49,
            "fps": 15,
            "steps": 25,
            "cfg_scale": 5.0,
            "width": 832,
            "height": 480,
            "sampler": "UniPC",
        },
        "install_hint": "Texte→vidéo le plus léger, viable sur 16 Go. Clips courts (~3 s, 480p).",
    },
    {
        "name": "wan22-ti2v-5b",
        "display_name": "Wan 2.2 TI2V-5B",
        "provider": "Alibaba / Wan-AI",
        "params_b": 5.0,
        "vram_gb": 16.0,  # limite haute 16 Go — offload requis, à valider
        "disk_gb": 28.0,
        "backend": "diffusers",
        "input_type": "text",
        "license": "Apache 2.0",
        "model_url": "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "hf_config": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "gen": {
            "num_frames": 49,
            "fps": 24,
            "steps": 30,
            "cfg_scale": 5.0,
            "width": 704,
            "height": 1280,
            "sampler": "UniPC",
        },
        "install_hint": "Meilleure qualité (texte + image→vidéo), serré à 16 Go (offload mémoire).",
    },
    {
        "name": "ltx2-int4-mlx",
        "display_name": "LTX-2 (int4, MLX)",
        "provider": "Lightricks / dgrauet",
        "params_b": 2.0,
        "vram_gb": 12.0,
        "disk_gb": 12.0,
        "backend": "mlx",  # runtime MLX dédié — non joué par le moteur diffusers
        "input_type": "text",
        "license": "LTX Open",
        "model_url": "https://huggingface.co/dgrauet/ltx-2.3-mlx-q4",
        "code_url": "https://github.com/dgrauet/ltx-2-mlx",
        "mlx": {
            "command": "ltx-2-mlx",
            "mode": "distilled",
            "low_ram": True,
            "extra_args": ["--quiet"],
        },
        "gen": {
            "num_frames": 49,
            "fps": 24,
            "steps": 20,
            "cfg_scale": 3.0,
            "width": 768,
            "height": 512,
            "sampler": "Euler",
        },
        "install_hint": "Port MLX Metal natif, 16 Go minimum. Lity installe le runtime ltx-2-mlx au premier lancement.",
    },
)

# Integrated local backends first: diffusers in-process, MLX through an isolated
# ltx-2-mlx runtime installed by Lity. ComfyUI remains external.
_BACKEND_PRIORITY = {"diffusers": 0, "mlx": 0, "comfyui": 2}


def rank_video_models(
    hardware: dict[str, Any], installed: list[str] | None = None
) -> list[dict[str, Any]]:
    """Rank local video-generation models for this device.

    Mirrors :func:`rank_image_models`. Rows carry the backend so the UI can
    explain whether Lity can use the model through the integrated diffusers
    engine or whether a dedicated runtime is still needed.
    """
    profile = profile_from_hardware(hardware)
    installed_names = {name.lower() for name in installed or [] if name}
    rows: list[dict[str, Any]] = []
    for model in VIDEO_MODEL_CATALOG:
        evaluation = evaluate_model_complete(
            float(model["vram_gb"]), profile, float(model["params_b"])
        )
        row = {
            **model,
            "kind": "video",
            "installed": _is_installed(model, installed_names),
            "verdict": _verdict(evaluation),
            "speed": _video_speed_label(evaluation),
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


def _video_speed_label(evaluation: dict[str, Any]) -> str:
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
    app_native = [
        row
        for row in rows
        if row["backend"] == "diffusers" and row["status"] in ("can-run", "tight")
    ]
    pool = app_native or [row for row in rows if row["status"] in ("can-run", "tight")]
    if not pool:
        pool = [row for row in rows if row["status"] == "can-run-slow"]
    if pool:
        best = max(pool, key=lambda row: (float(row["score"]), float(row["params_b"])))
        best["recommended"] = True
