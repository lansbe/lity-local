from __future__ import annotations

import base64
import json
import logging
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lity.infrastructure.paths import AppPaths
from lity.services.external import ServiceHealth

logger = logging.getLogger(__name__)


class StableDiffusionService:
    def __init__(self, paths: AppPaths, api_url: str = "http://127.0.0.1:7860"):
        self.paths = paths
        self.api_url = api_url.rstrip("/")

    def update_api_url(self, new_url: str) -> None:
        self.api_url = new_url.rstrip("/")

    def is_api_available(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.api_url}/sdapi/v1/sd-models", timeout=1.0
            ) as response:
                return response.status == 200
        except Exception:
            return False

    def check_health(self) -> ServiceHealth:
        if self.is_api_available():
            return ServiceHealth.up("Stable Diffusion", f"API disponible sur {self.api_url}")
        return ServiceHealth.down("Stable Diffusion", f"API indisponible sur {self.api_url}")

    def get_online_checkpoints(self) -> list[str]:
        data = self._get_json("/sdapi/v1/sd-models", timeout=2.0)
        checkpoints = []
        if isinstance(data, list):
            for model in data:
                if isinstance(model, dict):
                    checkpoints.append(model.get("title") or model.get("model_name") or "")
        return [checkpoint for checkpoint in checkpoints if checkpoint]

    def get_online_samplers(self) -> list[str]:
        data = self._get_json("/sdapi/v1/samplers", timeout=2.0)
        if not isinstance(data, list):
            return []
        return [
            sampler["name"] for sampler in data if isinstance(sampler, dict) and sampler.get("name")
        ]

    def get_generation_progress(self) -> dict[str, float]:
        data = self._get_json("/sdapi/v1/progress", timeout=0.5)
        if not isinstance(data, dict):
            return {"progress": 0.0, "eta": 0.0}
        return {
            "progress": round(float(data.get("progress", 0.0)) * 100.0, 1),
            "eta": round(float(data.get("eta_relative", 0.0)), 1),
        }

    def generate_image(self, params: dict[str, Any]) -> dict[str, Any]:
        final_seed = _safe_seed(params.get("seed", -1))
        if final_seed == -1:
            final_seed = random.randint(100000000, 999999999)

        normalized = normalize_generation_params(params, final_seed)
        try:
            self._switch_checkpoint(normalized.get("checkpoint", ""))
            payload = {
                "prompt": normalized["prompt"],
                "negative_prompt": normalized["negative_prompt"],
                "steps": normalized["steps"],
                "cfg_scale": normalized["cfg_scale"],
                "width": normalized["width"],
                "height": normalized["height"],
                "sampler_name": normalized["sampler_name"],
                "seed": normalized["seed"],
            }
            data = self._post_json("/sdapi/v1/txt2img", payload, timeout=120.0)
            image_path = self._save_first_image(data, final_seed)
            return {
                "status": "success",
                "message": "Génération réelle et sauvegarde physique effectuées avec succès.",
                "params": normalized,
                "image_path": str(image_path) if image_path else None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Stable Diffusion a rencontré une erreur : {exc}",
                "params": normalized,
                "image_path": None,
            }

    def _get_json(self, endpoint: str, timeout: float) -> Any:
        try:
            with urllib.request.urlopen(f"{self.api_url}{endpoint}", timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.info("Stable Diffusion GET %s failed: %s", endpoint, exc)
            return None

    def _post_json(self, endpoint: str, payload: dict[str, Any], timeout: float) -> Any:
        request = urllib.request.Request(
            f"{self.api_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _switch_checkpoint(self, checkpoint: str) -> None:
        if not checkpoint:
            return
        try:
            options = self._get_json("/sdapi/v1/options", timeout=2.0)
            current = options.get("sd_model_checkpoint", "") if isinstance(options, dict) else ""
            if current and checkpoint not in current and current not in checkpoint:
                self._post_json(
                    "/sdapi/v1/options", {"sd_model_checkpoint": checkpoint}, timeout=30.0
                )
        except Exception:
            return

    def _save_first_image(self, data: Any, seed: int) -> Path | None:
        if not isinstance(data, dict) or not data.get("images"):
            return None
        image_base64 = data["images"][0]
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        filename = f"image_{int(time.time())}_{seed}.png"
        image_path = self.paths.output_images_dir / filename
        image_path.write_bytes(image_bytes)
        return image_path


def normalize_generation_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "prompt": params.get("prompt", ""),
        "negative_prompt": params.get("negative_prompt", ""),
        "steps": _bounded_int(params.get("steps", 25), 1, 80, 25),
        "cfg_scale": _bounded_float(params.get("cfg_scale", 7.5), 1.0, 20.0, 7.5),
        "width": _bounded_int(params.get("width", 512), 256, 1536, 512),
        "height": _bounded_int(params.get("height", 512), 256, 1536, 512),
        "sampler_name": params.get("sampler", "Euler a"),
        "sampler": params.get("sampler", "Euler a"),
        "checkpoint": params.get("checkpoint", ""),
        "seed": seed,
        "style": params.get("style", "Non spécifié"),
    }


def _bounded_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(minimum, min(maximum, parsed))


def _safe_seed(value: Any) -> int:
    return _bounded_int(value, -1, 999999999, -1)
