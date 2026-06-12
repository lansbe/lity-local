from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Below this much total RAM, the chat LLM and the image pipeline cannot
# comfortably coexist (e.g. a 12B Q4 model ≈ 9 GB + SDXL fp16 ≈ 8 GB on a
# 16 GB Mac means heavy swapping). In that case the image-mode LLM calls ask
# Ollama to unload the model right after replying (keep_alive=0) so the
# render that follows gets the memory back.
_LLM_COEXIST_RAM_GB = 24.0


def image_llm_keep_alive() -> int | None:
    """``0`` (unload after the reply) on low-RAM machines, else ``None``."""
    try:
        import psutil

        total_gb = psutil.virtual_memory().total / 1024**3
    except Exception:  # pragma: no cover - psutil unavailable
        return None
    return 0 if total_gb < _LLM_COEXIST_RAM_GB else None


def image_llm_call_kwargs(engine: Any) -> dict[str, Any]:
    """Ollama kwargs for the image-mode helper calls (enrichment/interpreter).

    Always pins ``num_ctx``: without it Ollama falls back to its server
    default (the desktop app ships 262k!), ballooning the KV cache and
    forcing a full weight reload whenever the window differs from the chat's.

    - Low RAM → tiny window (4096) + ``keep_alive=0``: the load is transient
      and must leave the memory to the image render that follows.
    - Comfortable RAM → mirror the chat engine's window so the resident model
      is reused as-is, with no reload in either direction.
    """
    keep_alive = image_llm_keep_alive()
    if keep_alive is not None:
        return {"keep_alive": keep_alive, "options": {"num_ctx": 4096}}
    window = getattr(engine, "effective_num_ctx", None)
    num_ctx = int(window()) if callable(window) else 4096
    return {"options": {"num_ctx": num_ctx}}


SD_REFORMULATION_PROMPT = """Tu es un expert en ingénierie de prompt pour Stable Diffusion.
Convertis la description utilisateur en objet JSON de paramètres Stable Diffusion.

Règles :
1. Traduis les concepts en anglais.
2. Enrichis le prompt avec des modificateurs de style et qualité.
3. Déduis les dimensions : paysage 768x512, portrait 512x768, carré 512x512.
4. Fournis un negative prompt solide.
5. Propose steps=25, cfg_scale=7.5, seed=-1.
6. Choisis le sampler dans cette liste : {samplers_list}.

Réponds uniquement avec ce JSON :
{{
    "description_originale": "description française",
    "prompt": "prompt anglais enrichi",
    "negative_prompt": "negative prompt anglais",
    "style": "style",
    "width": 512,
    "height": 512,
    "steps": 25,
    "cfg_scale": 7.5,
    "sampler": "Euler a",
    "seed": -1,
    "checkpoint": ""
}}
"""


class ImagePromptBuilder:
    def __init__(self, ai_engine: Any):
        self.engine = ai_engine

    def build_initial_proposal(
        self,
        user_description: str,
        available_samplers: list[str] | None = None,
    ) -> dict[str, Any]:
        samplers = available_samplers or [
            "Euler a",
            "Euler",
            "Heun",
            "DPM++ 2M Karras",
            "DPM++ SDE Karras",
            "DDIM",
        ]
        system_prompt = SD_REFORMULATION_PROMPT.format(samplers_list=", ".join(samplers))
        try:
            import ollama

            response = ollama.chat(
                model=self.engine.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Description de l\'image : "{user_description}"'},
                ],
                **image_llm_call_kwargs(self.engine),
            )
            raw_content = response["message"]["content"].strip()
            parsed_data = _extract_json_object(raw_content)
            return _validated_proposal(parsed_data, user_description)
        except Exception as exc:
            logger.warning("Image prompt fallback: %s", exc)
            return _validated_proposal({}, user_description)


def _extract_json_object(raw_content: str) -> dict[str, Any]:
    start = raw_content.find("{")
    end = raw_content.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found")
    data = json.loads(raw_content[start:end])
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _validated_proposal(data: dict[str, Any], user_description: str) -> dict[str, Any]:
    return {
        "description_originale": data.get("description_originale", user_description),
        "prompt": data.get("prompt", user_description),
        "negative_prompt": data.get(
            "negative_prompt",
            "blurry, low quality, distorted, watermark, signature",
        ),
        "style": data.get("style", "digital art"),
        "width": _bounded_int(data.get("width", 512), 256, 1536, 512),
        "height": _bounded_int(data.get("height", 512), 256, 1536, 512),
        "steps": _bounded_int(data.get("steps", 25), 1, 80, 25),
        "cfg_scale": _bounded_float(data.get("cfg_scale", 7.5), 1.0, 20.0, 7.5),
        "sampler": data.get("sampler", "Euler a"),
        "seed": _bounded_int(data.get("seed", -1), -1, 2_147_483_647, -1),
        "checkpoint": data.get("checkpoint", ""),
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
