from __future__ import annotations

import json
import logging
from typing import Any

# Reuse the image helper that keeps the chat LLM and the heavy render from
# fighting over RAM (unload-after-reply on <24 GB machines, pinned num_ctx).
from lity.services.image_generation.prompt_builder import image_llm_call_kwargs

logger = logging.getLogger(__name__)

VIDEO_REFORMULATION_PROMPT = """Tu es un expert en ingénierie de prompt pour la génération de vidéo (text-to-video).
Convertis la description utilisateur en objet JSON de paramètres de génération vidéo.

Règles :
1. Traduis les concepts en anglais.
2. Enrichis le prompt avec des descripteurs de mouvement, de caméra, d'ambiance et de qualité.
3. Garde la vidéo COURTE : c'est du local sur machine modeste (num_frames entre 25 et 81, fps 12 à 24).
4. Fournis un negative prompt solide (artefacts, scintillement, basse qualité).
5. Propose steps=25, cfg_scale=5.0, seed=-1.
6. Choisis le sampler dans cette liste : {samplers_list}.

Réponds uniquement avec ce JSON :
{{
    "description_originale": "description française",
    "prompt": "prompt anglais enrichi (mouvement + caméra)",
    "negative_prompt": "negative prompt anglais",
    "style": "style",
    "width": 832,
    "height": 480,
    "num_frames": 49,
    "fps": 15,
    "steps": 25,
    "cfg_scale": 5.0,
    "sampler": "UniPC",
    "seed": -1,
    "checkpoint": ""
}}
"""


class VideoPromptBuilder:
    def __init__(self, ai_engine: Any):
        self.engine = ai_engine

    def build_initial_proposal(
        self,
        user_description: str,
        available_samplers: list[str] | None = None,
    ) -> dict[str, Any]:
        samplers = available_samplers or ["UniPC", "Euler", "DDIM"]
        system_prompt = VIDEO_REFORMULATION_PROMPT.format(samplers_list=", ".join(samplers))
        try:
            import ollama

            response = ollama.chat(
                model=self.engine.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Description de la vidéo : "{user_description}"'},
                ],
                **image_llm_call_kwargs(self.engine),
            )
            raw_content = response["message"]["content"].strip()
            parsed_data = _extract_json_object(raw_content)
            return _validated_proposal(parsed_data, user_description)
        except Exception as exc:
            logger.warning("Video prompt fallback: %s", exc)
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
            "blurry, low quality, distorted, flickering, watermark, jpeg artifacts",
        ),
        "style": data.get("style", "cinematic"),
        "width": _bounded_int(data.get("width", 832), 256, 1280, 832),
        "height": _bounded_int(data.get("height", 480), 256, 1280, 480),
        "num_frames": _bounded_int(data.get("num_frames", 49), 9, 121, 49),
        "fps": _bounded_int(data.get("fps", 15), 6, 30, 15),
        "steps": _bounded_int(data.get("steps", 25), 1, 80, 25),
        "cfg_scale": _bounded_float(data.get("cfg_scale", 5.0), 1.0, 20.0, 5.0),
        "sampler": data.get("sampler", "UniPC"),
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
