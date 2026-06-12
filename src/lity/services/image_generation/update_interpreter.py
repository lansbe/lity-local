from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

CONFIRM_GENERATION_COMMANDS = {
    "ok",
    "okay",
    "oui",
    "yes",
    "go",
    "genere",
    "generer",
    "generate",
    "lance",
    "lancer",
    "valide",
    "valider",
    "confirme",
    "confirmer",
    "c est bon",
    "cest bon",
}

CANCEL_GENERATION_COMMANDS = {
    "annule",
    "annuler",
    "cancel",
    "stop",
    "arrete",
    "arreter",
}

SD_UPDATE_INTERPRETER_PROMPT = """Tu es un interprète d'intentions pour paramètres Stable Diffusion.
Classe le message parmi :
- "update_image_params" avec "updates"
- "confirm_generation"
- "cancel_generation"
- "normal_message"

Clés autorisées dans updates : prompt, negative_prompt, width, height, steps, cfg_scale, sampler, seed, checkpoint.
Réponds uniquement en JSON valide.
"""


class ImageParamUpdateInterpreter:
    def __init__(self, ai_engine: Any):
        self.engine = ai_engine

    def interpret_correction(
        self, current_params: dict[str, Any], user_message: str
    ) -> dict[str, Any]:
        direct_action = _direct_generation_action(user_message)
        if direct_action:
            return {"action": direct_action}

        payload = (
            "PARAMÈTRES ACTUELS :\n"
            f"{json.dumps(current_params, indent=2, ensure_ascii=False)}\n\n"
            f'MESSAGE DE L\'UTILISATEUR :\n"{user_message}"'
        )
        try:
            import ollama

            from lity.services.image_generation.prompt_builder import (
                image_llm_call_kwargs,
            )

            response = ollama.chat(
                model=self.engine.model,
                messages=[
                    {"role": "system", "content": SD_UPDATE_INTERPRETER_PROMPT},
                    {"role": "user", "content": payload},
                ],
                **image_llm_call_kwargs(self.engine),
            )
            raw_content = response["message"]["content"].strip()
            parsed_data = _extract_json_object(raw_content)
            action = parsed_data.get("action", "normal_message")
            result = {"action": action}
            if action == "update_image_params":
                result["updates"] = parsed_data.get("updates", {})
                result["user_message"] = parsed_data.get(
                    "user_message",
                    "J'ai pris en compte vos modifications.",
                )
            return result
        except Exception as exc:
            logger.warning("Image update interpretation fallback: %s", exc)
            return {"action": "normal_message"}


def _direct_generation_action(user_message: str) -> str | None:
    command = _normalize_command(user_message)
    if command in CONFIRM_GENERATION_COMMANDS:
        return "confirm_generation"
    if command in CANCEL_GENERATION_COMMANDS:
        return "cancel_generation"
    return None


def _normalize_command(user_message: str) -> str:
    normalized = unicodedata.normalize("NFD", user_message.strip().lower())
    without_accents = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    for char in ".,;:!?\"'`’":
        without_accents = without_accents.replace(char, " ")
    return " ".join(without_accents.split())


def _extract_json_object(raw_content: str) -> dict[str, Any]:
    start = raw_content.find("{")
    end = raw_content.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found")
    data = json.loads(raw_content[start:end])
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data
