from __future__ import annotations

from pathlib import Path
from typing import Any

from lity.services.characters.store import CHARACTER_EMOTIONS, CharacterStore

_EMOTION_PROMPTS: dict[str, str] = {
    "neutral": "neutral expression, relaxed face",
    "happy": "warm smile, bright eyes",
    "thoughtful": "thoughtful expression, slightly distant gaze",
    "surprised": "surprised expression, raised eyebrows",
    "worried": "worried expression, tense eyes",
    "sad": "sad expression, soft melancholy",
    "amused": "amused expression, subtle playful smile",
    "focused": "focused expression, attentive eyes",
}

_NEGATIVE = (
    "different person, inconsistent face, extra face, duplicate, blurry, low quality, "
    "distorted, deformed, watermark, text, logo, cropped head"
)


class CharacterImageGenerator:
    """Generate a local emotion portrait pack for a stored character."""

    def __init__(self, store: CharacterStore, image_manager: Any):
        self.store = store
        self.image_manager = image_manager

    def generate(self, character_id: str, emotions: list[str] | None = None) -> dict[str, Any]:
        if self.image_manager is None:
            return {
                "ok": False,
                "message": "Le moteur image n'est pas disponible.",
                "generated": [],
                "character": self.store.get(character_id),
            }
        profile = self.store.get(character_id)
        if profile is None:
            return {"ok": False, "message": "Personnage introuvable.", "generated": []}

        selected = self._selected_emotions(emotions)
        generated: list[str] = []
        errors: list[dict[str, str]] = []
        for emotion in selected:
            params = self._params(profile, emotion)
            result = self.image_manager.execute_generation(params)
            if result.get("type") != "image_generation_result":
                errors.append(
                    {"emotion": emotion, "message": str(result.get("message") or "Échec image.")}
                )
                continue
            content = result.get("content") if isinstance(result.get("content"), dict) else {}
            image_path = content.get("image_path")
            if not image_path:
                errors.append({"emotion": emotion, "message": "Aucun fichier image généré."})
                continue
            self.store.save_emotion_image(profile["id"], emotion, Path(str(image_path)))
            generated.append(emotion)

        updated = self.store.get(profile["id"])
        return {
            "ok": bool(generated) and not errors,
            "message": self._message(generated, errors),
            "generated": generated,
            "errors": errors,
            "character": updated,
        }

    @staticmethod
    def _selected_emotions(emotions: list[str] | None) -> list[str]:
        if not emotions:
            return list(CHARACTER_EMOTIONS)
        selected: list[str] = []
        for emotion in emotions:
            key = str(emotion or "").strip().lower()
            if key in CHARACTER_EMOTIONS and key not in selected:
                selected.append(key)
        return selected or list(CHARACTER_EMOTIONS)

    def _params(self, profile: dict[str, Any], emotion: str) -> dict[str, Any]:
        description = str(profile.get("description") or "").strip()
        style = str(profile.get("style") or "").strip()
        gender = str(profile.get("gender") or "").strip()
        name = str(profile.get("name") or "character").strip()
        identity = ", ".join(part for part in (name, gender, description, style) if part)
        prompt = (
            "consistent character portrait, same identity across every emotion, "
            f"{identity}, {_EMOTION_PROMPTS[emotion]}, front-facing, detailed face, "
            "clean background, high quality"
        )
        return {
            "description_originale": f"{name} - {CHARACTER_EMOTIONS[emotion]}",
            "prompt": prompt,
            "negative_prompt": _NEGATIVE,
            "style": style or "character portrait",
            "width": 512,
            "height": 512,
            "steps": 25,
            "cfg_scale": 7.5,
            "sampler": "Euler a",
            "seed": int(profile.get("seed", -1)),
            "checkpoint": str(profile.get("image_model") or ""),
            "character_id": profile["id"],
            "character_emotion": emotion,
        }

    @staticmethod
    def _message(generated: list[str], errors: list[dict[str, str]]) -> str:
        if generated and not errors:
            return f"{len(generated)} émotion(s) générée(s)."
        if generated:
            return f"{len(generated)} émotion(s) générée(s), {len(errors)} échec(s)."
        if errors:
            return errors[0]["message"]
        return "Aucune émotion générée."
