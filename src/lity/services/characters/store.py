from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

CHARACTER_EMOTIONS: dict[str, str] = {
    "neutral": "Neutre",
    "happy": "Heureux",
    "thoughtful": "Pensif",
    "surprised": "Surpris",
    "worried": "Inquiet",
    "sad": "Triste",
    "amused": "Amusé",
    "focused": "Concentré",
}

_TEXT_FIELDS = ("name", "description", "gender", "style", "instructions", "voice", "image_model")


def _now() -> str:
    return datetime.now().isoformat()


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(text).strip().lower()).strip("-")
    return value[:40] or "character"


def _clean_text(value: Any, limit: int = 4000) -> str:
    return str(value or "").strip()[:limit]


def _clean_seed(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return -1
    return parsed if -1 <= parsed <= 2_147_483_647 else -1


class CharacterStore:
    """Local JSON store for user-created characters.

    Each character owns a folder under ``AppPaths.characters_dir``:
    ``profile.json`` plus generated portrait variants under ``images/``.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            profiles = [
                profile
                for folder in sorted(self.root.iterdir())
                if folder.is_dir()
                for profile in [self._read_profile(folder)]
                if profile is not None
            ]
        return sorted(profiles, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get(self, character_id: str) -> dict[str, Any] | None:
        safe_id = self._safe_id(character_id)
        if not safe_id:
            return None
        with self._lock:
            return self._read_profile(self.root / safe_id)

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            name = _clean_text(data.get("name"), 120) or "Personnage"
            character_id = self._unique_id(name)
            now = _now()
            profile = self._normalize(
                {
                    "id": character_id,
                    "name": name,
                    "created_at": now,
                    "updated_at": now,
                    **data,
                }
            )
            self._write_profile(profile)
            return profile

    def update(self, character_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            profile = self.get(character_id)
            if profile is None:
                return None
            merged = {**profile}
            for field in (*_TEXT_FIELDS, "seed"):
                if field in patch:
                    merged[field] = patch[field]
            merged["updated_at"] = _now()
            updated = self._normalize(merged)
            self._write_profile(updated)
            return updated

    def delete(self, character_id: str) -> bool:
        safe_id = self._safe_id(character_id)
        if not safe_id:
            return False
        with self._lock:
            folder = self.root / safe_id
            if not folder.is_dir():
                return False
            shutil.rmtree(folder)
            return True

    def save_emotion_image(
        self, character_id: str, emotion: str, source_path: Path
    ) -> dict[str, Any]:
        emotion_key = self._emotion_key(emotion)
        with self._lock:
            profile = self.get(character_id)
            if profile is None:
                raise KeyError(f"Character not found: {character_id}")
            source = Path(source_path)
            if not source.is_file():
                raise FileNotFoundError(str(source))
            images_dir = self._folder(profile["id"]) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            target = images_dir / f"{emotion_key}{source.suffix.lower() or '.png'}"
            shutil.copyfile(source, target)
            profile["emotions"][emotion_key]["image_path"] = str(target)
            profile["updated_at"] = _now()
            self._write_profile(profile)
            return profile

    def _read_profile(self, folder: Path) -> dict[str, Any] | None:
        path = folder / "profile.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return self._normalize(data) if isinstance(data, dict) else None

    def _write_profile(self, profile: dict[str, Any]) -> None:
        folder = self._folder(profile["id"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "profile.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _normalize(self, data: dict[str, Any]) -> dict[str, Any]:
        character_id = self._safe_id(data.get("id")) or self._unique_id(data.get("name", ""))
        emotions = data.get("emotions") if isinstance(data.get("emotions"), dict) else {}
        normalized_emotions: dict[str, dict[str, str]] = {}
        for key, label in CHARACTER_EMOTIONS.items():
            item = emotions.get(key) if isinstance(emotions.get(key), dict) else {}
            normalized_emotions[key] = {
                "label": str(item.get("label") or label),
                "image_path": str(item.get("image_path") or ""),
            }
        profile: dict[str, Any] = {
            "id": character_id,
            "created_at": str(data.get("created_at") or _now()),
            "updated_at": str(data.get("updated_at") or _now()),
            "seed": _clean_seed(data.get("seed", -1)),
            "emotions": normalized_emotions,
        }
        for field in _TEXT_FIELDS:
            limit = 120 if field in {"name", "gender", "voice", "image_model"} else 4000
            profile[field] = _clean_text(data.get(field), limit)
        if not profile["name"]:
            profile["name"] = "Personnage"
        return profile

    def _unique_id(self, name: Any) -> str:
        prefix = _slug(str(name or "character"))
        for _ in range(20):
            candidate = f"{prefix}-{uuid.uuid4().hex[:8]}"
            if not (self.root / candidate).exists():
                return candidate
        return uuid.uuid4().hex

    def _folder(self, character_id: str) -> Path:
        return self.root / self._safe_id(character_id)

    @staticmethod
    def _safe_id(character_id: Any) -> str:
        value = str(character_id or "").strip()
        return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else ""

    @staticmethod
    def _emotion_key(emotion: str) -> str:
        key = str(emotion or "").strip().lower()
        if key not in CHARACTER_EMOTIONS:
            raise ValueError(f"Unknown character emotion: {emotion}")
        return key
