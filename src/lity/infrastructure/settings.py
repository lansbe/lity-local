from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any


class SettingsStore:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._lock = RLock()
        self.settings: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.file_path.exists():
                self.settings = {}
                return
            try:
                content = self.file_path.read_text(encoding="utf-8").strip()
                data = json.loads(content) if content else {}
                self.settings = data if isinstance(data, dict) else {}
            except Exception:
                self.settings = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.settings[key] = value
            self.save()

    def save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(self.settings, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        temp_path.replace(self.file_path)
