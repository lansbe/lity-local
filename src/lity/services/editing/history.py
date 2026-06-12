from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any


class WorkspaceHistory:
    """Records the prior state of each written file so writes can be undone.

    Each entry is ``{"path", "prior"}`` where ``prior`` is the previous file
    content, or ``None`` if the file did not exist (i.e. it was created).
    ``undo_last`` reverts the most recent write: restoring the old content, or
    deleting the file if it was newly created.
    """

    def __init__(self, max_entries: int = 200):
        self._entries: list[dict[str, Any]] = []
        self._lock = RLock()
        self.max_entries = max_entries

    def record(self, path: str, prior: str | None) -> None:
        with self._lock:
            self._entries.append({"path": str(path), "prior": prior})
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries :]

    def can_undo(self) -> bool:
        return bool(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries = []

    def undo_last(self) -> dict[str, Any]:
        with self._lock:
            if not self._entries:
                return {"ok": False, "message": "Aucun changement à annuler."}
            entry = self._entries.pop()
        path = Path(entry["path"])
        try:
            if entry["prior"] is None:
                if path.exists():
                    path.unlink()
                return {"ok": True, "path": str(path), "message": f"Création annulée : {path.name}"}
            path.write_text(entry["prior"], encoding="utf-8")
            return {"ok": True, "path": str(path), "message": f"Modification annulée : {path.name}"}
        except Exception as exc:
            return {"ok": False, "path": str(path), "message": str(exc)}
