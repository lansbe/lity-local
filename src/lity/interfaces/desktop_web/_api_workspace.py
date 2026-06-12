from __future__ import annotations

from typing import Any


class WorkspaceApiMixin:
    """Workdir, context files, reviewed edits, undo, and project indexing."""

    def set_workdir(self, path: str) -> dict[str, Any]:
        if hasattr(self.controller, "set_working_dir"):
            success, message = self.controller.set_working_dir(path)
        else:
            success, message = self.controller.files.set_working_dir(path)
        return {
            "success": bool(success),
            "message": message,
            "workdir": self._workdir(),
            "files": self.list_workspace_files()["files"],
        }

    def choose_workdir(self) -> dict[str, Any]:
        if self._folder_picker is None:
            return {
                "success": False,
                "message": "Sélecteur de dossier indisponible.",
                "workdir": self._workdir(),
            }
        path = self._folder_picker()
        if not path:
            return {"success": False, "message": "Sélection annulée.", "workdir": self._workdir()}
        return self.set_workdir(path)

    def list_workspace_files(self) -> dict[str, Any]:
        files: list[str] = []
        getter = getattr(self.controller.files, "get_available_files", None)
        if callable(getter):
            files = getter(recursive=True)
        return {"workdir": self._workdir(), "files": files}

    def get_loaded_files(self) -> list[dict[str, str]]:
        loaded = getattr(self.controller.files, "loaded_files", {}) or {}
        workdir = getattr(self.controller.files, "working_dir", None)
        result: list[dict[str, str]] = []
        for path_str, item in loaded.items():
            path = getattr(item, "path", None)
            name = path.name if path is not None else str(path_str)
            rel = name
            if workdir is not None and path is not None:
                try:
                    rel = str(path.relative_to(workdir))
                except Exception:
                    rel = name
            result.append({"path": str(path_str), "name": name, "rel": rel})
        return result

    def load_context_file(self, path: str) -> dict[str, Any]:
        success, message = self.controller.files.load_file(path)
        return {"success": bool(success), "message": message, "loaded": self.get_loaded_files()}

    def close_context_file(self, path: str) -> dict[str, Any]:
        success, message = self.controller.files.close_file(path)
        return {"success": bool(success), "message": message, "loaded": self.get_loaded_files()}

    def apply_create(self, block: dict[str, Any]) -> dict[str, Any]:
        success, message = self.controller.apply_create_block(block)
        if success and hasattr(self.controller.files, "refresh_files"):
            self.controller.files.refresh_files()
        return {
            "success": bool(success),
            "message": message,
            "files": self.list_workspace_files()["files"],
            "change_count": self._change_count(),
        }

    def apply_edit(self, block: dict[str, Any]) -> dict[str, Any]:
        success, message = self.controller.apply_edit_block(block)
        return {"success": bool(success), "message": message, "change_count": self._change_count()}

    def _change_count(self) -> int:
        return self.controller.changes_count() if hasattr(self.controller, "changes_count") else 0

    def undo_change(self) -> dict[str, Any]:
        if not hasattr(self.controller, "undo_last_change"):
            return {"ok": False, "message": "Annulation non supportée."}
        result = self.controller.undo_last_change()
        result["change_count"] = self._change_count()
        result["files"] = self.list_workspace_files()["files"]
        return result

    def index_project(self) -> dict[str, Any]:
        if not hasattr(self.controller, "index_project"):
            return {"ok": False, "chunks": 0, "message": "Indexation non supportée."}
        return self.controller.index_project()
