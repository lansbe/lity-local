from __future__ import annotations

from typing import Any


class SkillsApiMixin:
    """Bridge methods for the "Compétences" panel. Thin proxies to the
    controller, returning JSON-serializable values for pywebview."""

    controller: Any

    def list_skills(self) -> dict[str, Any]:
        if hasattr(self.controller, "list_skills"):
            return self.controller.list_skills()
        return {"enabled": False, "semantic": False, "dir": "", "skills": []}

    def toggle_skill(self, name: str, enabled: bool) -> dict[str, Any]:
        if hasattr(self.controller, "toggle_skill"):
            return self.controller.toggle_skill(str(name), bool(enabled))
        return {"ok": False, "message": "Compétences indisponibles."}

    def create_skill(
        self,
        name: str,
        description: str,
        body: str,
        when_to_use: str = "",
        triggers: list[str] | None = None,
    ) -> dict[str, Any]:
        if hasattr(self.controller, "create_skill"):
            return self.controller.create_skill(
                str(name),
                str(description or ""),
                str(body or ""),
                when_to_use=str(when_to_use or ""),
                triggers=list(triggers or []),
            )
        return {"ok": False, "message": "Compétences indisponibles."}

    def delete_skill(self, name: str) -> dict[str, Any]:
        if hasattr(self.controller, "delete_skill"):
            return self.controller.delete_skill(str(name))
        return {"ok": False, "message": "Compétences indisponibles."}
