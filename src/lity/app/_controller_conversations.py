from __future__ import annotations

from typing import Any


class ConversationMixin:
    """Conversation metadata, export, and active-project binding."""

    @property
    def active_conversation_id(self) -> str:
        return getattr(self.memory, "active_conversation_id", "")

    def list_conversations(self) -> list[dict[str, Any]]:
        if hasattr(self.memory, "list_conversations"):
            return self.memory.list_conversations()
        return []

    def new_conversation(self) -> dict[str, Any]:
        if hasattr(self.memory, "new_conversation"):
            return self.memory.new_conversation(
                workdir=self._current_workdir(), model=getattr(self.engine, "model", "")
            )
        return {}

    def switch_conversation(self, conversation_id: str) -> bool:
        if not hasattr(self.memory, "switch_conversation"):
            return False
        if not self.memory.switch_conversation(conversation_id):
            return False
        self._restore_workdir()
        self._restore_model()
        return True

    def _restore_model(self) -> None:
        if not hasattr(self.memory, "get_active_model"):
            return
        model = self.memory.get_active_model()
        if model:
            self.engine.model = model
            if hasattr(self.router, "model"):
                self.router.model = model

    def set_conversation_pinned(self, conversation_id: str, pinned: bool) -> list[dict[str, Any]]:
        if hasattr(self.memory, "set_pinned"):
            self.memory.set_pinned(conversation_id, pinned)
        return self.list_conversations()

    def export_conversation(
        self, conversation_id: str | None = None, fmt: str = "markdown"
    ) -> dict[str, Any]:
        cid = conversation_id or self.active_conversation_id
        store = getattr(self.memory, "conversations", None)
        conversation = store.get_conversation(cid) if store is not None else None
        if conversation is None:
            return {"ok": False, "content": "", "filename": ""}
        title = conversation.get("title", "conversation")
        if fmt == "json":
            import json

            return {
                "ok": True,
                "content": json.dumps(conversation, ensure_ascii=False, indent=2),
                "filename": f"{_safe_filename(title)}.json",
            }
        lines = [f"# {title}", ""]
        roles = {"user": "Vous", "assistant": self.assistant_name, "system": "Système"}
        for message in conversation.get("messages", []):
            who = roles.get(message.get("role"), str(message.get("role")))
            lines.append(f"**{who}**\n\n{message.get('content', '')}\n")
        return {
            "ok": True,
            "content": "\n".join(lines),
            "filename": f"{_safe_filename(title)}.md",
        }

    def set_working_dir(self, path: str) -> tuple[bool, str]:
        """Set the working directory and bind it to the active conversation."""
        success, message = self.files.set_working_dir(path)
        if success and hasattr(self.memory, "set_conversation_workdir"):
            self.memory.set_conversation_workdir(self._current_workdir())
        return success, message

    def _current_workdir(self) -> str:
        working_dir = getattr(self.files, "working_dir", None)
        return str(working_dir) if working_dir else ""

    def _restore_workdir(self) -> None:
        workdir = (
            self.memory.get_active_workdir() if hasattr(self.memory, "get_active_workdir") else ""
        )
        if hasattr(self.files, "reset"):
            self.files.reset()
        if workdir:
            self.files.set_working_dir(workdir)

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        if hasattr(self.memory, "rename_conversation"):
            return bool(self.memory.rename_conversation(conversation_id, title))
        return False

    def active_message_count(self) -> int:
        if hasattr(self.memory, "get_active_messages"):
            return len(self.memory.get_active_messages())
        return 0

    def generate_title(self, text: str) -> str | None:
        """Ask the model for a short conversation title (None if unavailable)."""
        generator = getattr(self.engine, "generate_title", None)
        if not callable(generator):
            return None
        try:
            return generator(text)
        except Exception:
            return None

    def set_ai_title(self, conversation_id: str, title: str) -> bool:
        """Apply an AI-generated title unless the user renamed the conversation."""
        if hasattr(self.memory, "set_ai_title"):
            return bool(self.memory.set_ai_title(conversation_id, title))
        return False

    def delete_conversation(self, conversation_id: str) -> str:
        if hasattr(self.memory, "delete_conversation"):
            return self.memory.delete_conversation(conversation_id)
        return ""

    def get_messages(self, conversation_id: str | None = None) -> list[dict[str, Any]]:
        if conversation_id and hasattr(self.memory, "conversations"):
            return self.memory.conversations.get_messages(conversation_id)
        if hasattr(self.memory, "get_active_messages"):
            return self.memory.get_active_messages()
        if hasattr(self.memory, "get_context"):
            return self.memory.get_context()
        return []

    def search_conversations(self, query: str) -> list[dict[str, Any]]:
        if hasattr(self.memory, "search_conversations"):
            return self.memory.search_conversations(query)
        return self.list_conversations()


def _safe_filename(name: str) -> str:
    import re

    cleaned = re.sub(r"[^\w\-. ]", "", str(name)).strip().replace(" ", "_")
    return cleaned[:60] or "conversation"
