from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

DEFAULT_TITLE = "Nouvelle conversation"
TITLE_MAX_LEN = 48


def _now() -> str:
    return datetime.now().isoformat()


def derive_title(text: str, max_len: int = TITLE_MAX_LEN) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return DEFAULT_TITLE
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


class ConversationStore:
    """Persists multiple chat conversations as JSON files under a directory.

    Layout::

        <root>/index.json     -> [{"id", "title", "created_at", "updated_at"}, ...]
        <root>/<id>.json      -> {"id", "title", "created_at", "updated_at", "messages": [...]}

    The index is a denormalized cache for fast listing; the per-conversation
    files are the source of truth for messages. All public methods are
    thread-safe and write atomically.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    @property
    def _index_file(self) -> Path:
        return self.root / "index.json"

    def _conversation_file(self, conversation_id: str) -> Path:
        return self.root / f"{conversation_id}.json"

    # ------------------------------------------------------------- json io
    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _load_index(self) -> list[dict[str, Any]]:
        data = self._read_json(self._index_file, [])
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _save_index(self, index: list[dict[str, Any]]) -> None:
        self._write_json(self._index_file, index)

    @staticmethod
    def _meta(conversation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": conversation["id"],
            "title": conversation.get("title", DEFAULT_TITLE),
            "created_at": conversation.get("created_at", ""),
            "updated_at": conversation.get("updated_at", ""),
            "workdir": conversation.get("workdir", ""),
            "model": conversation.get("model", ""),
            "pinned": bool(conversation.get("pinned", False)),
            "character_id": conversation.get("character_id", ""),
            "message_count": len(conversation.get("messages", [])),
        }

    def get_meta(self, conversation_id: str) -> dict[str, Any] | None:
        conversation = self.get_conversation(conversation_id)
        return self._meta(conversation) if conversation else None

    def rebuild_index(self) -> None:
        """Recompute the index from the per-conversation files (e.g. to backfill counts)."""
        with self._lock:
            index: list[dict[str, Any]] = []
            for file_path in self.root.glob("*.json"):
                if file_path.name == "index.json":
                    continue
                data = self._read_json(file_path, None)
                if isinstance(data, dict) and data.get("id"):
                    index.append(self._meta(data))
            self._save_index(index)

    def ensure_message_counts(self) -> None:
        if any("message_count" not in item for item in self._load_index()):
            self.rebuild_index()

    def prune_empty(self, except_id: str | None = None) -> int:
        """Delete conversations with no messages (e.g. abandoned drafts). Returns count removed."""
        with self._lock:
            removed = 0
            kept: list[dict[str, Any]] = []
            for item in self._load_index():
                if item.get("id") != except_id and item.get("message_count", 0) == 0:
                    file_path = self._conversation_file(item["id"])
                    if file_path.exists():
                        file_path.unlink()
                    removed += 1
                else:
                    kept.append(item)
            self._save_index(kept)
            return removed

    def _touch_index(self, conversation: dict[str, Any]) -> None:
        index = self._load_index()
        meta = self._meta(conversation)
        for position, item in enumerate(index):
            if item.get("id") == conversation["id"]:
                index[position] = meta
                break
        else:
            index.append(meta)
        self._save_index(index)

    # --------------------------------------------------------------- public
    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock:
            index = self._load_index()
            by_recency = sorted(index, key=lambda item: item.get("updated_at", ""), reverse=True)
            # Pinned conversations float to the top (stable within each group).
            return sorted(by_recency, key=lambda item: not item.get("pinned", False))

    def create_conversation(
        self, title: str | None = None, workdir: str = "", model: str = ""
    ) -> dict[str, Any]:
        with self._lock:
            now = _now()
            conversation = {
                "id": uuid.uuid4().hex,
                "title": (title or DEFAULT_TITLE).strip() or DEFAULT_TITLE,
                "created_at": now,
                "updated_at": now,
                "workdir": workdir or "",
                "model": model or "",
                "pinned": False,
                "messages": [],
            }
            self._write_json(self._conversation_file(conversation["id"]), conversation)
            self._touch_index(conversation)
            return self._meta(conversation)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read_json(self._conversation_file(conversation_id), None)
            if not isinstance(data, dict):
                return None
            data.setdefault("id", conversation_id)
            data.setdefault("title", DEFAULT_TITLE)
            data.setdefault("workdir", "")
            data.setdefault("model", "")
            data.setdefault("pinned", False)
            data.setdefault("character_id", "")
            data.setdefault("messages", [])
            return data

    def exists(self, conversation_id: str) -> bool:
        return self._conversation_file(conversation_id).exists()

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        conversation = self.get_conversation(conversation_id)
        return list(conversation["messages"]) if conversation else []

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        images: list[str] | None = None,
    ) -> None:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return
            message: dict[str, Any] = {"role": role, "content": content, "timestamp": _now()}
            # Persist attachments (data URLs) so they survive reloads, redisplay
            # in the message bubble, and stay in context on follow-up turns.
            if images:
                message["images"] = [image for image in images if image]
            conversation["messages"].append(message)
            if role == "user" and self._has_default_title(conversation):
                conversation["title"] = derive_title(content)
            conversation["updated_at"] = _now()
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)

    def pop_last_assistant(self, conversation_id: str) -> bool:
        """Drop the trailing assistant message (used to regenerate a turn)."""
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            messages = conversation["messages"]
            if messages and messages[-1].get("role") == "assistant":
                messages.pop()
                conversation["updated_at"] = _now()
                self._write_json(self._conversation_file(conversation_id), conversation)
                self._touch_index(conversation)
                return True
            return False

    def replace_last_user(self, conversation_id: str, content: str) -> bool:
        """Replace the content of the last user message (used to edit & resend)."""
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            messages = conversation["messages"]
            for message in reversed(messages):
                if message.get("role") == "user":
                    message["content"] = content
                    conversation["updated_at"] = _now()
                    self._write_json(self._conversation_file(conversation_id), conversation)
                    self._touch_index(conversation)
                    return True
            return False

    def set_last_message_image(self, conversation_id: str, image: str) -> bool:
        """Attach a generated image (data URL) to the last message so it
        redisplays when the conversation is reopened."""
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None or not conversation["messages"] or not image:
                return False
            conversation["messages"][-1]["image"] = image
            self._write_json(self._conversation_file(conversation_id), conversation)
            return True

    def set_last_message_video(self, conversation_id: str, video: str) -> bool:
        """Attach a generated video (data URL) to the last message so it
        redisplays when the conversation is reopened."""
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None or not conversation["messages"] or not video:
                return False
            conversation["messages"][-1]["video"] = video
            self._write_json(self._conversation_file(conversation_id), conversation)
            return True

    def clear_messages(self, conversation_id: str) -> None:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return
            conversation["messages"] = []
            conversation["updated_at"] = _now()
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)

    def set_conversation_workdir(self, conversation_id: str, workdir: str) -> bool:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            conversation["workdir"] = workdir or ""
            conversation["updated_at"] = _now()
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)
            return True

    def get_conversation_workdir(self, conversation_id: str) -> str:
        conversation = self.get_conversation(conversation_id)
        return conversation.get("workdir", "") if conversation else ""

    def set_conversation_model(self, conversation_id: str, model: str) -> bool:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            conversation["model"] = model or ""
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)
            return True

    def get_conversation_model(self, conversation_id: str) -> str:
        conversation = self.get_conversation(conversation_id)
        return conversation.get("model", "") if conversation else ""

    def set_character_id(self, conversation_id: str, character_id: str) -> bool:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            conversation["character_id"] = str(character_id or "").strip()
            conversation["updated_at"] = _now()
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)
            return True

    def get_character_id(self, conversation_id: str) -> str:
        conversation = self.get_conversation(conversation_id)
        return str(conversation.get("character_id", "")) if conversation else ""

    def set_pinned(self, conversation_id: str, pinned: bool) -> bool:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            conversation["pinned"] = bool(pinned)
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)
            return True

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            cleaned = str(title).strip()
            if not cleaned:
                return False
            conversation["title"] = cleaned
            conversation["title_custom"] = True  # user-chosen: never auto-overwrite
            conversation["updated_at"] = _now()
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)
            return True

    def get_instructions(self, conversation_id: str) -> dict[str, Any]:
        """Per-conversation system prompt addition + temperature override."""
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return {"instructions": "", "temperature": None}
        instructions = conversation.get("instructions", "")
        return {
            "instructions": str(instructions or ""),
            "temperature": conversation.get("temperature"),
        }

    def set_instructions(
        self, conversation_id: str, instructions: str, temperature: float | None
    ) -> bool:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            conversation["instructions"] = instructions or ""
            conversation["temperature"] = temperature
            self._write_json(self._conversation_file(conversation_id), conversation)
            return True

    def get_summary(self, conversation_id: str) -> tuple[str, int]:
        """Rolling summary of older turns + how many messages it covers."""
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return "", 0
        return str(conversation.get("summary", "")), int(conversation.get("summary_count", 0))

    def set_summary(self, conversation_id: str, summary: str, count: int) -> bool:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                return False
            conversation["summary"] = summary or ""
            conversation["summary_count"] = int(count)
            self._write_json(self._conversation_file(conversation_id), conversation)
            return True

    def set_ai_title(self, conversation_id: str, title: str) -> bool:
        """Set an AI-generated title unless the user already renamed by hand."""
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None or conversation.get("title_custom"):
                return False
            cleaned = str(title).strip()
            if not cleaned:
                return False
            conversation["title"] = cleaned
            conversation["updated_at"] = _now()
            self._write_json(self._conversation_file(conversation_id), conversation)
            self._touch_index(conversation)
            return True

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            file_path = self._conversation_file(conversation_id)
            existed = file_path.exists()
            if existed:
                file_path.unlink()
            index = [item for item in self._load_index() if item.get("id") != conversation_id]
            self._save_index(index)
            return existed

    def search(self, query: str) -> list[dict[str, Any]]:
        """Full-text search across conversation titles and message contents."""
        needle = str(query).strip().lower()
        listed = self.list_conversations()
        if not needle:
            return listed
        matches: list[dict[str, Any]] = []
        for meta in listed:
            if needle in meta.get("title", "").lower():
                matches.append(meta)
                continue
            conversation = self.get_conversation(meta["id"])
            if conversation and any(
                needle in str(message.get("content", "")).lower()
                for message in conversation["messages"]
            ):
                matches.append(meta)
        return matches

    def ensure_default(self) -> str:
        with self._lock:
            existing = self.list_conversations()
            if existing:
                return existing[0]["id"]
            return self.create_conversation()["id"]

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _has_default_title(conversation: dict[str, Any]) -> bool:
        title = str(conversation.get("title", "")).strip()
        return title == DEFAULT_TITLE or not title
