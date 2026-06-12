from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from lity.infrastructure.paths import AppPaths
from lity.services.memory.conversation_store import ConversationStore

CONTEXT_WINDOW = 20


class MemoryManager:
    """Long-term profiles/facts (shared) plus per-conversation message history.

    Profiles and facts persist globally; the chat history is delegated to a
    :class:`ConversationStore` so the app can hold many conversations. The
    manager tracks an ``active_conversation_id`` and routes ``add_message`` /
    ``get_context`` / ``clear`` through it, keeping the original API intact for
    the CLI and Qt callers.
    """

    def __init__(
        self, paths: AppPaths | None = None, conversations: ConversationStore | None = None
    ):
        self.paths = paths or AppPaths.create()
        self._lock = RLock()
        self.conversations = conversations or ConversationStore(self.paths.conversations_dir)
        self.user_profile: dict[str, Any] = {}
        self.assistant_profile: dict[str, Any] = {}
        self.facts: dict[str, Any] = {}
        self.active_conversation_id: str = ""
        self.load_all()

    def _load_file(self, file_path: Path, default_value: Any) -> Any:
        if not file_path.exists():
            return default_value
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return default_value

    def _save_file(self, file_path: Path, data: Any) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        temp_path.replace(file_path)

    def load_all(self) -> None:
        with self._lock:
            self.user_profile = self._load_file(self.paths.user_profile_file, {})
            self.assistant_profile = self._load_file(self.paths.assistant_profile_file, {})
            self.facts = self._load_file(self.paths.facts_file, {})
            self.conversations.ensure_message_counts()
            self.active_conversation_id = self.conversations.ensure_default()
            # Clean up empty/abandoned conversations (keep the active draft).
            self.conversations.prune_empty(except_id=self.active_conversation_id)

    # --------------------------------------------------------- conversations
    def list_conversations(self) -> list[dict[str, Any]]:
        # Only conversations that actually have messages appear in the sidebar;
        # an empty/new conversation is a draft until the first message is sent.
        return [
            meta
            for meta in self.conversations.list_conversations()
            if meta.get("message_count", 0) > 0
        ]

    def search_conversations(self, query: str) -> list[dict[str, Any]]:
        return [
            meta for meta in self.conversations.search(query) if meta.get("message_count", 0) > 0
        ]

    def new_conversation(self, workdir: str = "", model: str = "") -> dict[str, Any]:
        with self._lock:
            # Reuse the current conversation if it is still an empty draft, so
            # clicking "New" repeatedly never piles up blank conversations.
            active = self.active_conversation_id
            if active and not self.conversations.get_messages(active):
                if workdir:
                    self.conversations.set_conversation_workdir(active, workdir)
                if model:
                    self.conversations.set_conversation_model(active, model)
                meta = self.conversations.get_meta(active)
                if meta is not None:
                    return meta
            meta = self.conversations.create_conversation(workdir=workdir, model=model)
            self.active_conversation_id = meta["id"]
            return meta

    def set_conversation_model(self, model: str) -> bool:
        return self.conversations.set_conversation_model(self.active_conversation_id, model)

    def get_active_model(self) -> str:
        return self.conversations.get_conversation_model(self.active_conversation_id)

    def set_active_character_id(self, character_id: str) -> bool:
        return self.conversations.set_character_id(self.active_conversation_id, character_id)

    def get_active_character_id(self) -> str:
        return self.conversations.get_character_id(self.active_conversation_id)

    def set_pinned(self, conversation_id: str, pinned: bool) -> bool:
        return self.conversations.set_pinned(conversation_id, pinned)

    def set_conversation_workdir(self, workdir: str) -> bool:
        return self.conversations.set_conversation_workdir(self.active_conversation_id, workdir)

    def get_active_workdir(self) -> str:
        return self.conversations.get_conversation_workdir(self.active_conversation_id)

    def switch_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            if not self.conversations.exists(conversation_id):
                return False
            self.active_conversation_id = conversation_id
            return True

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        return self.conversations.rename_conversation(conversation_id, title)

    def set_ai_title(self, conversation_id: str, title: str) -> bool:
        return self.conversations.set_ai_title(conversation_id, title)

    def delete_conversation(self, conversation_id: str) -> str:
        with self._lock:
            self.conversations.delete_conversation(conversation_id)
            if conversation_id == self.active_conversation_id:
                self.active_conversation_id = self.conversations.ensure_default()
            return self.active_conversation_id

    def get_active_messages(self) -> list[dict[str, Any]]:
        return self.conversations.get_messages(self.active_conversation_id)

    def drop_last_assistant(self) -> bool:
        return self.conversations.pop_last_assistant(self.active_conversation_id)

    def set_last_message_image(self, image: str) -> bool:
        return self.conversations.set_last_message_image(self.active_conversation_id, image)

    def set_last_message_video(self, video: str) -> bool:
        return self.conversations.set_last_message_video(self.active_conversation_id, video)

    def set_last_user_content(self, content: str) -> bool:
        return self.conversations.replace_last_user(self.active_conversation_id, content)

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.get_active_messages()

    # --------------------------------------------------------------- history
    def add_message(self, role: str, content: str, images: list[str] | None = None) -> None:
        with self._lock:
            self.conversations.add_message(
                self.active_conversation_id, role, content, images=images
            )

    def get_context(self) -> list[dict[str, Any]]:
        with self._lock:
            messages = self.conversations.get_messages(self.active_conversation_id)
            recent = list(messages[-CONTEXT_WINDOW:])
            summary, _count = self.conversations.get_summary(self.active_conversation_id)
            if summary and len(messages) > CONTEXT_WINDOW:
                # Keep older context alive on small local windows: prepend the
                # rolling summary as a system message ahead of the recent turns.
                return [
                    {"role": "system", "content": "[RÉSUMÉ DES ÉCHANGES PRÉCÉDENTS]\n" + summary}
                ] + recent
            return recent

    def get_active_instructions(self) -> dict[str, Any]:
        return self.conversations.get_instructions(self.active_conversation_id)

    def set_active_instructions(self, instructions: str, temperature: float | None) -> bool:
        return self.conversations.set_instructions(
            self.active_conversation_id, instructions, temperature
        )

    def get_conversation_summary(self) -> str:
        return self.conversations.get_summary(self.active_conversation_id)[0]

    def set_conversation_summary(self, summary: str, count: int) -> bool:
        return self.conversations.set_summary(self.active_conversation_id, summary, count)

    def pending_summary(self) -> dict[str, Any] | None:
        """Messages that scrolled out of the window and aren't summarized yet."""
        with self._lock:
            messages = self.conversations.get_messages(self.active_conversation_id)
            recent_start = max(0, len(messages) - CONTEXT_WINDOW)
            summary, count = self.conversations.get_summary(self.active_conversation_id)
            if recent_start <= count:
                return None
            chunk = messages[count:recent_start]
            if not chunk:
                return None
            return {"prior": summary, "messages": chunk, "count": recent_start}

    def clear(self) -> None:
        with self._lock:
            self.conversations.clear_messages(self.active_conversation_id)

    # -------------------------------------------------------------- profiles
    def update_user_profile(self, key: str, value: str) -> None:
        with self._lock:
            self.user_profile[key] = value
            self._save_file(self.paths.user_profile_file, self.user_profile)

    def update_assistant_profile(self, key: str, value: str) -> None:
        with self._lock:
            self.assistant_profile[key] = value
            self._save_file(self.paths.assistant_profile_file, self.assistant_profile)

    def add_fact(self, key: str, value: str) -> None:
        with self._lock:
            now = datetime.now().isoformat()
            if key in self.facts:
                self.facts[key]["count"] += 1
                self.facts[key]["last_seen"] = now
            else:
                self.facts[key] = {
                    "value": value,
                    "count": 1,
                    "first_seen": now,
                    "last_seen": now,
                }
            self._save_file(self.paths.facts_file, self.facts)

    def process_extracted_fact(self, fact_dict: dict[str, Any] | None) -> None:
        if not fact_dict:
            return
        category = fact_dict.get("categorie")
        key = fact_dict.get("cle")
        value = fact_dict.get("valeur")
        if not category or not key or not value:
            return

        if category == "user_profile":
            self.update_user_profile(str(key), str(value))
        elif category == "assistant_profile":
            self.update_assistant_profile(str(key), str(value))
        elif category == "long_term_facts":
            self.add_fact(str(key), str(value))

    def get_user_info_summary(self) -> str:
        with self._lock:
            if not self.user_profile and not self.facts:
                return ""

            summary = "\n--- CE QUE TU SAIS SUR L'HUMAIN ---\n"
            for key, value in self.user_profile.items():
                summary += f"- Son {key} est {value}\n"
            if self.facts:
                summary += "\nFaits marquants et préférences :\n"
                for fact in self.facts.values():
                    if fact.get("count", 0) > 1:
                        summary += f"- {fact.get('value')}\n"
            return summary + "--- FIN DES CONNAISSANCES ---\n"

    # ---------------------------------------------------- inspect / edit
    def get_memory(self) -> dict[str, Any]:
        with self._lock:
            facts = {
                key: (value.get("value") if isinstance(value, dict) else value)
                for key, value in self.facts.items()
            }
            return {
                "user_profile": dict(self.user_profile),
                "assistant_profile": dict(self.assistant_profile),
                "facts": facts,
            }

    def set_fact(self, key: str, value: str) -> None:
        with self._lock:
            now = datetime.now().isoformat()
            if key in self.facts and isinstance(self.facts[key], dict):
                self.facts[key]["value"] = value
                self.facts[key]["last_seen"] = now
            else:
                self.facts[key] = {
                    "value": value,
                    "count": 1,
                    "first_seen": now,
                    "last_seen": now,
                }
            self._save_file(self.paths.facts_file, self.facts)

    def delete_user_profile(self, key: str) -> None:
        with self._lock:
            if self.user_profile.pop(key, None) is not None:
                self._save_file(self.paths.user_profile_file, self.user_profile)

    def delete_assistant_profile(self, key: str) -> None:
        with self._lock:
            if self.assistant_profile.pop(key, None) is not None:
                self._save_file(self.paths.assistant_profile_file, self.assistant_profile)

    def delete_fact(self, key: str) -> None:
        with self._lock:
            if self.facts.pop(key, None) is not None:
                self._save_file(self.paths.facts_file, self.facts)

    def clear_all_memory(self) -> None:
        with self._lock:
            self.user_profile = {}
            self.assistant_profile = {}
            self.facts = {}
            self._save_file(self.paths.user_profile_file, self.user_profile)
            self._save_file(self.paths.assistant_profile_file, self.assistant_profile)
            self._save_file(self.paths.facts_file, self.facts)

    def get_assistant_info_summary(self) -> str:
        with self._lock:
            if not self.assistant_profile:
                return ""
            summary = "\n--- TON IDENTITÉ ET TES TRAITS ---\n"
            for key, value in self.assistant_profile.items():
                summary += f"- Ton/Ta {key} : {value}\n"
            return summary + "--- FIN DE TON IDENTITÉ ---\n"
