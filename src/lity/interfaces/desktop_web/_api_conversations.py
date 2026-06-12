from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from lity.app.results import result_to_dict

logger = logging.getLogger(__name__)


class ConversationApiMixin:
    """Conversation CRUD and chat runtime for DesktopApi."""

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.controller.list_conversations()

    def new_conversation(self) -> dict[str, Any]:
        payload = self.controller.new_conversation()
        payload["active_character"] = self._active_character_payload()
        return payload

    def switch_conversation(self, conversation_id: str) -> dict[str, Any]:
        success = self.controller.switch_conversation(conversation_id)
        return {
            "success": bool(success),
            "active_conversation_id": self.controller.active_conversation_id,
            "messages": self.controller.get_messages(),
            "workdir": self._workdir(),
            "files": self.list_workspace_files()["files"],
            "model": getattr(self.controller.engine, "model", ""),
            "active_character": self._active_character_payload(),
        }

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        return {
            "success": self.controller.rename_conversation(conversation_id, title),
            "conversations": self.controller.list_conversations(),
        }

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        active = self.controller.delete_conversation(conversation_id)
        return {
            "active_conversation_id": active,
            "conversations": self.controller.list_conversations(),
            "messages": self.controller.get_messages(),
            "active_character": self._active_character_payload(),
        }

    def get_messages(self, conversation_id: str | None = None) -> list[dict[str, Any]]:
        return self.controller.get_messages(conversation_id)

    def stop(self) -> dict[str, Any]:
        self._cancelled = True
        return {"stopped": True}

    def send_message(
        self,
        text: str,
        conversation_id: str | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._busy:
            return {"type": "error", "message": "Une requête est déjà en cours."}
        if conversation_id:
            self.controller.switch_conversation(conversation_id)

        slash_result = self.controller.process_slash_command(text)
        if slash_result:
            return {
                "type": "slash",
                "action": slash_result.get("action"),
                "message": str(slash_result.get("message", "")),
                "conversations": self.controller.list_conversations(),
                "active_conversation_id": self.controller.active_conversation_id,
                "active_character": self._active_character_payload(),
            }

        conversation_id_active = self.controller.active_conversation_id
        is_first_message = (
            self.controller.active_message_count() == 0
            if hasattr(self.controller, "active_message_count")
            else False
        )

        payload = self._run_and_finalize(lambda: self._run(text, images))

        using_cli = (
            hasattr(self.controller, "using_cli_provider") and self.controller.using_cli_provider()
        )
        if is_first_message and not using_cli and payload.get("type") in ("ai_response", "text"):
            self._generate_title_async(conversation_id_active, text)
        return payload

    def _generate_title_async(self, conversation_id: str, first_message: str) -> None:
        if not conversation_id or not hasattr(self.controller, "generate_title"):
            return

        def work() -> None:
            try:
                title = self.controller.generate_title(first_message)
                if title and self.controller.set_ai_title(conversation_id, title):
                    self._bus(
                        "title_update",
                        {
                            "id": conversation_id,
                            "title": title,
                            "conversations": self.controller.list_conversations(),
                        },
                    )
            except Exception:  # pragma: no cover - never let titling break a turn
                logger.exception("Background title generation failed")

        self._title_thread = threading.Thread(target=work, daemon=True)
        self._title_thread.start()

    def regenerate(self) -> dict[str, Any]:
        if not hasattr(self.controller, "regenerate_active"):
            return {"type": "error", "message": "Régénération non supportée."}
        return self._run_and_finalize(
            lambda: self.controller.regenerate_active(
                self._on_chunk, should_cancel=lambda: self._cancelled
            )
        )

    def edit_and_resend(self, new_text: str) -> dict[str, Any]:
        if not hasattr(self.controller, "edit_and_regenerate"):
            return {"type": "error", "message": "Édition non supportée."}
        return self._run_and_finalize(
            lambda: self.controller.edit_and_regenerate(
                new_text, self._on_chunk, should_cancel=lambda: self._cancelled
            )
        )

    def _run_and_finalize(self, runner: Callable[[], Any]) -> dict[str, Any]:
        self._cancelled = False
        self._busy = True
        try:
            payload = self._videoify(self._imageify(result_to_dict(runner())))
        except Exception as exc:
            logger.exception("message run failed")
            payload = {"type": "error", "message": str(exc)}
        finally:
            self._busy = False
        payload["cancelled"] = self._cancelled
        payload["conversations"] = self.controller.list_conversations()
        payload["active_conversation_id"] = self.controller.active_conversation_id
        payload["change_count"] = self._change_count()
        payload["active_character"] = self._active_character_payload()
        return payload

    def _run(self, text: str, images: list[str] | None = None) -> Any:
        if hasattr(self.controller, "using_cli_provider") and self.controller.using_cli_provider():
            if hasattr(self.controller, "process_user_message_stream"):
                return self.controller.process_user_message_stream(
                    text, self._on_chunk, should_cancel=lambda: self._cancelled, images=images
                )
            return self.controller.process_user_message_sync(text)
        if (
            self._agent_mode
            and not images
            and hasattr(self.controller, "process_user_message_agent")
        ):
            commands_on = self._allow_commands or self._yolo
            confirm = self._confirm_command if (commands_on and self._command_approval) else None
            return self.controller.process_user_message_agent(
                text,
                self._on_event,
                should_cancel=lambda: self._cancelled,
                allow_commands=commands_on,
                allow_write=self._yolo,
                allow_web=self._web_enabled,
                confirm=confirm,
            )
        if hasattr(self.controller, "process_user_message_stream"):
            return self.controller.process_user_message_stream(
                text, self._on_chunk, should_cancel=lambda: self._cancelled, images=images
            )
        return self.controller.process_user_message_sync(text)

    def _on_chunk(self, chunk: str) -> None:
        if chunk and not self._cancelled:
            self._bus("chunk", {"content": chunk})

    def _on_event(self, kind: str, payload: dict) -> None:
        if not self._cancelled:
            self._bus("step", {"kind": kind, **payload})
