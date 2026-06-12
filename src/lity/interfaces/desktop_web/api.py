from __future__ import annotations

import base64
import contextlib
import logging
import mimetypes
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lity.interfaces.desktop_web._api_conversations import ConversationApiMixin
from lity.interfaces.desktop_web._api_skills import SkillsApiMixin
from lity.interfaces.desktop_web._api_workspace import WorkspaceApiMixin
from lity.services.editing.write_policy import mode_from_yolo

logger = logging.getLogger(__name__)


def _image_to_data_url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        file_path = Path(path)
        if not file_path.exists():
            return None
        mime = mimetypes.guess_type(str(file_path))[0] or "image/png"
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def _character_to_payload(character: dict[str, Any] | None) -> dict[str, Any] | None:
    if character is None:
        return None
    payload = dict(character)
    emotions: dict[str, Any] = {}
    for key, value in (character.get("emotions") or {}).items():
        item = dict(value) if isinstance(value, dict) else {}
        item["image"] = _image_to_data_url(item.get("image_path")) or ""
        emotions[str(key)] = item
    payload["emotions"] = emotions
    payload["thumbnail"] = emotions.get("neutral", {}).get("image") or next(
        (item.get("image") for item in emotions.values() if item.get("image")), ""
    )
    return payload


def _characters_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload["characters"] = [
        _character_to_payload(character) for character in result.get("characters", [])
    ]
    if "character" in payload:
        payload["character"] = _character_to_payload(payload.get("character"))
    if "active_character" in payload:
        payload["active_character"] = _character_to_payload(payload.get("active_character"))
    return payload


EmitCallback = Callable[[str, dict[str, Any]], None]
FolderPicker = Callable[[], "str | None"]


class DesktopApi(WorkspaceApiMixin, ConversationApiMixin, SkillsApiMixin):
    """Bridge object exposed to the web frontend through pywebview's ``js_api``.

    Every method returns JSON-serializable values so pywebview can hand them
    back to JavaScript. Streaming tokens are pushed to the frontend through the
    injected ``emit`` callback (the desktop shell wires it to
    ``window.evaluate_js``); the final assistant result is also returned from
    :meth:`send_message` so the caller can ``await`` it.

    The class never imports ``webview`` itself, which keeps it unit-testable
    with a plain callback collector and no GUI.
    """

    def __init__(
        self,
        controller: Any,
        emit: EmitCallback | None = None,
        folder_picker: FolderPicker | None = None,
    ):
        self.controller = controller
        self._emit: EmitCallback = emit or (lambda event, payload: None)
        self._folder_picker = folder_picker
        self._cancelled = False
        self._busy = False
        defaults = controller.get_settings() if hasattr(controller, "get_settings") else {}
        self._yolo = bool(defaults.get("default_yolo", False))
        self._agent_mode = bool(defaults.get("default_agent", False)) or self._yolo
        # Web search is a per-SESSION opt-in: the toolbar pill always starts OFF
        # on launch, never restored from the persisted flag (so it never reads
        # "activé" with nothing behind it). If a previous session left it
        # persisted on, clear it so the health panel agrees web starts off.
        self._web_enabled = False
        if defaults.get("web_search_enabled") and hasattr(controller, "update_settings"):
            with contextlib.suppress(Exception):
                controller.update_settings({"web_search_enabled": False})
        self._allow_commands = False
        self._command_approval = False
        self._approval_seq = 0
        self._approvals: dict[int, dict[str, Any]] = {}
        self._title_thread: threading.Thread | None = None
        # Background model-pull queue (survives modal close/reopen; one at a time).
        self._pull_lock = threading.RLock()
        self._pull_queue: list[str] = []
        self._pull_current: str | None = None
        self._pull_progress: dict[str, Any] = {}
        self._pull_running = False
        self._pull_worker: threading.Thread | None = None
        # Set by :meth:`cancel_pull` to abort the in-flight download; the worker
        # resets it before each new pull.
        self._pull_cancel = False
        # One-click local SearXNG setup (Docker) — one install at a time.
        self._searxng_lock = threading.RLock()
        self._searxng_running = False
        self._searxng_installer: Any = None
        # Codex login can wait on a browser/device auth flow. Keep it off the UI call path.
        self._codex_login_lock = threading.RLock()
        self._codex_login_running = False
        # Claude login (`claude setup-token`) waits on the same kind of browser flow.
        self._claude_login_lock = threading.RLock()
        self._claude_login_running = False
        # Grok login (`grok login`) waits on the same kind of browser flow.
        self._grok_login_lock = threading.RLock()
        self._grok_login_running = False

    def set_emit(self, emit: EmitCallback) -> None:
        self._emit = emit

    def _bus(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._emit(event, payload)
        except Exception:  # pragma: no cover - never let UI push break the call
            logger.exception("Failed to emit event %s", event)

    def _active_character_payload(self) -> dict[str, Any] | None:
        character = (
            self.controller.get_active_character()
            if hasattr(self.controller, "get_active_character")
            else None
        )
        return _character_to_payload(character)

    # --------------------------------------------------------------- state
    def get_state(self) -> dict[str, Any]:
        return {
            "assistant_name": self.controller.assistant_name,
            "model": getattr(self.controller.engine, "model", ""),
            "workdir": self._workdir(),
            "active_conversation_id": self.controller.active_conversation_id,
            "conversations": self.controller.list_conversations(),
            "agent_mode": self._agent_mode,
            "allow_commands": self._allow_commands,
            "command_approval": self._command_approval,
            "yolo": self._yolo,
            "write_mode": mode_from_yolo(self._yolo).value,
            "web_search": self._web_enabled,
            "chat_provider": self.controller.get_settings().get("chat_provider", "ollama")
            if hasattr(self.controller, "get_settings")
            else "ollama",
            "rag_enabled": getattr(self.controller, "rag_enabled", False),
            "indexed_chunks": self.controller.index_size()
            if hasattr(self.controller, "index_size")
            else 0,
            "change_count": self._change_count(),
            "image_active": bool(self.controller.is_image_session_active())
            if hasattr(self.controller, "is_image_session_active")
            else False,
            "video_active": bool(self.controller.is_video_session_active())
            if hasattr(self.controller, "is_video_session_active")
            else False,
            "active_character": self._active_character_payload(),
        }

    def set_agent_mode(self, enabled: bool) -> dict[str, Any]:
        self._agent_mode = bool(enabled)
        return {"agent_mode": self._agent_mode}

    def set_allow_commands(self, enabled: bool) -> dict[str, Any]:
        self._allow_commands = bool(enabled)
        return {"allow_commands": self._allow_commands}

    def set_web_search(self, enabled: bool) -> dict[str, Any]:
        """Toggle the agent's web search/fetch tools for THIS session.

        Persisting mirrors the live choice so the health panel stays in sync
        during the session; it is NOT a sticky default — every launch starts web
        OFF (see ``__init__``), so the toolbar pill never reads "activé" with
        nothing behind it.
        """
        self._web_enabled = bool(enabled)
        if self._web_enabled:
            self._agent_mode = True
        if hasattr(self.controller, "update_settings"):
            try:
                self.controller.update_settings({"web_search_enabled": self._web_enabled})
            except Exception:  # pragma: no cover - settings are best-effort
                logger.info("could not persist web_search_enabled")
        return {"web_search": self._web_enabled, "agent_mode": self._agent_mode}

    # -------------------------------------------------------- web setup
    def _ensure_searxng_installer(self) -> Any:
        if self._searxng_installer is None:
            from lity.services.web.searxng_setup import SearxngInstaller

            self._searxng_installer = SearxngInstaller(self.controller.paths.config_dir)
        return self._searxng_installer

    def web_status(self) -> dict[str, Any]:
        """Is web search REALLY usable? (the toggle alone proves nothing).

        Reports SearXNG reachability, Docker availability, the fallback engines,
        and whether the user already RESOLVED web setup (installed SearXNG or
        chose the fallback). The UI uses this on the FIRST Web click to propose
        the automatic install — but only until the user has decided."""
        from lity.app._modutil import _module_available

        settings = (
            self.controller.get_settings() if hasattr(self.controller, "get_settings") else {}
        )
        url = (settings.get("searxng_url") or "").strip()
        try:
            status = self._ensure_searxng_installer().status(url)
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("web_status failed: %s", exc)
            status = {"url": url, "reachable": False, "docker": False, "container": "unknown"}
        status["fallback_ddg"] = _module_available("ddgs") or _module_available("duckduckgo_search")
        status["setup_resolved"] = bool(settings.get("web_setup_resolved", False))
        with self._searxng_lock:
            status["setup_running"] = self._searxng_running
        return status

    def codex_status(self) -> dict[str, Any]:
        if hasattr(self.controller, "codex_status"):
            return self.controller.codex_status()
        return {
            "available": False,
            "authenticated": False,
            "message": "Provider Codex non supporté.",
        }

    def codex_login(self) -> dict[str, Any]:
        if not hasattr(self.controller, "codex_login"):
            return {
                "ok": False,
                "running": False,
                "message": "Connexion Codex non supportée.",
                "status": self.codex_status(),
            }
        with self._codex_login_lock:
            if self._codex_login_running:
                return {
                    "ok": True,
                    "running": True,
                    "message": "Connexion Codex déjà en cours.",
                    "status": self.codex_status(),
                }
            result = self.controller.codex_login()
            if not result.get("ok"):
                return {
                    "ok": False,
                    "running": False,
                    "message": result.get("message", "Impossible de lancer Codex login."),
                    "status": self.codex_status(),
                }
            process = result.get("process")
            if process is None:
                return {
                    "ok": False,
                    "running": False,
                    "message": "Codex login n'a pas renvoyé de processus.",
                    "status": self.codex_status(),
                }
            self._codex_login_running = True
            threading.Thread(target=self._finish_codex_login, args=(process,), daemon=True).start()
            return {
                "ok": True,
                "running": True,
                "message": result.get("message", "Connexion Codex lancée."),
                "status": self.codex_status(),
            }

    def codex_models(self) -> dict[str, Any]:
        if hasattr(self.controller, "codex_models"):
            return self.controller.codex_models()
        return {
            "ok": False,
            "models": [],
            "default_model": "",
            "message": "Catalogue Codex non supporté.",
        }

    def _finish_codex_login(self, process: Any) -> None:
        message = ""
        ok = False
        try:
            output, _stderr = process.communicate(timeout=600)
            status = self.codex_status()
            message = (output or "").strip() or status.get("message", "")
            ok = bool(status.get("authenticated")) or getattr(process, "returncode", 1) == 0
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
            status = self.codex_status()
            message = "Connexion Codex expirée. Relance la connexion si nécessaire."
        except Exception as exc:  # pragma: no cover - defensive background path
            status = self.codex_status()
            message = str(exc)
        finally:
            with self._codex_login_lock:
                self._codex_login_running = False
        self._bus(
            "codex_login",
            {
                "stage": "done",
                "done": True,
                "ok": ok,
                "message": message
                or ("Codex est connecté." if ok else "Connexion Codex terminée."),
                "status": status,
            },
        )

    def claude_status(self) -> dict[str, Any]:
        if hasattr(self.controller, "claude_status"):
            return self.controller.claude_status()
        return {
            "available": False,
            "authenticated": False,
            "message": "Provider Claude non supporté.",
        }

    def claude_login(self) -> dict[str, Any]:
        if not hasattr(self.controller, "claude_login"):
            return {
                "ok": False,
                "running": False,
                "message": "Connexion Claude non supportée.",
                "status": self.claude_status(),
            }
        with self._claude_login_lock:
            if self._claude_login_running:
                return {
                    "ok": True,
                    "running": True,
                    "message": "Connexion Claude déjà en cours.",
                    "status": self.claude_status(),
                }
            result = self.controller.claude_login()
            if not result.get("ok"):
                return {
                    "ok": False,
                    "running": False,
                    "message": result.get("message", "Impossible de lancer la connexion Claude."),
                    "status": self.claude_status(),
                }
            process = result.get("process")
            if process is None:
                return {
                    "ok": False,
                    "running": False,
                    "message": "La connexion Claude n'a pas renvoyé de processus.",
                    "status": self.claude_status(),
                }
            self._claude_login_running = True
            threading.Thread(target=self._finish_claude_login, args=(process,), daemon=True).start()
            return {
                "ok": True,
                "running": True,
                "message": result.get("message", "Connexion Claude lancée."),
                "status": self.claude_status(),
            }

    def claude_models(self) -> dict[str, Any]:
        if hasattr(self.controller, "claude_models"):
            return self.controller.claude_models()
        return {
            "ok": False,
            "models": [],
            "default_model": "",
            "message": "Catalogue Claude non supporté.",
        }

    def usage(self) -> dict[str, Any]:
        """Per-model usage tally for the CLI providers (Codex/Claude/Grok)."""
        if hasattr(self.controller, "cli_usage"):
            return self.controller.cli_usage()
        empty = {
            "turns": 0,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "by_model": [],
        }
        return {"claude": dict(empty), "codex": dict(empty), "grok": dict(empty)}

    def _finish_claude_login(self, process: Any) -> None:
        message = ""
        ok = False
        try:
            output, _stderr = process.communicate(timeout=600)
            status = self.claude_status()
            message = (output or "").strip() or status.get("message", "")
            ok = bool(status.get("authenticated")) or getattr(process, "returncode", 1) == 0
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
            status = self.claude_status()
            message = "Connexion Claude expirée. Relance la connexion si nécessaire."
        except Exception as exc:  # pragma: no cover - defensive background path
            status = self.claude_status()
            message = str(exc)
        finally:
            with self._claude_login_lock:
                self._claude_login_running = False
        self._bus(
            "claude_login",
            {
                "stage": "done",
                "done": True,
                "ok": ok,
                "message": message
                or ("Claude est connecté." if ok else "Connexion Claude terminée."),
                "status": status,
            },
        )

    def grok_status(self) -> dict[str, Any]:
        if hasattr(self.controller, "grok_status"):
            return self.controller.grok_status()
        return {
            "available": False,
            "authenticated": False,
            "message": "Provider Grok non supporté.",
        }

    def grok_login(self, device_auth: bool = False) -> dict[str, Any]:
        if not hasattr(self.controller, "grok_login"):
            return {
                "ok": False,
                "running": False,
                "message": "Connexion Grok non supportée.",
                "status": self.grok_status(),
            }
        with self._grok_login_lock:
            if self._grok_login_running:
                return {
                    "ok": True,
                    "running": True,
                    "message": "Connexion Grok déjà en cours.",
                    "status": self.grok_status(),
                }
            result = self.controller.grok_login(device_auth=bool(device_auth))
            if not result.get("ok"):
                return {
                    "ok": False,
                    "running": False,
                    "message": result.get("message", "Impossible de lancer la connexion Grok."),
                    "status": self.grok_status(),
                }
            process = result.get("process")
            if process is None:
                return {
                    "ok": False,
                    "running": False,
                    "message": "La connexion Grok n'a pas renvoyé de processus.",
                    "status": self.grok_status(),
                }
            self._grok_login_running = True
            threading.Thread(target=self._finish_grok_login, args=(process,), daemon=True).start()
            return {
                "ok": True,
                "running": True,
                "message": result.get("message", "Connexion Grok lancée."),
                "status": self.grok_status(),
            }

    def grok_models(self) -> dict[str, Any]:
        if hasattr(self.controller, "grok_models"):
            return self.controller.grok_models()
        return {
            "ok": False,
            "models": [],
            "default_model": "",
            "message": "Catalogue Grok non supporté.",
        }

    def _finish_grok_login(self, process: Any) -> None:
        message = ""
        ok = False
        try:
            output, _stderr = process.communicate(timeout=600)
            status = self.grok_status()
            message = (output or "").strip() or status.get("message", "")
            ok = bool(status.get("authenticated")) or getattr(process, "returncode", 1) == 0
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
            status = self.grok_status()
            message = "Connexion Grok expirée. Relance la connexion si nécessaire."
        except Exception as exc:  # pragma: no cover - defensive background path
            status = self.grok_status()
            message = str(exc)
        finally:
            with self._grok_login_lock:
                self._grok_login_running = False
        self._bus(
            "grok_login",
            {
                "stage": "done",
                "done": True,
                "ok": ok,
                "message": message or ("Grok est connecté." if ok else "Connexion Grok terminée."),
                "status": status,
            },
        )

    def mark_web_setup_resolved(self) -> dict[str, Any]:
        """Record that the user made a DELIBERATE web-setup choice (installed or
        chose the fallback), so the one-click modal stops auto-appearing. NOT
        called when the user merely dismisses the modal — dismissing isn't a
        decision, so the offer returns on the next Web click."""
        if hasattr(self.controller, "update_settings"):
            with contextlib.suppress(Exception):
                self.controller.update_settings({"web_setup_resolved": True})
        return {"ok": True}

    def setup_searxng(self) -> dict[str, Any]:
        """Launch the automatic local SearXNG install (Docker) in background.

        Progress and the final verdict are pushed via ``searxng_setup`` events;
        on success the working URL is persisted in the settings and the web
        facade is rebuilt, so search works immediately."""

        def persist(url: str) -> None:
            # A working install records the URL and resolves web setup, but does
            # NOT auto-enable the web toggle — installing SearXNG makes web search
            # AVAILABLE; the user still opts in (the setup modal flips it on for
            # the current session via set_web_search). Best-effort: a settings
            # hiccup must never corrupt the install result or abort the done event.
            if not hasattr(self.controller, "update_settings"):
                return
            try:
                self.controller.update_settings({"searxng_url": url, "web_setup_resolved": True})
            except Exception:  # pragma: no cover - settings are best-effort
                logger.info("could not persist SearXNG settings after install")

        def task() -> None:
            try:
                installer = self._ensure_searxng_installer()
                result = installer.install(self._bus, persist)
            except Exception as exc:  # pragma: no cover - defensive
                result = {"ok": False, "url": "", "message": str(exc)}
            finally:
                with self._searxng_lock:
                    self._searxng_running = False
            self._bus(
                "searxng_setup",
                {
                    "stage": "done",
                    "done": True,
                    "ok": result["ok"],
                    "url": result.get("url", ""),
                    "message": result.get("message", ""),
                },
            )

        # Set the running flag AND start the worker under the same lock, so a
        # rapid second click can never slip a duplicate install through the gap
        # between the check and the thread actually starting.
        with self._searxng_lock:
            if self._searxng_running:
                return {"ok": True, "running": True}
            self._searxng_running = True
            threading.Thread(target=task, daemon=True).start()
        return {"ok": True, "running": True}

    def set_yolo(self, enabled: bool) -> dict[str, Any]:
        """Autonomous write mode: the agent may write files directly."""
        self._yolo = bool(enabled)
        if self._yolo:
            self._agent_mode = True
        return {
            "yolo": self._yolo,
            "agent_mode": self._agent_mode,
            "write_mode": mode_from_yolo(self._yolo).value,
        }

    def set_command_approval(self, ask: bool) -> dict[str, Any]:
        self._command_approval = bool(ask)
        return {"command_approval": self._command_approval}

    def approve_command(self, request_id: int, allow: bool) -> dict[str, Any]:
        entry = self._approvals.get(int(request_id))
        if entry is not None:
            entry["allow"] = bool(allow)
            entry["event"].set()
        return {"ok": True}

    def _confirm_command(self, command: str) -> bool:
        self._approval_seq += 1
        request_id = self._approval_seq
        event = threading.Event()
        self._approvals[request_id] = {"event": event, "allow": False}
        self._bus("approval_request", {"id": request_id, "command": command})
        event.wait(timeout=300)
        entry = self._approvals.pop(request_id, {})
        return bool(entry.get("allow", False))

    def _workdir(self) -> str:
        working_dir = getattr(self.controller.files, "working_dir", None)
        return str(working_dir) if working_dir else ""

    # --------------------------------------------------------------- models
    def list_models(self) -> dict[str, Any]:
        settings = (
            self.controller.get_settings() if hasattr(self.controller, "get_settings") else {}
        )
        restore: tuple[Any, Any] | None = None
        engine = self.controller.engine
        if settings.get("chat_provider") == "lmstudio" and hasattr(engine, "chat_backend"):
            restore = (getattr(engine, "chat_backend", "ollama"), getattr(engine, "model", ""))
            engine.chat_backend = "ollama"
            engine.model = settings.get("selected_model", "") or restore[1]
        try:
            models = self.controller.engine.get_installed_models()
        except Exception as exc:
            logger.info("list_models failed: %s", exc)
            return {"models": [], "selected": "", "error": str(exc)}
        finally:
            if restore is not None:
                engine.chat_backend, engine.model = restore
        from lity.core.model_catalog import is_embedding_model

        # The chat-model selector must only offer models that can CHAT:
        # embeddings (bge-m3, nomic-embed-text…) are picked in Settings, and
        # selecting one here would silently break every conversation.
        chat_models = [name for name in models if not is_embedding_model(name)]
        if not chat_models:
            # Nothing usable installed: never surface the engine's default
            # name as if it were available.
            return {"models": [], "selected": "", "error": None}
        if restore is not None:
            selected = settings.get("selected_model", "") or ""
            if selected not in chat_models:
                selected = chat_models[0]
            return {"models": chat_models, "selected": selected, "error": None}
        selected = self.controller.sync_available_models(chat_models)
        return {"models": chat_models, "selected": selected, "error": None}

    def set_model(self, model_name: str) -> dict[str, Any]:
        return {"selected": self.controller.change_model(model_name)}

    def lmstudio_models(self) -> dict[str, Any]:
        if hasattr(self.controller, "lmstudio_models"):
            return self.controller.lmstudio_models()
        return {
            "ok": False,
            "models": [],
            "default_model": "",
            "base_url": "",
            "recommended": [],
            "message": "Provider LM Studio non supporté.",
        }

    # ----------------------------------------------------------- model mgmt
    def list_models_detailed(self) -> list[dict[str, Any]]:
        engine = self.controller.engine
        settings = (
            self.controller.get_settings() if hasattr(self.controller, "get_settings") else {}
        )
        restore: tuple[Any, Any] | None = None
        if settings.get("chat_provider") == "lmstudio" and hasattr(engine, "chat_backend"):
            restore = (getattr(engine, "chat_backend", "ollama"), getattr(engine, "model", ""))
            engine.chat_backend = "ollama"
            engine.model = settings.get("selected_model", "") or restore[1]
        try:
            if hasattr(engine, "get_models_detailed"):
                return engine.get_models_detailed()
            return []
        finally:
            if restore is not None:
                engine.chat_backend, engine.model = restore

    def pull_model(self, name: str) -> dict[str, Any]:
        """Enqueue a model download. Pulls run in the background, one at a time.

        Returns immediately with the current queue state so several models can be
        queued; progress is pushed via ``pull_progress`` / ``pull_done`` events
        and can be restored after a modal reopen via :meth:`pull_status`.
        """
        name = (name or "").strip()
        if not name:
            return self.pull_status()
        if not hasattr(self.controller.engine, "pull_model"):
            return {**self.pull_status(), "error": "Non supporté."}
        with self._pull_lock:
            if name != self._pull_current and name not in self._pull_queue:
                self._pull_queue.append(name)
            # `_pull_running` is flipped under the same lock the worker uses to
            # exit, so a freshly enqueued model can never be left without a worker.
            if not self._pull_running:
                self._pull_running = True
                self._pull_worker = threading.Thread(target=self._pull_drain, daemon=True)
                self._pull_worker.start()
        status = self.pull_status()
        self._bus("pull_progress", {"name": self._pull_current or name, **status})
        return status

    def pull_status(self) -> dict[str, Any]:
        """Snapshot of the download queue, so the modal can restore on reopen."""
        with self._pull_lock:
            return {
                "active": self._pull_current,
                "queue": list(self._pull_queue),
                "progress": dict(self._pull_progress),
            }

    def cancel_pull(self, name: str = "") -> dict[str, Any]:
        """Cancel a download. With no name (or the active model's name) the
        in-flight pull is aborted; a queued model's name just drops it from the
        queue. Either way the new queue snapshot is returned and broadcast."""
        name = (name or "").strip()
        with self._pull_lock:
            if not name or name == self._pull_current:
                # The worker checks this flag between chunks and stops the pull.
                self._pull_cancel = True
            elif name in self._pull_queue:
                self._pull_queue.remove(name)
        status = self.pull_status()
        self._bus("pull_progress", {"name": self._pull_current, **status})
        return status

    def _invoke_pull(self, name: str, on_progress: Any, should_cancel: Any) -> dict[str, Any]:
        """Call the engine's ``pull_model``, passing ``should_cancel`` only when
        the engine accepts it so older/stub engines keep working."""
        import inspect

        pull = self.controller.engine.pull_model
        kwargs: dict[str, Any] = {"on_progress": on_progress}
        try:
            params = inspect.signature(pull).parameters
            accepts = "should_cancel" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):  # pragma: no cover - builtins lack a signature
            accepts = False
        if accepts:
            kwargs["should_cancel"] = should_cancel
        return pull(name, **kwargs)

    def _pull_drain(self) -> None:
        while True:
            with self._pull_lock:
                # `_pull_current` may already hold the next model, promoted by the
                # previous iteration's completion handler for a seamless handoff.
                # Its cancel flag was already armed (reset) at promotion time, so
                # we must NOT reset it again here: a cancel that lands during the
                # handoff window would otherwise be silently dropped. The flag is
                # reset *only* together with assigning a fresh model — here on a
                # queue dequeue, and below at promotion.
                if self._pull_current is None:
                    if not self._pull_queue:
                        self._pull_progress = {}
                        self._pull_running = False
                        return
                    self._pull_current = self._pull_queue.pop(0)
                    self._pull_cancel = False
                self._pull_progress = {"status": "Démarrage…"}
            name = self._pull_current
            self._bus("pull_progress", {"name": name, **self.pull_status()})
            try:
                result = self._invoke_pull(
                    name,
                    lambda progress, n=name: self._on_pull_progress(n, progress),
                    lambda: self._pull_cancel,
                )
            except Exception as exc:  # pragma: no cover - defensive
                result = {"ok": False, "message": str(exc)}
            # Promote the next queued model into the active slot *before*
            # announcing completion: the banner moves straight from A→B with no
            # empty flash, and clears entirely (active=None) when nothing is left
            # — so it never lingers on a model that already finished.
            with self._pull_lock:
                self._pull_current = self._pull_queue.pop(0) if self._pull_queue else None
                # Arm a clean cancel slate for the promoted model: this clears the
                # just-finished model's flag (so cancelling A never bleeds into B)
                # while leaving the window open for a *new* cancel aimed at B.
                self._pull_cancel = False
                self._pull_progress = {}
            self._bus(
                "pull_done",
                {
                    "name": name,
                    "ok": bool(result.get("ok")),
                    "cancelled": bool(result.get("cancelled")),
                    "message": result.get("message", ""),
                    "models": self.list_models_detailed(),
                    **self.pull_status(),
                },
            )

    def _on_pull_progress(self, name: str, progress: dict[str, Any]) -> None:
        with self._pull_lock:
            self._pull_progress = {
                "status": progress.get("status", ""),
                "completed": progress.get("completed", 0),
                "total": progress.get("total", 0),
            }
        self._bus("pull_progress", {"name": name, **progress, **self.pull_status()})

    def delete_model(self, name: str) -> dict[str, Any]:
        engine = self.controller.engine
        if not hasattr(engine, "delete_model"):
            return {"ok": False, "message": "Non supporté."}
        result = engine.delete_model(name)
        result["models"] = self.list_models_detailed()
        return result

    def model_info(self, name: str) -> dict[str, Any]:
        if hasattr(self.controller.engine, "model_info"):
            return self.controller.engine.model_info(name)
        return {}

    # ----------------------------------------------------------- git
    def git_status(self) -> dict[str, Any]:
        return (
            self.controller.git_status()
            if hasattr(self.controller, "git_status")
            else {"is_repo": False}
        )

    def git_diff(self, path: str | None = None) -> dict[str, Any]:
        return (
            self.controller.git_diff(path) if hasattr(self.controller, "git_diff") else {"diff": ""}
        )

    def git_branches(self) -> dict[str, Any]:
        return (
            self.controller.git_branches()
            if hasattr(self.controller, "git_branches")
            else {"branches": [], "current": ""}
        )

    def git_commit(self, message: str) -> dict[str, Any]:
        if hasattr(self.controller, "git_commit"):
            return self.controller.git_commit(message)
        return {"ok": False, "message": "Non supporté."}

    # ----------------------------------------------------------- export / pin
    def export_conversation(
        self, conversation_id: str | None = None, fmt: str = "markdown"
    ) -> dict[str, Any]:
        if hasattr(self.controller, "export_conversation"):
            return self.controller.export_conversation(conversation_id, fmt)
        return {"ok": False, "content": "", "filename": ""}

    def set_pinned(self, conversation_id: str, pinned: bool) -> dict[str, Any]:
        if hasattr(self.controller, "set_conversation_pinned"):
            return {
                "conversations": self.controller.set_conversation_pinned(conversation_id, pinned)
            }
        return {"conversations": self.controller.list_conversations()}

    def set_rag(self, enabled: bool) -> dict[str, Any]:
        rag_enabled = (
            self.controller.set_rag(enabled) if hasattr(self.controller, "set_rag") else False
        )
        return {"rag_enabled": bool(rag_enabled)}

    def search_conversations(self, query: str) -> list[dict[str, Any]]:
        if hasattr(self.controller, "search_conversations"):
            return self.controller.search_conversations(query)
        return self.controller.list_conversations()

    # --------------------------------------------------------------- health
    def get_health(self) -> list[dict[str, Any]]:
        return self.controller.health() if hasattr(self.controller, "health") else []

    # --------------------------------------------------------------- memory
    def get_memory(self) -> dict[str, Any]:
        return self.controller.get_memory() if hasattr(self.controller, "get_memory") else {}

    def update_memory_entry(self, category: str, key: str, value: str) -> dict[str, Any]:
        if hasattr(self.controller, "update_memory"):
            return self.controller.update_memory(category, key, value)
        return self.get_memory()

    def delete_memory_entry(self, category: str, key: str) -> dict[str, Any]:
        if hasattr(self.controller, "delete_memory"):
            return self.controller.delete_memory(category, key)
        return self.get_memory()

    def clear_memory(self) -> dict[str, Any]:
        if hasattr(self.controller, "clear_memory"):
            return self.controller.clear_memory()
        return self.get_memory()

    # --------------------------------------------------------------- voice
    def audio_status(self) -> dict[str, Any]:
        return self.controller.audio_status() if hasattr(self.controller, "audio_status") else {}

    def start_recording(self) -> dict[str, Any]:
        if hasattr(self.controller, "start_recording"):
            return self.controller.start_recording()
        return {"ok": False, "message": "Voix non supportée."}

    def stop_recording(self) -> dict[str, Any]:
        if hasattr(self.controller, "stop_recording"):
            return self.controller.stop_recording()
        return {"ok": False, "text": ""}

    def speak(self, text: str) -> dict[str, Any]:
        if hasattr(self.controller, "speak"):
            # Emit "tts_done" when playback finishes so the hands-free voice loop
            # can re-arm the mic for the next turn.
            return self.controller.speak(text, on_finish=lambda: self._bus("tts_done", {}))
        return {"ok": False, "message": "Voix non supportée."}

    def stop_speaking(self) -> dict[str, Any]:
        if hasattr(self.controller, "stop_speaking"):
            return self.controller.stop_speaking()
        return {"ok": True}

    def download_voice(self, voice_id: str = "") -> dict[str, Any]:
        if hasattr(self.controller, "download_voice"):
            return self.controller.download_voice(voice_id)
        return {"ok": False, "message": "Voix non supportée."}

    def list_voices(self) -> dict[str, Any]:
        if hasattr(self.controller, "list_voices"):
            return self.controller.list_voices()
        return {"available": False, "installed": [], "current": "", "catalog": []}

    def set_voice(self, name: str) -> dict[str, Any]:
        if hasattr(self.controller, "set_voice"):
            return self.controller.set_voice(name)
        return {"ok": False}

    def model_suggestions(self) -> list[dict[str, str]]:
        if hasattr(self.controller, "model_suggestions"):
            return self.controller.model_suggestions()
        return []

    def model_recommendations(self) -> dict[str, Any]:
        """Hardware-aware ranking of Ollama models (best→worst for this device)."""
        if hasattr(self.controller, "model_recommendations"):
            return self.controller.model_recommendations()
        return {"hardware": {}, "models": []}

    def fetch_page(self, url: str) -> dict[str, Any]:
        """Fetch + clean a web page so it can be added to the chat as context."""
        if hasattr(self.controller, "fetch_page"):
            return self.controller.fetch_page(url)
        return {"ok": False, "url": url, "text": "", "error": "Non supporté."}

    def get_conversation_instructions(self) -> dict[str, Any]:
        if hasattr(self.controller, "get_conversation_instructions"):
            return self.controller.get_conversation_instructions()
        return {"instructions": "", "temperature": None}

    def set_conversation_instructions(
        self, instructions: str, temperature: float | None = None
    ) -> dict[str, Any]:
        if hasattr(self.controller, "set_conversation_instructions"):
            ok = self.controller.set_conversation_instructions(instructions or "", temperature)
            return {"ok": bool(ok), **self.controller.get_conversation_instructions()}
        return {"ok": False, "instructions": "", "temperature": None}

    def list_characters(self) -> dict[str, Any]:
        if hasattr(self.controller, "list_characters"):
            return _characters_result_payload(self.controller.list_characters())
        return {"characters": [], "active_character_id": ""}

    def create_character(self, data: dict[str, Any]) -> dict[str, Any]:
        if hasattr(self.controller, "create_character"):
            return _characters_result_payload(self.controller.create_character(data or {}))
        return {"ok": False, "message": "Personnages non supportés.", "characters": []}

    def update_character(self, character_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        if hasattr(self.controller, "update_character"):
            return _characters_result_payload(
                self.controller.update_character(character_id, patch or {})
            )
        return {"ok": False, "message": "Personnages non supportés.", "characters": []}

    def delete_character(self, character_id: str) -> dict[str, Any]:
        if hasattr(self.controller, "delete_character"):
            return _characters_result_payload(self.controller.delete_character(character_id))
        return {"ok": False, "characters": []}

    def set_conversation_character(self, character_id: str) -> dict[str, Any]:
        if hasattr(self.controller, "set_active_character"):
            return _characters_result_payload(
                self.controller.set_active_character(character_id or "")
            )
        return {"ok": False, "active_character": None, "characters": []}

    def generate_character_emotions(
        self, character_id: str, emotions: list[str] | None = None
    ) -> dict[str, Any]:
        if hasattr(self.controller, "generate_character_emotions"):
            return _characters_result_payload(
                self.controller.generate_character_emotions(character_id, emotions or None)
            )
        return {"ok": False, "message": "Génération indisponible.", "generated": []}

    def generation_stats(self) -> dict[str, Any]:
        """Speed + context usage of the latest turn (for the HUD)."""
        if hasattr(self.controller, "generation_stats"):
            return self.controller.generation_stats()
        return {"tokens_per_sec": 0.0, "context_used": 0, "context_length": 0, "usage_pct": 0}

    def open_external(self, url: str) -> dict[str, Any]:
        """Open a URL in the system browser (used by web source cards)."""
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return {"ok": False, "error": "URL invalide."}
        try:
            import webbrowser

            webbrowser.open(url)
            return {"ok": True}
        except Exception as exc:  # pragma: no cover - platform-specific
            return {"ok": False, "error": str(exc)}

    def extract_document(self, name: str, data: str) -> dict[str, Any]:
        """Extract text from an uploaded PDF/DOCX (base64 or data URL)."""
        raw = data.split(",", 1)[1] if isinstance(data, str) and data.startswith("data:") else data
        try:
            blob = base64.b64decode(raw)
        except Exception:
            return {"ok": False, "name": name, "text": "", "error": "Données illisibles."}
        from lity.services.files.documents import extract_document

        return {"name": name, **extract_document(name, blob)}

    def model_supports_tools(self, name: str = "") -> dict[str, Any]:
        """Report whether a model can tool-call (for the agent/web warning)."""
        supports = None
        if hasattr(self.controller, "model_supports_tools"):
            supports = self.controller.model_supports_tools(name or "")
        model = (name or "").strip() or getattr(self.controller.engine, "model", "")
        return {"model": model, "supports": supports}

    # --------------------------------------------------------------- image
    def image_active(self) -> bool:
        return bool(self.controller.is_image_session_active())

    def start_image_session(self) -> dict[str, Any]:
        return self.controller.start_image_session()

    def poll_image_launch(self) -> dict[str, Any]:
        return self.controller.poll_image_launch_status()

    def stop_image_session(self) -> dict[str, Any]:
        manager = self.controller.image_manager
        if manager is not None:
            manager.cancel_session()
            # Free the loaded checkpoint (several GB) when leaving image mode.
            with contextlib.suppress(Exception):
                manager.engine.unload()
        return {"active": False}

    def select_image_model(self, name: str) -> dict[str, Any]:
        """Choose which downloaded model the in-process engine generates with."""
        return self.controller.select_image_model(name or "")

    def image_pull_status(self) -> dict[str, Any]:
        return {"active": getattr(self, "_image_pull_active", None)}

    def cancel_image_download(self) -> dict[str, Any]:
        self._image_pull_cancel = True
        return {"ok": True}

    def download_image_model(self, name: str) -> dict[str, Any]:
        """Auto-download a non-Ollama image checkpoint into
        ~/Documents/Lity/Models/Images/<name>/, streaming with live progress."""
        from lity.core.image_model_advisor import IMAGE_MODEL_CATALOG

        model = next((m for m in IMAGE_MODEL_CATALOG if str(m.get("name")) == name), None)
        if model is None:
            return {"ok": False, "running": False, "message": f"Modèle image inconnu : {name}"}
        if getattr(self, "_image_pull_active", None):
            return {"ok": False, "running": True, "message": "Un téléchargement est déjà en cours."}

        dest_root = self.controller.paths.image_models_dir
        self._image_pull_active = name
        self._image_pull_cancel = False

        def on_progress(done: int, total: int, fn: str, idx: int, count: int) -> None:
            pct = round(done / total * 100) if total else 0
            self._bus(
                "image_pull_progress",
                {
                    "name": name,
                    "file": fn,
                    "downloaded": done,
                    "total": total,
                    "pct": pct,
                    "file_index": idx,
                    "file_count": count,
                },
            )

        def task() -> None:
            from lity.services.image_generation.model_download import (
                download_image_model as run_download,
            )

            try:
                result = run_download(
                    model,
                    dest_root,
                    on_progress=on_progress,
                    should_cancel=lambda: getattr(self, "_image_pull_cancel", False),
                )
            except Exception as exc:  # pragma: no cover - defensive
                result = {"ok": False, "message": str(exc)}
            finally:
                self._image_pull_active = None
            self._bus("image_pull_done", {"name": name, **result})

        threading.Thread(target=task, daemon=True).start()
        return {
            "ok": True,
            "running": True,
            "message": f"Téléchargement de {model.get('display_name', name)}…",
        }

    def _imageify(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, dict) and payload.get("type") == "image_generation_result":
            content = payload.get("content") or {}
            data_url = _image_to_data_url(content.get("image_path"))
            if data_url:
                payload["image"] = data_url
        return payload

    # --------------------------------------------------------------- video
    def video_active(self) -> bool:
        return bool(self.controller.is_video_session_active())

    def start_video_session(self) -> dict[str, Any]:
        return self.controller.start_video_session()

    def poll_video_launch(self) -> dict[str, Any]:
        return self.controller.poll_video_launch_status()

    def stop_video_session(self) -> dict[str, Any]:
        manager = self.controller.video_manager
        if manager is not None:
            manager.cancel_session()
            # Free the loaded model (several GB) when leaving video mode.
            with contextlib.suppress(Exception):
                manager.engine.unload()
        return {"active": False}

    def select_video_model(self, name: str) -> dict[str, Any]:
        """Choose which downloaded model the in-process engine generates with."""
        return self.controller.select_video_model(name or "")

    def video_pull_status(self) -> dict[str, Any]:
        return {"active": getattr(self, "_video_pull_active", None)}

    def cancel_video_download(self) -> dict[str, Any]:
        self._video_pull_cancel = True
        return {"ok": True}

    def download_video_model(self, name: str) -> dict[str, Any]:
        """Auto-download a video model into
        ~/Documents/Lity/Models/Videos/<name>/, streaming with live progress."""
        from lity.core.video_model_advisor import VIDEO_MODEL_CATALOG

        model = next((m for m in VIDEO_MODEL_CATALOG if str(m.get("name")) == name), None)
        if model is None:
            return {"ok": False, "running": False, "message": f"Modèle vidéo inconnu : {name}"}
        if getattr(self, "_video_pull_active", None):
            return {"ok": False, "running": True, "message": "Un téléchargement est déjà en cours."}

        dest_root = self.controller.paths.video_models_dir
        self._video_pull_active = name
        self._video_pull_cancel = False

        def on_progress(done: int, total: int, fn: str, idx: int, count: int) -> None:
            pct = round(done / total * 100) if total else 0
            self._bus(
                "video_pull_progress",
                {
                    "name": name,
                    "file": fn,
                    "downloaded": done,
                    "total": total,
                    "pct": pct,
                    "file_index": idx,
                    "file_count": count,
                },
            )

        def task() -> None:
            from lity.services.video_generation.model_download import (
                download_video_model as run_download,
            )

            try:
                result = run_download(
                    model,
                    dest_root,
                    on_progress=on_progress,
                    should_cancel=lambda: getattr(self, "_video_pull_cancel", False),
                )
            except Exception as exc:  # pragma: no cover - defensive
                result = {"ok": False, "message": str(exc)}
            finally:
                self._video_pull_active = None
            self._bus("video_pull_done", {"name": name, **result})

        threading.Thread(target=task, daemon=True).start()
        return {
            "ok": True,
            "running": True,
            "message": f"Téléchargement de {model.get('display_name', name)}…",
        }

    def _videoify(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, dict) and payload.get("type") == "video_generation_result":
            content = payload.get("content") or {}
            data_url = _image_to_data_url(content.get("video_path"))  # MIME-agnostic
            if data_url:
                payload["video"] = data_url
        return payload

    # --------------------------------------------------------------- settings
    def get_settings(self) -> dict[str, Any]:
        return self.controller.get_settings() if hasattr(self.controller, "get_settings") else {}

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        if hasattr(self.controller, "update_settings"):
            return self.controller.update_settings(patch)
        return {}
