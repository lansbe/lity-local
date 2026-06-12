from __future__ import annotations

import threading
from typing import Any

from lity.app._modutil import _module_available


class AudioMixin:
    """Voice (STT dictation + TTS read-aloud) for AgentController.

    Uses ``self._stt`` / ``self._tts`` (lazy caches set in __init__) and
    ``self.paths``. Degrades gracefully when the audio deps are unavailable.
    """

    def _ensure_stt(self) -> Any:
        if self._stt is None and _module_available("faster_whisper"):
            from lity.services.audio.stt import STTManager

            self._stt = STTManager(self.paths)
        return self._stt

    def _ensure_tts(self) -> Any:
        if self._tts is None and _module_available("piper"):
            from lity.services.audio.tts import TTSManager

            self._tts = TTSManager(self.paths)
        return self._tts

    def audio_status(self) -> dict[str, Any]:
        stt = self._stt
        tts = self._tts
        return {
            "stt_available": _module_available("faster_whisper"),
            "stt_ready": bool(stt and getattr(stt, "is_model_ready", False)),
            "stt_error": getattr(stt, "model_load_error", None) if stt else None,
            "tts_available": _module_available("piper"),
            "has_voice": bool(tts and tts.get_voices()),
        }

    def start_recording(self) -> dict[str, Any]:
        stt = self._ensure_stt()
        if stt is None:
            return {"ok": False, "message": "Dépendances audio indisponibles sur ce système."}
        ok, message = stt.start_recording()
        return {"ok": bool(ok), "message": message}

    def stop_recording(self, timeout: float = 60.0) -> dict[str, Any]:
        stt = self._stt
        if stt is None:
            return {"ok": False, "text": "", "message": "Aucun enregistrement en cours."}
        done = threading.Event()
        holder: dict[str, str] = {"text": ""}

        def on_done(text: str) -> None:
            holder["text"] = text
            done.set()

        stt.stop_recording_and_transcribe(on_done)
        done.wait(timeout=timeout)
        return {"ok": True, "text": holder["text"]}

    def speak(self, text: str, on_finish: Any | None = None) -> dict[str, Any]:
        tts = self._ensure_tts()
        if tts is None:
            return {"ok": False, "message": "Dépendances audio indisponibles sur ce système."}
        if not tts.get_voices():
            return {"ok": False, "needs_voice": True, "message": "Aucune voix installée."}
        tts.speak(text, on_finish=on_finish)
        return {"ok": True}

    def stop_speaking(self) -> dict[str, Any]:
        if self._tts is not None:
            self._tts.stop()
        return {"ok": True}

    def list_voices(self) -> dict[str, Any]:
        from lity.services.audio.tts import PIPER_VOICES

        tts = self._ensure_tts()
        if tts is None:
            return {"available": False, "installed": [], "current": "", "catalog": PIPER_VOICES}
        return {
            "available": True,
            "installed": tts.get_voices(),
            "current": getattr(tts, "current_voice_name", "") or "",
            "catalog": PIPER_VOICES,
        }

    def download_voice(self, voice_id: str = "") -> dict[str, Any]:
        from lity.services.audio.tts import DEFAULT_VOICE_PATH, PIPER_VOICES

        tts = self._ensure_tts()
        if tts is None:
            return {"ok": False, "message": "Dépendances audio indisponibles sur ce système."}
        entry = next((voice for voice in PIPER_VOICES if voice["id"] == voice_id), None)
        path = entry["path"] if entry else DEFAULT_VOICE_PATH
        tts.download_voice(path)
        label = entry["label"] if entry else "voix par défaut"
        return {"ok": True, "message": f"Téléchargement lancé : {label}…"}

    def set_voice(self, name: str) -> dict[str, Any]:
        tts = self._ensure_tts()
        if tts is None:
            return {"ok": False}
        return {"ok": bool(tts.load_voice(name)), "current": getattr(tts, "current_voice_name", "")}
