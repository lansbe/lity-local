from __future__ import annotations

import logging
import threading
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lity.infrastructure.paths import AppPaths

logger = logging.getLogger(__name__)

PIPER_REPOSITORY = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


@dataclass(frozen=True)
class VoiceAsset:
    id: str
    label: str
    lang: str
    remote_stem: str

    def as_payload(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "lang": self.lang,
            "path": self.remote_stem,
        }


_VOICE_ASSETS: tuple[VoiceAsset, ...] = (
    VoiceAsset(
        id="fr_FR-upmc-medium",
        label="Français — UPMC (medium)",
        lang="fr",
        remote_stem="fr/fr_FR/upmc/medium/fr_FR-upmc-medium",
    ),
    VoiceAsset(
        id="fr_FR-siwis-medium",
        label="Français — Siwis (medium)",
        lang="fr",
        remote_stem="fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
    ),
    VoiceAsset(
        id="en_US-amy-medium",
        label="English US — Amy (medium)",
        lang="en",
        remote_stem="en/en_US/amy/medium/en_US-amy-medium",
    ),
    VoiceAsset(
        id="en_GB-alan-medium",
        label="English GB — Alan (medium)",
        lang="en",
        remote_stem="en/en_GB/alan/medium/en_GB-alan-medium",
    ),
)

PIPER_VOICES = [voice.as_payload() for voice in _VOICE_ASSETS]
DEFAULT_VOICE_PATH = _VOICE_ASSETS[0].remote_stem


@dataclass
class PlaybackState:
    generation: int = 0
    active: bool = False
    stream: Any | None = None


class TTSManager:
    """Small Piper wrapper used by the UI read-aloud action.

    The manager intentionally owns only local files and playback state. Importing
    Piper, NumPy and sounddevice is delayed until the user actually loads or
    plays a voice, so the application can start on systems without audio wheels.
    """

    def __init__(self, paths: AppPaths):
        self.voices_dir = paths.data_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.current_voice_name: str | None = None
        self.voice: Any | None = None
        self.volume = 0.5
        self._state = PlaybackState()
        self._lock = threading.RLock()
        installed = self.get_voices()
        if installed:
            self.load_voice(installed[0])

    def get_voices(self) -> list[str]:
        return sorted(path.stem for path in self._iter_voice_models())

    def download_default_voice(self, on_progress: Callable[[str], None] | None = None) -> None:
        self.download_voice(DEFAULT_VOICE_PATH, on_progress=on_progress)

    def download_voice(
        self,
        voice_path: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        remote_stem = voice_path.strip().strip("/")
        local_name = Path(remote_stem).name

        def worker() -> None:
            try:
                self._notify(on_progress, f"Téléchargement {local_name}…")
                self._fetch_voice_file(remote_stem, ".onnx")
                self._notify(on_progress, "Téléchargement configuration…")
                self._fetch_voice_file(remote_stem, ".onnx.json")
                self.load_voice(local_name)
                self._notify(on_progress, "Voix prête.")
            except Exception as exc:
                logger.warning("Piper voice download failed: %s", exc)
                self._notify(on_progress, "Erreur téléchargement.")

        threading.Thread(target=worker, daemon=True).start()

    def load_voice(self, voice_name: str) -> bool:
        onnx_path, config_path = self._voice_files(voice_name)
        if not onnx_path.exists() or not config_path.exists():
            return False
        try:
            from piper.voice import PiperVoice

            self.voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))
            self.current_voice_name = voice_name
            return True
        except Exception as exc:
            logger.warning("Piper voice load failed: %s", exc)
            self.voice = None
            return False

    def stop(self) -> None:
        with self._lock:
            self._state.generation += 1
            self._state.active = False
            stream = self._state.stream
            self._state.stream = None
        if stream is not None:
            try:
                stream.abort()
            except Exception as exc:
                logger.debug("TTS stream abort failed: %s", exc)

    def speak(
        self,
        text: str,
        on_start: Callable[[], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        message = (text or "").strip()
        if self.voice is None or not message:
            self._notify_done(on_finish)
            return

        with self._lock:
            self.stop()
            self._state.active = True
            generation = self._state.generation

        threading.Thread(
            target=self._play,
            args=(message, generation, on_start, on_finish),
            daemon=True,
        ).start()

    def _play(
        self,
        text: str,
        generation: int,
        on_start: Callable[[], None] | None,
        on_finish: Callable[[], None] | None,
    ) -> None:
        try:
            import numpy as np
            import sounddevice as sd

            self._notify(on_start)
            sample_rate = int(self.voice.config.sample_rate)
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
                with self._lock:
                    if generation != self._state.generation:
                        return
                    self._state.stream = stream
                for audio_chunk in self.voice.synthesize(text):
                    if not self._is_current(generation):
                        break
                    for segment in self._segments(audio_chunk.audio_int16_array, sample_rate):
                        if not self._is_current(generation):
                            break
                        stream.write(self._scale(segment, np))
        except Exception as exc:
            logger.warning("TTS playback failed: %s", exc)
        finally:
            finished = False
            with self._lock:
                if generation == self._state.generation:
                    self._state.active = False
                    self._state.stream = None
                    finished = True
            if finished:
                self._notify_done(on_finish)

    def _iter_voice_models(self) -> Iterable[Path]:
        for model_path in self.voices_dir.glob("*.onnx"):
            if model_path.with_suffix(".onnx.json").exists():
                yield model_path

    def _voice_files(self, voice_name: str) -> tuple[Path, Path]:
        model_path = self.voices_dir / f"{voice_name}.onnx"
        return model_path, model_path.with_suffix(".onnx.json")

    def _fetch_voice_file(self, remote_stem: str, suffix: str) -> None:
        url = f"{PIPER_REPOSITORY}/{remote_stem}{suffix}?download=true"
        target_name = f"{Path(remote_stem).name}{suffix}"
        urllib.request.urlretrieve(url, self.voices_dir / target_name)

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return self._state.active and generation == self._state.generation

    @staticmethod
    def _segments(audio: Any, sample_rate: int) -> Iterable[Any]:
        frame_count = max(1, int(sample_rate * 0.1))
        for start in range(0, len(audio), frame_count):
            yield audio[start : start + frame_count]

    def _scale(self, segment: Any, np: Any) -> Any:
        if self.volume == 1.0:
            return segment
        return (segment * self.volume).astype(np.int16)

    @staticmethod
    def _notify(
        callback: Callable[[], None] | Callable[[str], None] | None, message: str = ""
    ) -> None:
        if callback is None:
            return
        if message:
            callback(message)  # type: ignore[misc]
        else:
            callback()  # type: ignore[misc]

    @staticmethod
    def _notify_done(callback: Callable[[], None] | None) -> None:
        if callback is not None:
            callback()
