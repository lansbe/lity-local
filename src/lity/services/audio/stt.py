from __future__ import annotations

import logging
import threading
import wave
from collections.abc import Callable
from pathlib import Path

from lity.infrastructure.paths import AppPaths

logger = logging.getLogger(__name__)


class STTManager:
    def __init__(
        self,
        paths: AppPaths,
        on_model_ready: Callable[[], None] | None = None,
        on_model_error: Callable[[str], None] | None = None,
    ):
        self.paths = paths
        self.is_model_loading = False
        self.is_model_ready = False
        self.model_load_error: str | None = None
        self.model = None
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.stream = None
        self.on_model_ready = on_model_ready
        self.on_model_error = on_model_error
        self._load_model_in_background()

    def _load_model_in_background(self) -> None:
        self.is_model_loading = True
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self) -> None:
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
            self.is_model_loading = False
            self.is_model_ready = True
            if self.on_model_ready:
                self.on_model_ready()
        except Exception as exc:
            self.is_model_loading = False
            self.model_load_error = str(exc)
            logger.warning("STT model load failed: %s", exc)
            if self.on_model_error:
                self.on_model_error(str(exc))

    def start_recording(self) -> tuple[bool, str]:
        if not self.is_model_ready:
            return False, "Le modèle n'est pas encore prêt."
        if self.is_recording:
            return False, "Enregistrement déjà en cours."

        try:
            import sounddevice as sd

            self.audio_data = []
            self.is_recording = True

            def callback(indata, frames, time, status):  # noqa: ANN001
                if status:
                    logger.warning("STT audio status: %s", status)
                self.audio_data.append(indata.copy())

            self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback)
            self.stream.start()
            return True, "Enregistrement démarré."
        except Exception as exc:
            self.is_recording = False
            return False, f"Erreur lors de l'accès au micro : {exc}"

    def stop_recording_and_transcribe(self, on_transcription_done: Callable[[str], None]) -> None:
        if not self.is_recording:
            return

        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.audio_data:
            on_transcription_done("")
            return

        threading.Thread(
            target=self._transcribe_thread,
            args=(list(self.audio_data), on_transcription_done),
            daemon=True,
        ).start()

    def _transcribe_thread(self, audio_data, on_transcription_done: Callable[[str], None]) -> None:  # noqa: ANN001
        temp_file = self.paths.temp_recording_file
        try:
            import numpy as np

            audio_np = np.concatenate(audio_data, axis=0)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            _write_wav(temp_file, audio_int16, self.sample_rate)
            segments, _info = self.model.transcribe(str(temp_file), beam_size=5, language="fr")
            text = " ".join(segment.text for segment in segments)
            temp_file.unlink(missing_ok=True)
            on_transcription_done(text.strip())
        except Exception as exc:
            logger.warning("STT transcription failed: %s", exc)
            on_transcription_done("")


def _write_wav(path: Path, audio_int16, sample_rate: int) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
