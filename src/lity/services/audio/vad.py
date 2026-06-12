from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class VADConfig:
    """Tuning for the energy VAD. Frames are mono float samples in [-1, 1]."""

    frame_ms: int = 30  # duration of each pushed frame
    energy_threshold: float = 0.015  # RMS above this = speech
    start_frames: int = 3  # consecutive speech frames needed to START a segment
    silence_ms: int = 800  # trailing silence that ENDS a segment


class VoiceActivityDetector:
    """Streaming energy-based VAD for hands-free voice (no deps, fully testable).

    Feed it one audio frame at a time with :meth:`push`; it returns ``"start"``
    when speech begins, ``"stop"`` when a segment ends (after enough trailing
    silence), or ``None`` otherwise. Energy = RMS of the frame. This is the
    walkie-talkie → hands-free upgrade: not as sharp as silero/webrtc, but
    dependency-free and good enough to auto-detect when the user starts/stops
    talking. Barge-in = treat any ``"start"`` while TTS is playing as a request
    to stop speaking.
    """

    def __init__(self, config: VADConfig | None = None):
        self.cfg = config or VADConfig()
        self._speaking = False
        self._speech_run = 0
        self._silence_ms = 0

    @staticmethod
    def rms(frame: Sequence[float]) -> float:
        if not len(frame):
            return 0.0
        return (sum(float(x) * float(x) for x in frame) / len(frame)) ** 0.5

    @property
    def speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._speaking = False
        self._speech_run = 0
        self._silence_ms = 0

    def push(self, frame: Sequence[float]) -> str | None:
        """Process one frame; return 'start' / 'stop' / None."""
        is_speech = self.rms(frame) >= self.cfg.energy_threshold
        if not self._speaking:
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.cfg.start_frames:
                    self._speaking = True
                    self._speech_run = 0
                    self._silence_ms = 0
                    return "start"
            else:
                self._speech_run = 0
            return None
        # Currently in a speech segment: count trailing silence to decide the end.
        if is_speech:
            self._silence_ms = 0
        else:
            self._silence_ms += self.cfg.frame_ms
            if self._silence_ms >= self.cfg.silence_ms:
                self._speaking = False
                self._silence_ms = 0
                return "stop"
        return None
