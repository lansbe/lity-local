import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.audio.vad import VADConfig, VoiceActivityDetector

LOUD = [0.5] * 160
QUIET = [0.0] * 160
SOFT = [0.004] * 160  # below the default energy threshold


class VoiceActivityDetectorTests(unittest.TestCase):
    def test_rms(self):
        self.assertEqual(VoiceActivityDetector.rms([]), 0.0)
        self.assertAlmostEqual(VoiceActivityDetector.rms([0.5, -0.5]), 0.5)

    def test_starts_after_consecutive_speech_frames(self):
        vad = VoiceActivityDetector()
        self.assertEqual([vad.push(LOUD) for _ in range(3)], [None, None, "start"])
        self.assertTrue(vad.speaking)
        self.assertIsNone(vad.push(LOUD))  # already speaking → no repeat event

    def test_soft_noise_never_starts(self):
        vad = VoiceActivityDetector()
        self.assertEqual([vad.push(SOFT) for _ in range(10)], [None] * 10)
        self.assertFalse(vad.speaking)

    def test_stops_after_trailing_silence(self):
        vad = VoiceActivityDetector(VADConfig(frame_ms=30, silence_ms=300))
        for _ in range(3):
            vad.push(LOUD)
        self.assertTrue(vad.speaking)
        events = [vad.push(QUIET) for _ in range(20)]
        self.assertIn("stop", events)  # 300ms / 30ms = 10 quiet frames → stop
        self.assertFalse(vad.speaking)

    def test_brief_silence_does_not_stop(self):
        vad = VoiceActivityDetector(VADConfig(frame_ms=30, silence_ms=300))
        for _ in range(3):
            vad.push(LOUD)
        # A couple of quiet frames (60ms) < 300ms → still speaking, no stop.
        self.assertEqual([vad.push(QUIET) for _ in range(2)], [None, None])
        self.assertTrue(vad.speaking)
        self.assertIsNone(vad.push(LOUD))  # speech resumes, silence counter resets

    def test_reset(self):
        vad = VoiceActivityDetector()
        for _ in range(3):
            vad.push(LOUD)
        vad.reset()
        self.assertFalse(vad.speaking)


if __name__ == "__main__":
    unittest.main()
