import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.ollama_engine import _stats


class StatsTests(unittest.TestCase):
    def test_extracts_tokens_per_sec_and_context_used(self):
        stats = _stats({"eval_count": 100, "eval_duration": 2_000_000_000, "prompt_eval_count": 50})
        self.assertEqual(stats["tokens_per_sec"], 50.0)  # 100 tokens / 2.0 s
        self.assertEqual(stats["context_used"], 150)  # prompt + eval

    def test_handles_missing_fields(self):
        stats = _stats({})
        self.assertEqual(stats["tokens_per_sec"], 0.0)
        self.assertEqual(stats["context_used"], 0)

    def test_reads_object_attributes(self):
        class Resp:
            eval_count = 30
            eval_duration = 1_000_000_000
            prompt_eval_count = 10

        stats = _stats(Resp())
        self.assertEqual(stats["tokens_per_sec"], 30.0)
        self.assertEqual(stats["context_used"], 40)


if __name__ == "__main__":
    unittest.main()
