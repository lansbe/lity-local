import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai._engine_common import DEFAULT_NUM_CTX, _sampling_for
from lity.services.ai.ollama_engine import AIEngine


class _FakeOllama(types.ModuleType):
    def __init__(self):
        super().__init__("ollama")
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "ok"}}


class _patch_ollama:
    def __init__(self, fake):
        self.fake = fake
        self._saved = None

    def __enter__(self):
        self._saved = sys.modules.get("ollama")
        sys.modules["ollama"] = self.fake
        return self.fake

    def __exit__(self, *exc):
        if self._saved is not None:
            sys.modules["ollama"] = self._saved
        else:
            sys.modules.pop("ollama", None)
        return False


class SamplingTests(unittest.TestCase):
    def test_never_greedy_and_carries_num_ctx(self):
        for model in ["qwen3:8b", "llama3.1:8b", "mistral:7b", "deepseek-r1:8b", "gemma2:9b"]:
            params = _sampling_for(model)
            self.assertGreater(params["temperature"], 0.0, model)  # never greedy
            self.assertEqual(params["num_ctx"], DEFAULT_NUM_CTX, model)
            self.assertIn("top_p", params, model)

    def test_num_ctx_beats_ollama_4096_default(self):
        self.assertGreaterEqual(DEFAULT_NUM_CTX, 8192)  # the whole point: no 4096 truncation

    def test_qwen_thinking_profile_differs(self):
        plain = _sampling_for("qwen3:8b", thinking=False)
        thinking = _sampling_for("qwen3:8b", thinking=True)
        self.assertNotEqual(plain, thinking)
        self.assertIn("min_p", thinking)


class OptionsAndThinkTests(unittest.TestCase):
    def test_options_has_num_ctx_and_instruction_temperature_wins(self):
        engine = AIEngine(model="qwen3:8b")
        self.assertEqual(engine._options()["num_ctx"], DEFAULT_NUM_CTX)
        engine.temperature = 0.2  # per-conversation override
        self.assertEqual(engine._options()["temperature"], 0.2)

    def test_hardware_aware_num_ctx_override_wins(self):
        engine = AIEngine(model="qwen3:8b")
        engine.num_ctx = 32768  # set by the controller from the hardware probe
        self.assertEqual(engine._options()["num_ctx"], 32768)
        self.assertEqual(engine.effective_num_ctx(), 32768)
        engine.num_ctx = None
        self.assertEqual(engine.effective_num_ctx(), DEFAULT_NUM_CTX)

    def test_utility_model_routes_cheap_structured_calls(self):
        fake = _FakeOllama()
        with _patch_ollama(fake):
            engine = AIEngine(model="qwen3:8b")
            engine.utility_model = "qwen3:1.7b"
            engine.generate_structured("p", {"type": "object"}, prefer_utility=True)
            engine.generate_structured("p", {"type": "object"})  # default: big model
        self.assertEqual(fake.calls[0]["model"], "qwen3:1.7b")
        self.assertEqual(fake.calls[1]["model"], "qwen3:8b")

    def test_aux_tasks_prefer_the_utility_model(self):
        fake = _FakeOllama()
        with _patch_ollama(fake):
            engine = AIEngine(model="qwen3:8b")
            engine.utility_model = "qwen3:1.7b"
            engine.generate_title("Bonjour, parlons de mon projet Python")
        self.assertEqual(fake.calls[0]["model"], "qwen3:1.7b")

    def test_chat_with_tools_passes_think_false_and_num_ctx(self):
        fake = _FakeOllama()
        with _patch_ollama(fake):
            AIEngine(model="qwen3:8b").chat_with_tools(
                [{"role": "user", "content": "x"}], think=False
            )
        call = fake.calls[0]
        self.assertIs(call["think"], False)
        self.assertEqual(call["options"]["num_ctx"], DEFAULT_NUM_CTX)

    def test_chat_with_tools_omits_think_when_not_requested(self):
        fake = _FakeOllama()
        with _patch_ollama(fake):
            AIEngine(model="qwen3:8b").chat_with_tools([{"role": "user", "content": "x"}])
        self.assertNotIn("think", fake.calls[0])


if __name__ == "__main__":
    unittest.main()
