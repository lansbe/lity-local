import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app.controller import AgentController
from lity.services.ai._engine_common import is_thinking_model, wants_thinking


class IsThinkingModelTests(unittest.TestCase):
    def test_reasoning_families_detected(self):
        for model in ("qwen3:8b", "qwen3:14b", "deepseek-r1:8b", "qwq:32b"):
            self.assertTrue(is_thinking_model(model), model)

    def test_non_reasoning_models_not_detected(self):
        for model in ("llama3.1:8b", "gemma2:9b", "qwen2.5:7b", "mistral:7b", ""):
            self.assertFalse(is_thinking_model(model), model)


class WantsThinkingTests(unittest.TestCase):
    def test_trivial_messages_skip(self):
        for text in ("merci", "ok", "bonjour", "salut ça va ?", "parfait merci", ""):
            self.assertFalse(wants_thinking(text), text)

    def test_reasoning_cues_think(self):
        for text in (
            "explique la récursivité",
            "pourquoi le ciel est bleu ?",
            "résous cette équation",
            "corrige ce bug",
            "implémente un tri rapide",
        ):
            self.assertTrue(wants_thinking(text), text)

    def test_arithmetic_thinks(self):
        self.assertTrue(wants_thinking("combien font 12 * 8"))

    def test_long_message_thinks(self):
        self.assertTrue(
            wants_thinking("voici un paragraphe avec beaucoup plus de huit mots à traiter")
        )


def _stub(model, last_user, *, think_routing=True, settings=True):
    """A minimal stand-in exposing exactly what _think_for_turn touches."""
    obj = SimpleNamespace()
    obj.settings = {"think_routing": think_routing} if settings else None
    obj.engine = SimpleNamespace(model=model)
    obj.memory = SimpleNamespace(get_context=lambda: [{"role": "user", "content": last_user}])
    return obj


class ThinkForTurnTests(unittest.TestCase):
    def test_reasoning_model_skips_on_trivial(self):
        self.assertIs(AgentController._think_for_turn(_stub("qwen3:8b", "merci")), False)

    def test_reasoning_model_defaults_on_complex(self):
        # None = model default → preserves the inline <think> the UI renders.
        self.assertIsNone(
            AgentController._think_for_turn(_stub("qwen3:8b", "explique la récursivité"))
        )

    def test_non_reasoning_model_never_routes(self):
        self.assertIsNone(AgentController._think_for_turn(_stub("llama3.1:8b", "merci")))

    def test_setting_off_disables_routing(self):
        self.assertIsNone(
            AgentController._think_for_turn(_stub("qwen3:8b", "merci", think_routing=False))
        )

    def test_no_settings_object_still_works(self):
        self.assertIs(
            AgentController._think_for_turn(_stub("qwen3:8b", "merci", settings=False)), False
        )


if __name__ == "__main__":
    unittest.main()
