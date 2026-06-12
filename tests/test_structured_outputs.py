import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.intent_router import IntentRouter
from lity.services.ai.ollama_engine import AIEngine, _normalize_fact

_FACT_JSON = '{"found": true, "categorie": "user_profile", "cle": "prenom", "valeur": "Alex"}'
_INTENT_JSON = '{"action": "open_file", "path_raw": "a.py"}'


class _FakeOllama(types.ModuleType):
    """Stand-in 'ollama' module that records calls and returns canned data."""

    def __init__(self, chat_response=None, generate_response=None):
        super().__init__("ollama")
        self.calls = []
        self._chat_response = chat_response or {"message": {"content": ""}}
        self._generate_response = generate_response or {"response": ""}

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return self._chat_response

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return self._generate_response


class _patch_ollama:
    """Swap sys.modules['ollama'] for a fake, restoring it afterwards."""

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


class NormalizeFactTests(unittest.TestCase):
    def test_found_false_returns_none(self):
        self.assertIsNone(_normalize_fact({"found": False}))

    def test_requires_both_key_and_value(self):
        self.assertIsNone(_normalize_fact({"found": True, "categorie": "user_profile"}))
        self.assertIsNone(_normalize_fact({"cle": "prenom"}))  # no value

    def test_maps_fields(self):
        fact = _normalize_fact(
            {"found": True, "categorie": "user_profile", "cle": "prenom", "valeur": "Alex"}
        )
        self.assertEqual(fact, {"categorie": "user_profile", "cle": "prenom", "valeur": "Alex"})

    def test_accepts_previous_shape_without_found(self):
        # Backward compatible with the previous prompt shape (no "found", "cat"/"val").
        fact = _normalize_fact({"cat": "long_term_facts", "cle": "ville", "val": "Montréal"})
        self.assertEqual(
            fact, {"categorie": "long_term_facts", "cle": "ville", "valeur": "Montréal"}
        )

    def test_non_dict_returns_none(self):
        self.assertIsNone(_normalize_fact("nope"))
        self.assertIsNone(_normalize_fact(None))


class StructuredOutputCallTests(unittest.TestCase):
    def test_extract_fact_passes_schema_and_normalizes(self):
        fake = _FakeOllama(generate_response={"response": _FACT_JSON})
        with _patch_ollama(fake):
            fact = AIEngine(model="m").extract_fact("je m'appelle Alex")
        self.assertEqual(fact["valeur"], "Alex")
        kind, kwargs = fake.calls[0]
        self.assertEqual(kind, "generate")
        self.assertEqual(kwargs["format"]["required"], ["found"])
        self.assertEqual(kwargs["options"], {"temperature": 0})
        self.assertIs(kwargs["think"], False)

    def test_extract_fact_found_false_returns_none(self):
        fake = _FakeOllama(generate_response={"response": '{"found": false}'})
        with _patch_ollama(fake):
            self.assertIsNone(AIEngine(model="m").extract_fact("merci beaucoup !"))

    def test_intent_router_passes_schema_and_temperature(self):
        fake = _FakeOllama(chat_response={"message": {"content": _INTENT_JSON}})
        with _patch_ollama(fake):
            intent = IntentRouter(model="m").get_file_intent("ouvre a.py stp")
        self.assertEqual(intent["action"], "open_file")
        _, kwargs = fake.calls[0]
        self.assertEqual(kwargs["format"]["properties"]["action"]["enum"][0], "set_working_dir")
        self.assertEqual(kwargs["options"], {"temperature": 0})
        self.assertIs(kwargs["think"], False)


if __name__ == "__main__":
    unittest.main()
