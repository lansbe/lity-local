import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.ollama_engine import _clean_title


class CleanTitleTests(unittest.TestCase):
    def test_strips_prefix_quotes_and_punctuation(self):
        self.assertEqual(_clean_title('Titre : "Bonjour le monde".'), "Bonjour le monde")
        self.assertEqual(_clean_title("Title: Configuration serveur!"), "Configuration serveur")

    def test_keeps_only_first_line(self):
        self.assertEqual(
            _clean_title("Décorateurs Python\nVoici une explication détaillée…"),
            "Décorateurs Python",
        )

    def test_caps_word_count(self):
        out = _clean_title("un deux trois quatre cinq six sept huit neuf dix")
        self.assertIsNotNone(out)
        self.assertLessEqual(len(out.split()), 8)

    def test_empty_or_blank_returns_none(self):
        self.assertIsNone(_clean_title(""))
        self.assertIsNone(_clean_title("   "))
        self.assertIsNone(_clean_title('"  "'))


if __name__ == "__main__":
    unittest.main()
