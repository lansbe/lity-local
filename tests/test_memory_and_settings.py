import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.infrastructure.paths import AppPaths
from lity.infrastructure.settings import SettingsStore
from lity.services.memory.json_memory import MemoryManager


class MemoryAndSettingsTests(unittest.TestCase):
    def test_settings_persist_json_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = SettingsStore(Path(tmp) / "settings.json")

            settings.set("selected_model", "llama3")

            reloaded = SettingsStore(Path(tmp) / "settings.json")
            self.assertEqual(reloaded.get("selected_model"), "llama3")

    def test_memory_uses_injected_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            memory = MemoryManager(paths=paths)

            memory.add_message("user", "bonjour")
            memory.process_extracted_fact(
                {"categorie": "user_profile", "cle": "prénom", "valeur": "Louis"}
            )

            self.assertEqual(memory.get_context()[-1]["content"], "bonjour")
            self.assertIn("Louis", memory.get_user_info_summary())

            # History and profile persist for a fresh manager on the same paths.
            reloaded = MemoryManager(paths=paths)
            self.assertEqual(reloaded.get_context()[-1]["content"], "bonjour")
            self.assertIn("Louis", reloaded.get_user_info_summary())


if __name__ == "__main__":
    unittest.main()
