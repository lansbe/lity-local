import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.intent_router import IntentRouter
from lity.services.files.manager import FileManager


class LatencyOptimizationTests(unittest.TestCase):
    def test_intent_router_avoids_llm_for_plain_chat(self):
        class NoLlmRouter(IntentRouter):
            def get_file_intent(self, user_input):
                raise AssertionError("plain chat should not call Ollama intent routing")

        result = NoLlmRouter().process_intent("raconte-moi une histoire", object())

        self.assertEqual(result["action"], "none")
        self.assertFalse(result["handled"])

    def test_intent_router_handles_obvious_open_file_without_llm(self):
        class NoLlmRouter(IntentRouter):
            def get_file_intent(self, user_input):
                raise AssertionError("obvious file commands should use heuristics")

        class Files:
            current_file_path = None

            def __init__(self):
                self.loaded = []

            def load_file(self, path, user_input=None):
                self.loaded.append((path, user_input))
                return True, f"Fichier chargé : {path}"

        files = Files()

        result = NoLlmRouter().process_intent("ouvre README.md", files)

        self.assertTrue(result["handled"])
        self.assertEqual(result["action"], "open_file")
        self.assertEqual(files.loaded, [("README.md", "ouvre README.md")])

    def test_intent_router_uses_current_file_for_active_file_commands(self):
        class NoLlmRouter(IntentRouter):
            def get_file_intent(self, user_input):
                raise AssertionError("active file commands should use heuristics")

        class Files:
            current_file_path = "src/app.py"

            def __init__(self):
                self.closed = []
                self.loaded = []

            def close_file(self, target=None):
                self.closed.append(target)
                return True, "Fichier fermé."

            def load_file(self, path, user_input=None):
                self.loaded.append((path, user_input))
                return True, f"Fichier chargé : {path}"

        files = Files()
        router = NoLlmRouter()

        close_result = router.process_intent("ferme le fichier actif", files)
        reload_result = router.process_intent("recharge le fichier actif", files)

        self.assertTrue(close_result["handled"])
        self.assertTrue(reload_result["handled"])
        self.assertEqual(files.closed, ["src/app.py"])
        self.assertEqual(files.loaded, [("src/app.py", None)])

    def test_intent_router_avoids_llm_for_chat_with_broad_file_words(self):
        class NoLlmRouter(IntentRouter):
            def get_file_intent(self, user_input):
                raise AssertionError("broad words like code/open should not route to Ollama")

        router = NoLlmRouter()

        for message in ["qu'est-ce qu'un code postal ?", "parle-moi d'OpenAI"]:
            with self.subTest(message=message):
                result = router.process_intent(message, object())

                self.assertEqual(result["action"], "none")
                self.assertFalse(result["handled"])

    def test_file_context_reuses_numbered_content_cache(self):
        class CountingFileManager(FileManager):
            def __init__(self):
                super().__init__()
                self.numbering_calls = 0

            def _add_line_numbers(self, content):
                self.numbering_calls += 1
                return super()._add_line_numbers(content)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("alpha\nbeta", encoding="utf-8")
            manager = CountingFileManager()
            manager.set_working_dir(root)
            manager.load_file("notes.txt")

            first = manager.get_context_for_ai()
            second = manager.get_context_for_ai()

            self.assertIn("1: alpha", first)
            self.assertEqual(first, second)
            self.assertEqual(manager.numbering_calls, 1)

    def test_recursive_inventory_refreshes_when_directory_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.py").write_text("print('old')", encoding="utf-8")
            manager = FileManager()
            manager.set_working_dir(root)

            self.assertEqual(manager.get_available_files(recursive=True), ["old.py"])

            (root / "new.py").write_text("print('new')", encoding="utf-8")
            success, message = manager.load_file("new.py", user_input="ouvre new.py")

            self.assertTrue(success, message)
            self.assertEqual(Path(manager.current_file_path).name, "new.py")


if __name__ == "__main__":
    unittest.main()
