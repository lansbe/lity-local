import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.editing.code_editor import CodeEditor
from lity.services.files.manager import FileManager


class FileScalingAndSecurityTests(unittest.TestCase):
    def test_recursive_inventory_ignores_generated_and_hidden_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('app')", encoding="utf-8")
            (root / ".env").write_text("DATABASE_URL=postgres://local", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("private", encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "x.js").write_text("x", encoding="utf-8")
            (root / "README.md").write_text("hello", encoding="utf-8")

            manager = FileManager()
            manager.set_working_dir(root)

            self.assertEqual(
                manager.get_available_files(recursive=True), ["README.md", "src/app.py"]
            )

    def test_list_files_without_loaded_files_returns_available_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            manager = FileManager()
            manager.set_working_dir(root)

            listing = manager.list_files()

            self.assertIn("Fichiers disponibles", listing)
            self.assertIn("README.md", listing)

    def test_context_budget_truncates_loaded_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "long.txt").write_text("a" * 200, encoding="utf-8")
            manager = FileManager(max_context_chars=80)
            manager.set_working_dir(root)
            manager.load_file("long.txt")

            context = manager.get_context_for_ai()

            self.assertLessEqual(len(context), 180)
            self.assertIn("[CONTENU TRONQUÉ]", context)

    def test_editor_refuses_to_write_common_secret_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            editor = CodeEditor()

            success, message = editor.create_file(
                "secrets.env",
                "OPENAI_API_KEY=REDACTED_TEST_VALUE",
                working_dir=root,
            )

            self.assertFalse(success)
            self.assertIn("secret", message.lower())

    def test_editor_refuses_database_urls_jwt_tokens_and_private_key_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            editor = CodeEditor()

            cases = [
                ("settings.py", 'DATABASE_URL = "postgres://user:pass@localhost/db"'),
                ("auth.py", 'JWT_SECRET = "super-secret-signing-value"'),
                ("tokens.txt", "PRIVATE_TOKEN=REDACTED_TEST_VALUE"),
                ("private.pem", "placeholder private key material"),
            ]

            for file_name, content in cases:
                with self.subTest(file_name=file_name):
                    success, message = editor.create_file(file_name, content, working_dir=root)

                    self.assertFalse(success)
                    self.assertIn("secret", message.lower())

    def test_editor_refuses_edit_when_final_file_contains_secret_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "tokens.txt"
            target.write_text("PRIVATE_TOKEN=placeholder\n", encoding="utf-8")
            editor = CodeEditor()

            success, message = editor.apply_edit(
                target,
                "placeholder",
                "local-value",
                working_dir=root,
            )

            self.assertFalse(success)
            self.assertIn("secret", message.lower())
            self.assertEqual(target.read_text(encoding="utf-8"), "PRIVATE_TOKEN=placeholder\n")


if __name__ == "__main__":
    unittest.main()
