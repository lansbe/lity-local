import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.editing.code_editor import CodeEditor
from lity.services.files.manager import FileManager


class FileAndEditorTests(unittest.TestCase):
    def test_loads_file_and_injects_numbered_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hello.py").write_text("print('Hello')\nprint('World')\n", encoding="utf-8")
            manager = FileManager()

            success, _message = manager.set_working_dir(root)
            self.assertTrue(success)
            success, _message = manager.load_file("hello.py")

            self.assertTrue(success)
            self.assertIn("1: print('Hello')", manager.get_context_for_ai())
            self.assertIn("hello.py", manager.list_files())

    def test_refuses_file_outside_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside.py"
            outside.write_text("print('outside')", encoding="utf-8")
            manager = FileManager()
            manager.set_working_dir(root)

            success, message = manager.load_file(outside)

            self.assertFalse(success)
            self.assertIn("hors du répertoire", message)

    def test_load_file_resolves_nested_file_from_user_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('nested')", encoding="utf-8")
            manager = FileManager()
            manager.set_working_dir(root)

            success, message = manager.load_file(
                "app.py",
                user_input="ouvre le fichier src/app.py s'il te plait",
            )

            self.assertTrue(success, message)
            self.assertIn("1: print('nested')", manager.get_context_for_ai())

    def test_refresh_loaded_file_updates_cached_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.py"
            target.write_text("print('old')\n", encoding="utf-8")
            manager = FileManager()
            manager.set_working_dir(root)
            self.assertTrue(manager.load_file("app.py")[0])

            target.write_text("print('new')\n", encoding="utf-8")

            self.assertTrue(manager.refresh_loaded_file("app.py"))
            self.assertEqual(manager.current_file_content, "print('new')\n")
            self.assertIn("1: print('new')", manager.get_context_for_ai())

    def test_editor_applies_single_exact_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "hello.py"
            file_path.write_text("print('Hello')\nprint('World')\n", encoding="utf-8")
            editor = CodeEditor()

            success, message = editor.apply_edit(
                file_path,
                "print('World')",
                "print('Lity')",
                working_dir=root,
            )

            self.assertTrue(success, message)
            self.assertEqual(
                file_path.read_text(encoding="utf-8"), "print('Hello')\nprint('Lity')\n"
            )

    def test_editor_refuses_ambiguous_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "hello.py"
            file_path.write_text("print('World')\nprint('World')\n", encoding="utf-8")
            editor = CodeEditor()

            success, message = editor.apply_edit(
                file_path,
                "print('World')",
                "print('Lity')",
                working_dir=root,
            )

            self.assertFalse(success)
            self.assertIn("plusieurs fois", message)


if __name__ == "__main__":
    unittest.main()
