import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.editing.code_editor import CodeEditor

FILE_CONTENT = """def alpha():
    return 1


def beta(x):
    if x:
        return x + 1
    return 0


def gamma():
    return "fin"
"""


class _Workspace:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.path = root / "module.py"
        self.path.write_text(FILE_CONTENT, encoding="utf-8")
        self.root = root
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()
        return False


class TolerantApplyEditTests(unittest.TestCase):
    def _apply(self, workspace, search, replace):
        editor = CodeEditor()
        return editor.apply_edit(workspace.path, search, replace, working_dir=workspace.root)

    def test_exact_match_still_works(self):
        with _Workspace() as ws:
            ok, message = self._apply(
                ws, "def alpha():\n    return 1", "def alpha():\n    return 2"
            )
            self.assertTrue(ok, message)
            self.assertIn("return 2", ws.path.read_text(encoding="utf-8"))

    def test_trailing_whitespace_difference_is_forgiven(self):
        with _Workspace() as ws:
            ok, message = self._apply(
                ws,
                "def beta(x):  \n    if x:   \n        return x + 1",
                "def beta(x):\n    if x:\n        return x + 2",
            )
            self.assertTrue(ok, message)
            self.assertIn("return x + 2", ws.path.read_text(encoding="utf-8"))

    def test_indent_shift_is_recovered_and_reindented(self):
        with _Workspace() as ws:
            # The model recopied the inner block WITHOUT its real indentation.
            ok, message = self._apply(
                ws,
                "if x:\n    return x + 1",
                "if x:\n    return x * 2",
            )
            self.assertTrue(ok, message)
            content = ws.path.read_text(encoding="utf-8")
            self.assertIn("        return x * 2", content)  # file's real depth restored
            self.assertNotIn("return x + 1", content)

    def test_near_miss_is_recovered_by_fuzzy(self):
        with _Workspace() as ws:
            # One recopied word differs ("retourn") — unique near-identical window.
            ok, message = self._apply(
                ws,
                'def gamma():\n    return "fin "',
                'def gamma():\n    return "FIN"',
            )
            self.assertTrue(ok, message)
            self.assertIn('return "FIN"', ws.path.read_text(encoding="utf-8"))

    def test_total_miss_reports_the_closest_fragment(self):
        with _Workspace() as ws:
            ok, message = self._apply(
                ws,
                "def beta(x):\n    while x:\n        x -= 1",
                "def beta(x):\n    pass",
            )
            self.assertFalse(ok)
            self.assertIn("le plus proche", message)
            self.assertIn("def beta(x):", message)  # real file text handed back
            # The file was not touched.
            self.assertEqual(ws.path.read_text(encoding="utf-8"), FILE_CONTENT)

    def test_ambiguous_block_still_refuses(self):
        with _Workspace() as ws:
            ws.path.write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
            ok, message = self._apply(ws, "x = 1", "x = 3")
            self.assertFalse(ok)
            self.assertIn("plusieurs fois", message)


class TolerantParsersTests(unittest.TestCase):
    def test_bold_file_line_and_four_chevrons_parse(self):
        editor = CodeEditor()
        text = (
            "Voici le fichier :\n"
            "**FILE: app/main.py**\n"
            "<<<< CREATE\n"
            "print('bonjour')\n"
            ">>>> CREATE\n"
        )
        blocks = editor.parse_create_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["file_path"], "app/main.py")
        self.assertEqual(blocks[0]["content"], "print('bonjour')")

    def test_lowercase_search_replace_parses(self):
        editor = CodeEditor()
        text = "FILE: a.py\n<<<<<<< search\nx = 1\n=======\nx = 2\n>>>>>>> replace\n"
        blocks = editor.parse_search_replace_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["search_content"], "x = 1")
        self.assertEqual(blocks[0]["replace_content"], "x = 2")

    def test_malformed_blocks_are_detected(self):
        editor = CodeEditor()
        broken = "FILE: a.py\n<<<<<<< CREATE\nprint('x')\n"  # never closed
        self.assertTrue(editor.detect_malformed_blocks(broken))
        fine = "FILE: a.py\n<<<<<<< CREATE\nprint('x')\n>>>>>>> CREATE\n"
        self.assertFalse(editor.detect_malformed_blocks(fine))
        plain = "Bonjour, voici une explication sans fichier."
        self.assertFalse(editor.detect_malformed_blocks(plain))


if __name__ == "__main__":
    unittest.main()
