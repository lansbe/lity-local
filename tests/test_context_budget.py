import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai._engine_common import (
    _INJECTED_CONTEXT_BUDGET_CHARS,
    _TRUNCATION_MARKER,
    budget_injected_context,
)
from lity.services.ai.ollama_engine import AIEngine


class BudgetInjectedContextTests(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        text = "[CONTEXTE PROJET]\nun petit extrait pertinent\n[/CONTEXTE]"
        self.assertEqual(budget_injected_context(text), text)

    def test_empty_is_unchanged(self):
        self.assertEqual(budget_injected_context(""), "")

    def test_oversized_is_clipped_within_budget(self):
        text = "x" * (_INJECTED_CONTEXT_BUDGET_CHARS * 3)
        out = budget_injected_context(text)
        self.assertLessEqual(len(out), _INJECTED_CONTEXT_BUDGET_CHARS)
        self.assertTrue(out.endswith(_TRUNCATION_MARKER))

    def test_head_is_preserved(self):
        head = "EXTRAIT IMPORTANT EN TÊTE\n"
        text = head + ("z" * (_INJECTED_CONTEXT_BUDGET_CHARS * 2))
        out = budget_injected_context(text)
        self.assertTrue(out.startswith("EXTRAIT IMPORTANT EN TÊTE"))  # head kept, tail dropped

    def test_custom_budget(self):
        out = budget_injected_context("a" * 1000, max_chars=200)
        self.assertLessEqual(len(out), 200)
        self.assertIn("tronqué", out)

    def test_snaps_to_line_boundary_when_cheap(self):
        # A clean newline near the cut should be preferred over mid-line slicing.
        line = "ligne pleine de contenu pertinent\n"
        text = line * 1000
        out = budget_injected_context(text, max_chars=500)
        body = out[: -len(_TRUNCATION_MARKER)]
        self.assertTrue(body.endswith("\n") or body.endswith("pertinent"))


class BuildMessagesBudgetTests(unittest.TestCase):
    """Real AIEngine (no mock): a giant injected context must not evict the system
    prompt or the conversation history — they have to survive within num_ctx."""

    def test_history_survives_a_giant_files_context(self):
        engine = AIEngine(model="qwen3:8b")
        history = [
            {"role": "user", "content": "Mon nom est Alex."},
            {"role": "assistant", "content": "Enchanté, Alex."},
            {"role": "user", "content": "Quel est mon nom ?"},
        ]
        giant = "--- CONTENU DE gros_fichier.txt ---\n" + ("bla " * 100_000)  # ~400k chars
        messages = engine._build_messages(history, files_context=giant)

        self.assertEqual(messages[0]["role"], "system")
        # The system block is bounded, far below the raw 400k-char injection…
        self.assertLess(len(messages[0]["content"]), _INJECTED_CONTEXT_BUDGET_CHARS + 20_000)
        self.assertIn(_TRUNCATION_MARKER.strip(), messages[0]["content"])
        # …and every conversation turn is preserved verbatim, in order.
        self.assertEqual(messages[1:], history)


class BuildMessagesImageTests(unittest.TestCase):
    """Attachments must reach Ollama as raw base64 on the right user message."""

    def test_persisted_image_is_carried_as_base64(self):
        engine = AIEngine(model="llava:7b")
        history = [
            {"role": "user", "content": "Décris", "images": ["data:image/png;base64,AAAA"]},
            {"role": "assistant", "content": "C'est un chat."},
            {"role": "user", "content": "Et la couleur ?"},
        ]
        messages = engine._build_messages(history)
        # The image stays on the turn it belongs to (the FIRST user message),
        # stripped of its data-URL prefix — and survives later turns.
        self.assertEqual(messages[1]["images"], ["AAAA"])
        self.assertNotIn("images", messages[3])

    def test_live_image_attaches_to_last_user(self):
        engine = AIEngine(model="llava:7b")
        history = [{"role": "user", "content": "Décris"}]
        messages = engine._build_messages(history, images=["data:image/jpeg;base64,BBBB"])
        self.assertEqual(messages[-1]["images"], ["BBBB"])


if __name__ == "__main__":
    unittest.main()
