import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app._controller_retrieval import RetrievalMixin
from lity.services.rag.contextualize import contextualize_query


class ContextualizeQueryTests(unittest.TestCase):
    def test_passthrough_without_generator(self):
        history = [{"role": "user", "content": "x"}]
        self.assertEqual(contextualize_query(None, history, "et le deuxième ?"), "et le deuxième ?")

    def test_blank_query_returns_empty(self):
        self.assertEqual(contextualize_query(lambda p, s: {"standalone": "X"}, [], "   "), "")

    def test_no_prior_history_skips_llm_and_returns_query(self):
        called = []

        def gen(prompt, schema):
            called.append(1)
            return {"standalone": "NE DOIT PAS ÊTRE UTILISÉ"}

        # Only the current user message → nothing prior to disambiguate against.
        out = contextualize_query(gen, [{"role": "user", "content": "bonjour"}], "bonjour")
        self.assertEqual(out, "bonjour")
        self.assertEqual(called, [])  # no LLM call paid when already standalone

    def test_reformulates_followup(self):
        history = [
            {"role": "user", "content": "parle-moi du cache de prompt"},
            {"role": "assistant", "content": "avantages : coût, latence, cohérence"},
            {"role": "user", "content": "et le deuxième ?"},
        ]
        out = contextualize_query(
            lambda p, s: {"standalone": "quel est le deuxième avantage du cache de prompt"},
            history,
            "et le deuxième ?",
        )
        self.assertEqual(out, "quel est le deuxième avantage du cache de prompt")

    def test_current_turn_not_duplicated_in_history(self):
        seen = {}

        def gen(prompt, schema):
            seen["prompt"] = prompt
            return {"standalone": "ok"}

        history = [
            {"role": "user", "content": "sujet initial"},
            {"role": "assistant", "content": "réponse"},
            {"role": "user", "content": "et ensuite ?"},
        ]
        contextualize_query(gen, history, "et ensuite ?")
        # "et ensuite ?" must appear once (as NOUVELLE DEMANDE), not echoed in history.
        self.assertEqual(seen["prompt"].count("et ensuite ?"), 1)

    def test_exception_falls_back_to_query(self):
        def boom(prompt, schema):
            raise RuntimeError("model down")

        history = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "ça"}]
        self.assertEqual(contextualize_query(boom, history, "ça"), "ça")

    def test_none_or_non_dict_falls_back(self):
        history = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]
        self.assertEqual(contextualize_query(lambda p, s: None, history, "q"), "q")
        self.assertEqual(contextualize_query(lambda p, s: "nope", history, "q"), "q")

    def test_blank_rewrite_falls_back(self):
        history = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]
        self.assertEqual(contextualize_query(lambda p, s: {"standalone": "   "}, history, "q"), "q")

    def test_oversized_rewrite_falls_back(self):
        history = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]
        big = "x" * 600
        self.assertEqual(contextualize_query(lambda p, s: {"standalone": big}, history, "q"), "q")

    def test_strips_surrounding_quotes(self):
        history = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]
        out = contextualize_query(lambda p, s: {"standalone": '"requête propre"'}, history, "q")
        self.assertEqual(out, "requête propre")

    def test_only_recent_turns_are_used(self):
        seen = {}

        def gen(prompt, schema):
            seen["prompt"] = prompt
            return {"standalone": "ok"}

        history = [{"role": "user", "content": f"vieux message {i}"} for i in range(10)]
        history.append({"role": "user", "content": "question actuelle"})
        contextualize_query(gen, history, "question actuelle", max_turns=4)
        self.assertNotIn("vieux message 0", seen["prompt"])  # trimmed
        self.assertIn("vieux message", seen["prompt"])  # recent ones kept


class PassiveRetrievalQueryTests(unittest.TestCase):
    """The controller-side helper that wires history-aware rewriting into both
    project RAG and cross-session memory, with per-turn memoization + gating."""

    def _stub(self, *, context, settings=None, engine=None):
        memory = SimpleNamespace(get_context=lambda: context, active_conversation_id="c1")
        return SimpleNamespace(
            memory=memory,
            settings=settings,
            engine=engine if engine is not None else SimpleNamespace(),
        )

    def test_no_user_message_returns_empty(self):
        stub = self._stub(context=[{"role": "assistant", "content": "bonjour"}])
        self.assertEqual(RetrievalMixin._passive_retrieval_query(stub), "")

    def test_engine_without_structured_returns_last_user(self):
        stub = self._stub(context=[{"role": "user", "content": "et le deuxième ?"}])
        self.assertEqual(RetrievalMixin._passive_retrieval_query(stub), "et le deuxième ?")

    def test_setting_off_returns_last_user(self):
        engine = SimpleNamespace(
            generate_structured=lambda p, s, think=False, **kw: {"standalone": "REFORMULÉ"}
        )
        stub = self._stub(
            context=[
                {"role": "assistant", "content": "…"},
                {"role": "user", "content": "et le deuxième ?"},
            ],
            settings={"history_aware_retrieval": False},
            engine=engine,
        )
        self.assertEqual(RetrievalMixin._passive_retrieval_query(stub), "et le deuxième ?")

    def test_reformulates_with_engine(self):
        engine = SimpleNamespace(
            generate_structured=lambda p, s, think=False, **kw: {
                "standalone": "deuxième avantage du cache de prompt"
            }
        )
        stub = self._stub(
            context=[
                {"role": "user", "content": "avantages du cache de prompt ?"},
                {"role": "assistant", "content": "1. coût 2. latence"},
                {"role": "user", "content": "et le deuxième ?"},
            ],
            engine=engine,
        )
        self.assertEqual(
            RetrievalMixin._passive_retrieval_query(stub), "deuxième avantage du cache de prompt"
        )

    def test_result_is_memoized_within_turn(self):
        calls = []

        def gen(prompt, schema, think=False, **kw):
            calls.append(1)
            return {"standalone": "X"}

        engine = SimpleNamespace(generate_structured=gen)
        stub = self._stub(
            context=[
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "ça ?"},
            ],
            engine=engine,
        )
        first = RetrievalMixin._passive_retrieval_query(stub)
        second = RetrievalMixin._passive_retrieval_query(stub)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)  # second served from cache, no extra LLM call


if __name__ == "__main__":
    unittest.main()
