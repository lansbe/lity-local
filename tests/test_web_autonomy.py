import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_loop import FakeFiles, ScriptedEngine, _collect_events

from lity.services.ai.agent import AgentLoop
from lity.services.web.fetch import (
    _crude_text,
    query_coverage,
    select_relevant,
)


def _no_embed(_text):
    return None  # simulates "no embedding model pulled" (the user's 404 case)


class LexicalRelevanceTests(unittest.TestCase):
    """select_relevant must surface on-topic passages WITHOUT embeddings — the
    old behaviour returned the page head (the menu), losing the answer."""

    def _doc(self):
        intro = "Menu accueil contact mentions légales " * 40  # boilerplate head
        answer = "Le score final du match était 4 à 2 pour les visiteurs. " * 3
        tail = "Articles connexes publicité abonnement " * 40
        return intro + answer + tail

    def test_lexical_fallback_finds_the_answer_passage(self):
        out = select_relevant(self._doc(), "score final du match", _no_embed, chunk_chars=200)
        self.assertIn("score final du match était 4 à 2", out)

    def test_dense_path_used_when_embeddings_available(self):
        # A degenerate embedder that scores by length still exercises the dense
        # path (returns a vector), proving embeddings are preferred when present.
        calls = {"n": 0}

        def embed(text):
            calls["n"] += 1
            return [float(len(text)), 1.0]

        out = select_relevant(self._doc(), "score", embed, chunk_chars=200)
        self.assertTrue(out)
        self.assertGreater(calls["n"], 1)  # embedder was actually used

    def test_no_query_returns_head(self):
        out = select_relevant("abc def ghi", "", _no_embed)
        self.assertEqual(out, "abc def ghi")


class QueryCoverageTests(unittest.TestCase):
    def test_full_and_zero_coverage(self):
        self.assertEqual(query_coverage("le chat noir dort", "chat noir"), 1.0)
        self.assertEqual(query_coverage("texte sans rapport aucun", "quantum hockey"), 0.0)

    def test_partial_coverage(self):
        cov = query_coverage("on parle du chat ici", "chat noir extraordinaire")
        self.assertTrue(0.0 < cov < 1.0)

    def test_short_words_are_ignored(self):
        self.assertEqual(query_coverage("rien", "de la le"), 1.0)  # no 3+ char terms


class CrudeExtractionTests(unittest.TestCase):
    def test_prefers_main_region_and_decodes_entities(self):
        html = (
            "<html><head><title>T</title></head><body>"
            "<nav>menu menu menu</nav>"
            "<main>" + ("Contenu principal pertinent. " * 20) + "Caf&eacute; &amp; th&eacute;."
            "</main>"
            "<footer>pied de page</footer></body></html>"
        )
        text = _crude_text(html)
        self.assertIn("Contenu principal pertinent", text)
        self.assertIn("Café & thé", text)  # entities decoded, not dropped
        self.assertNotIn("menu menu menu", text)  # boilerplate region skipped


class _WebFakeFiles(FakeFiles):
    working_dir = None


class _Fetcher:
    def __init__(self, pages):
        self._pages = pages

    def fetch(self, url, max_chars=6000):
        page = self._pages.get(url)
        if page is None:
            return {"ok": False, "url": url, "error": "introuvable"}
        return {"ok": True, "url": url, "title": page.get("title", url), "text": page["text"]}


def _web(pages):
    return {"searcher": None, "fetcher": _Fetcher(pages)}


class FetchDeadEndSignalTests(unittest.TestCase):
    def test_off_topic_page_tells_the_model_to_try_another_source(self):
        pages = {"http://x": {"text": "recettes de cuisine et jardinage", "title": "X"}}
        loop = AgentLoop(ScriptedEngine([]), _WebFakeFiles({}), allow_files=False, web=_web(pages))
        ok, out = loop._fetch_url({"url": "http://x", "query": "résultat match hockey"})
        self.assertTrue(ok)
        self.assertIn("AUTRE source", out)

    def test_on_topic_page_has_no_dead_end_note(self):
        pages = {"http://x": {"text": "le résultat du match hockey: 3-1", "title": "X"}}
        loop = AgentLoop(ScriptedEngine([]), _WebFakeFiles({}), allow_files=False, web=_web(pages))
        ok, out = loop._fetch_url({"url": "http://x", "query": "résultat match hockey"})
        self.assertTrue(ok)
        self.assertNotIn("AUTRE source", out)


class AnswerSufficiencyGateTests(unittest.TestCase):
    """The harness-enforced persistence: a hedge is pushed back so the model
    keeps researching instead of giving up after one source."""

    def _grader(self, verdicts):
        calls = []

        def grade(question, answer):
            calls.append((question, answer))
            return verdicts.pop(0) if verdicts else {"answered": True}

        return grade, calls

    def test_hedge_is_pushed_back_then_real_answer_accepted(self):
        engine = ScriptedEngine(
            [
                {
                    "content": "Je n'ai pas trouvé, consulte les sources officielles.",
                    "tool_calls": [],
                },
                {"content": "Les visiteurs ont gagné 4 à 2.", "tool_calls": []},
            ]
        )
        grader, calls = self._grader([{"answered": False}, {"answered": True}])
        loop = AgentLoop(
            engine,
            _WebFakeFiles({}),
            allow_files=False,
            web=_web({}),
            answer_grader=grader,
        )
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "qui a gagné le match ?"}], on_event)

        self.assertEqual(answer, "Les visiteurs ont gagné 4 à 2.")
        self.assertEqual(len(engine.calls), 2)  # hedge pushed back once, then answered
        self.assertGreaterEqual(len(calls), 1)
        nudged = any(
            "web_research" in str(m.get("content", ""))
            for call in engine.calls
            for m in call["messages"]
        )
        self.assertTrue(nudged)

    def test_gate_is_bounded_and_accepts_after_retries(self):
        hedge = {"content": "Je n'ai pas trouvé.", "tool_calls": []}
        engine = ScriptedEngine([hedge, hedge, hedge, hedge])
        grader, _calls = self._grader([{"answered": False}] * 9)
        loop = AgentLoop(
            engine, _WebFakeFiles({}), allow_files=False, web=_web({}), answer_grader=grader
        )
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "question impossible ?"}], on_event)

        self.assertIn("trouvé", answer)  # eventually accepts the hedge
        self.assertLessEqual(len(engine.calls), 3)  # 1 + at most 2 retries

    def test_no_grader_means_no_gate(self):
        engine = ScriptedEngine([{"content": "Je n'ai pas trouvé.", "tool_calls": []}])
        loop = AgentLoop(engine, _WebFakeFiles({}), allow_files=False, web=_web({}))
        events, on_event = _collect_events()
        answer = loop.run([{"role": "user", "content": "q ?"}], on_event)
        self.assertEqual(answer, "Je n'ai pas trouvé.")
        self.assertEqual(len(engine.calls), 1)

    def test_gate_off_without_web(self):
        # No web facade → the gate never fires even with a grader present.
        engine = ScriptedEngine([{"content": "Je n'ai pas trouvé.", "tool_calls": []}])
        grader, _calls = self._grader([{"answered": False}])
        loop = AgentLoop(engine, FakeFiles({}), answer_grader=grader)
        events, on_event = _collect_events()
        answer = loop.run([{"role": "user", "content": "q ?"}], on_event)
        self.assertEqual(answer, "Je n'ai pas trouvé.")


if __name__ == "__main__":
    unittest.main()
