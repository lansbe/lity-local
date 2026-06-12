import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app.controller import AgentController
from lity.app.services import AppServices
from lity.infrastructure.paths import AppPaths
from lity.services.ai.agent import AgentLoop
from lity.services.memory.json_memory import MemoryManager
from lity.services.web.fetch import PageFetcher, _crude_text, select_relevant
from lity.services.web.http_util import get_text
from lity.services.web.research import WebResearcher
from lity.services.web.router import WEB_NUDGE, looks_time_sensitive
from lity.services.web.search import SearxngProvider, WebSearcher


class FakeProvider:
    def __init__(self, name, results, available=True, raise_exc=False):
        self.name = name
        self._results = results
        self._available = available
        self._raise = raise_exc

    def available(self):
        return self._available

    def search(self, query, max_results):
        if self._raise:
            raise RuntimeError("boom")
        return list(self._results)[:max_results]


class WebSearcherTests(unittest.TestCase):
    def test_empty_query_is_not_ok(self):
        self.assertFalse(WebSearcher([]).search("  ")["ok"])

    def test_first_nonempty_provider_wins(self):
        chain = WebSearcher(
            [
                FakeProvider("searxng", []),  # configured but nothing this time
                FakeProvider("ddg", [{"title": "T", "url": "http://x", "snippet": "s"}]),
            ]
        )
        out = chain.search("python")
        self.assertTrue(out["ok"])
        self.assertEqual(out["provider"], "ddg")
        self.assertEqual(out["results"][0]["url"], "http://x")

    def test_skips_unavailable_and_errored_providers(self):
        chain = WebSearcher(
            [
                FakeProvider("searxng", [], available=False),
                FakeProvider("ddg", [], raise_exc=True),
                FakeProvider("wiki", [{"title": "W", "url": "http://w", "snippet": ""}]),
            ]
        )
        self.assertEqual(chain.search("q")["provider"], "wiki")

    def test_all_empty_returns_not_ok(self):
        out = WebSearcher([FakeProvider("a", [])]).search("q")
        self.assertFalse(out["ok"])
        self.assertEqual(out["results"], [])

    def test_search_results_are_cached_per_session(self):
        provider = FakeProvider("ddg", [{"title": "T", "url": "http://x", "snippet": "s"}])
        provider.calls = 0

        def search(query, max_results):
            provider.calls += 1
            return list(provider._results)[:max_results]

        provider.search = search
        chain = WebSearcher([provider])

        first = chain.search("python")
        second = chain.search("python")

        self.assertTrue(first["ok"])
        self.assertEqual(second["results"][0]["url"], "http://x")
        self.assertEqual(provider.calls, 1)


class HttpUtilTests(unittest.TestCase):
    def test_httpx_failure_does_not_pay_a_second_urllib_timeout(self):
        with (
            mock.patch("httpx.get", side_effect=RuntimeError("timeout")),
            mock.patch("urllib.request.urlopen") as urlopen,
        ):
            self.assertIsNone(get_text("https://slow.example", timeout=0.1))

        urlopen.assert_not_called()


class SearxngProviderTests(unittest.TestCase):
    def test_parses_and_skips_urlless_results(self):
        payload = {
            "results": [
                {"title": "Py", "url": "http://py", "content": "lang"},
                {"title": "No url", "url": "", "content": "x"},
            ]
        }
        with mock.patch("lity.services.web.search.get_json", return_value=payload):
            results = SearxngProvider("http://searx").search("python", 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "http://py")
        self.assertEqual(results[0]["snippet"], "lang")

    def test_unavailable_without_url(self):
        self.assertFalse(SearxngProvider("").available())
        self.assertTrue(SearxngProvider("http://x").available())


class FetchTests(unittest.TestCase):
    def test_invalid_url_is_not_ok(self):
        self.assertFalse(PageFetcher().fetch("not-a-url")["ok"])

    def test_crude_text_strips_scripts_and_tags(self):
        html = (
            "<html><head><title>T</title><script>var x=1</script></head>"
            "<body><p>Hello world</p></body></html>"
        )
        text = _crude_text(html)
        self.assertIn("Hello world", text)
        self.assertNotIn("var x", text)

    def test_fetch_extracts_and_caches(self):
        html = (
            "<html><head><title>Doc</title></head><body><article>"
            "<p>Bonjour le monde, ceci est un texte de test.</p></article></body></html>"
        )
        fetcher = PageFetcher()
        with mock.patch("lity.services.web.fetch.get_text", return_value=html) as get_text:
            first = fetcher.fetch("https://example.com")
            fetcher.fetch("https://example.com")  # served from cache
        self.assertTrue(first["ok"])
        self.assertIn("Bonjour", first["text"])
        self.assertEqual(get_text.call_count, 1)

    def test_fetch_network_failure_is_not_ok(self):
        with mock.patch("lity.services.web.fetch.get_text", return_value=None):
            self.assertFalse(PageFetcher().fetch("https://example.com")["ok"])

    def test_select_relevant_picks_matching_chunk(self):
        def embed(text):
            lowered = text.lower()
            return [1.0 if "python" in lowered else 0.0, 1.0 if "banane" in lowered else 0.0]

        text = ("python " * 120) + ("banane " * 120)
        self.assertIn("python", select_relevant(text, "python", embed, top_k=1, chunk_chars=200))
        self.assertIn("banane", select_relevant(text, "banane", embed, top_k=1, chunk_chars=200))

    def test_select_relevant_falls_back_without_embedder(self):
        self.assertEqual(select_relevant("abc def", "q", lambda _t: None), "abc def")

    def test_select_relevant_no_query_returns_head(self):
        self.assertEqual(select_relevant("abc", "", lambda _t: [1.0]), "abc")


class WebResearcherTests(unittest.TestCase):
    def test_research_fetches_top_sources_in_parallel(self):
        class Searcher:
            def search(self, query, max_results=6):
                return {
                    "ok": True,
                    "provider": "fake",
                    "results": [
                        {"title": "A", "url": "http://a", "snippet": "sa"},
                        {"title": "B", "url": "http://b", "snippet": "sb"},
                        {"title": "C", "url": "http://c", "snippet": "sc"},
                    ],
                }

        class Fetcher:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def fetch(self, url, max_chars=6000):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.05)
                    return {
                        "ok": True,
                        "url": url,
                        "title": url.rsplit("/", 1)[-1].upper(),
                        "text": f"intro {url} résultat précis météo 2026 " * 20,
                    }
                finally:
                    with self.lock:
                        self.active -= 1

        fetcher = Fetcher()
        researcher = WebResearcher(Searcher(), fetcher, fetch_limit=3, max_workers=3)

        out = researcher.research("météo 2026")

        self.assertTrue(out["ok"])
        self.assertEqual(len(out["sources"]), 3)
        self.assertGreaterEqual(fetcher.max_active, 2)
        self.assertIn("résultat précis", out["sources"][0]["text"])

    def test_research_keeps_successful_sources_when_one_fetch_fails(self):
        class Searcher:
            def search(self, query, max_results=6):
                return {
                    "ok": True,
                    "provider": "fake",
                    "results": [
                        {"title": "A", "url": "http://a", "snippet": ""},
                        {"title": "B", "url": "http://b", "snippet": ""},
                    ],
                }

        class Fetcher:
            def fetch(self, url, max_chars=6000):
                if url.endswith("a"):
                    return {"ok": False, "url": url, "error": "blocked"}
                return {"ok": True, "url": url, "title": "B", "text": "réponse concrète " * 50}

        out = WebResearcher(Searcher(), Fetcher(), fetch_limit=2).research("réponse concrète")

        self.assertTrue(out["ok"])
        self.assertEqual([source["url"] for source in out["sources"]], ["http://b"])
        self.assertEqual(out["failed"][0]["url"], "http://a")


class _ScriptedEngine:
    def __init__(self, script):
        self.script = list(script)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        return self.script.pop(0) if self.script else {"content": "fin", "tool_calls": []}


class _FakeFiles:
    working_dir = None

    def get_available_files(self, recursive=False):
        return []

    def read_file_safe(self, path, max_chars=20_000):
        return False, "introuvable"


def _fake_web(results=None, page=None):
    class Searcher:
        def search(self, query, max_results=6):
            return {
                "ok": True,
                "provider": "fake",
                "results": results or [{"title": "T", "url": "http://x", "snippet": "snip"}],
            }

    class Fetcher:
        def fetch(self, url, max_chars=6000):
            return page or {"ok": True, "url": url, "title": "Doc", "text": "contenu de la page"}

    return {"searcher": Searcher(), "fetcher": Fetcher()}


class AgentWebToolsTests(unittest.TestCase):
    def test_web_tools_advertised_only_with_web(self):
        with_web = AgentLoop(_ScriptedEngine([]), _FakeFiles(), web=_fake_web())
        names = [tool["function"]["name"] for tool in with_web.tool_specs()]
        self.assertIn("web_research", names)
        self.assertIn("web_search", names)
        self.assertIn("fetch_url", names)

        without = AgentLoop(_ScriptedEngine([]), _FakeFiles())
        names = [tool["function"]["name"] for tool in without.tool_specs()]
        self.assertNotIn("web_research", names)
        self.assertNotIn("web_search", names)
        self.assertNotIn("fetch_url", names)

    def test_web_research_returns_read_sources_in_one_tool_call(self):
        class Researcher:
            def research(self, query, **_kwargs):
                return {
                    "ok": True,
                    "provider": "fake",
                    "query": query,
                    "sources": [
                        {
                            "title": "Source A",
                            "url": "http://a",
                            "snippet": "résumé",
                            "text": "passage pertinent lu",
                        }
                    ],
                    "failed": [],
                    "searched": 1,
                    "fetched": 1,
                }

        loop = AgentLoop(
            _ScriptedEngine([]),
            _FakeFiles(),
            web={"searcher": None, "fetcher": None, "researcher": Researcher()},
        )

        ok, out = loop._web_research({"query": "question"})

        self.assertTrue(ok)
        self.assertIn("Source A", out)
        self.assertIn("passage pertinent lu", out)

    def test_web_search_then_final_answer(self):
        engine = _ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "web_search", "arguments": {"query": "x"}}],
                },
                {"content": "Réponse avec sources.", "tool_calls": []},
            ]
        )
        events = []
        loop = AgentLoop(engine, _FakeFiles(), web=_fake_web())
        answer = loop.run(
            [{"role": "user", "content": "cherche"}],
            lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual(answer, "Réponse avec sources.")
        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertTrue(result["ok"])
        self.assertIn("http://x", result["summary"])

    def test_fetch_url_returns_clean_text(self):
        engine = _ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "fetch_url", "arguments": {"url": "http://x"}}],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        events = []
        loop = AgentLoop(engine, _FakeFiles(), web=_fake_web())
        loop.run(
            [{"role": "user", "content": "lis"}],
            lambda kind, payload: events.append((kind, payload)),
        )
        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertTrue(result["ok"])
        self.assertIn("contenu", result["summary"])


class _RecordingEngine:
    model = "fake"

    def __init__(self):
        self.captured = {}

    def get_installed_models(self):
        return ["fake"]

    def extract_fact(self, message):
        return None

    def _build_messages(self, context, **kwargs):
        return [{"role": "system", "content": "sys"}] + [
            {"role": m["role"], "content": m["content"]} for m in context
        ]

    def chat_with_tools(self, messages, tools=None, **kwargs):
        self.captured["system"] = messages[0]["content"]
        self.captured["tools"] = [t["function"]["name"] for t in (tools or [])]
        return {"content": "réponse finale", "tool_calls": []}


class _Files:
    working_dir = None
    loaded_files: dict = {}

    def get_context_for_ai(self):
        return ""


class _Router:
    def process_intent(self, *args):
        return {"handled": False, "action": "none", "message": "", "system_context": ""}


class _Editor:
    def parse_create_blocks(self, text):
        return []

    def parse_search_replace_blocks(self, text):
        return []


class WebRouterTests(unittest.TestCase):
    def test_time_sensitive_queries(self):
        for query in [
            "quel est le dernier iPhone sorti ?",
            "score du match d'hier",
            "prix actuel du bitcoin",
            "qui a gagné la finale en 2026",
            "la météo aujourd'hui",
        ]:
            with self.subTest(query=query):
                self.assertTrue(looks_time_sensitive(query))

    def test_evergreen_queries(self):
        for query in [
            "explique-moi la photosynthèse",
            "écris une fonction qui trie une liste",
            "qu'est-ce qu'un trou noir ?",
            "",
        ]:
            with self.subTest(query=query):
                self.assertFalse(looks_time_sensitive(query))


class AgentWebWiringTests(unittest.TestCase):
    def _capture_system(self, user_input: str) -> str:
        engine = _RecordingEngine()
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=_Files(),
                router=_Router(),
                editor=_Editor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            controller.process_user_message_agent(
                user_input, on_event=lambda kind, payload: None, allow_web=True
            )
        return engine.captured["system"]

    def test_time_sensitive_query_adds_web_nudge(self):
        system = self._capture_system("quel est le dernier iPhone sorti ?")
        self.assertIn("RECHERCHE WEB", system)  # web method still injected
        self.assertIn(WEB_NUDGE, system)  # plus the time-sensitive steer

    def test_evergreen_query_keeps_web_without_nudge(self):
        system = self._capture_system("explique-moi la photosynthèse en détail")
        self.assertIn("RECHERCHE WEB", system)  # web stays available (model decides)
        self.assertNotIn(WEB_NUDGE, system)  # but no steer on an evergreen question

    def test_allow_web_injects_date_guidance_and_tools(self):
        engine = _RecordingEngine()
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=_Files(),
                router=_Router(),
                editor=_Editor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            controller.process_user_message_agent(
                "score du dernier match", on_event=lambda kind, payload: None, allow_web=True
            )

        system = engine.captured["system"]
        self.assertIn("RECHERCHE WEB", system)  # web method injected
        self.assertIn("Date du jour", system)  # today's date for recency
        self.assertIn("web_search", engine.captured["tools"])
        self.assertIn("fetch_url", engine.captured["tools"])


class FetchPageContextTests(unittest.TestCase):
    def test_fetch_page_returns_clean_text(self):
        html = (
            "<html><head><title>Doc</title></head><body><article>"
            "<p>Bonjour, ceci est le contenu de la page de test.</p></article></body></html>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=_RecordingEngine(),
                memory=MemoryManager(paths=paths),
                files=_Files(),
                router=_Router(),
                editor=_Editor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            with mock.patch("lity.services.web.fetch.get_text", return_value=html):
                result = controller.fetch_page("https://example.com")
        self.assertTrue(result["ok"])
        self.assertIn("Bonjour", result["text"])


if __name__ == "__main__":
    unittest.main()
