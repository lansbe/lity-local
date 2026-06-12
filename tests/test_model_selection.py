import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_desktop_web_api import FakeEditor, FakeFiles, FakeRouter, FakeStreamingEngine

from lity.app._controller_models import ModelsMixin
from lity.app.controller import AgentController
from lity.app.services import AppServices
from lity.core.model_advisor import _flag_recommended, rank_models
from lity.core.model_catalog import is_embedding_model
from lity.infrastructure.paths import AppPaths
from lity.interfaces.desktop_web.api import DesktopApi
from lity.services.memory.json_memory import MemoryManager


def _row(name, params_b, toks, release, *, tools=True, grade="B", status="can-run", score=60):
    return {
        "name": name,
        "params_b": params_b,
        "tokens_per_sec": toks,
        "release_date": release,
        "tool_use": tools,
        "grade": grade,
        "status": status,
        "score": score,
        "kind": "chat",
        "verdict": "bon",
    }


class RecommendationRuleTests(unittest.TestCase):
    def _recommended(self, rows):
        _flag_recommended(rows)
        flagged = [row for row in rows if row.get("recommended")]
        self.assertEqual(len(flagged), 1)
        return flagged[0]["name"]

    def test_speed_floor_excludes_sluggish_bigger_model(self):
        # The 12B is more capable but too slow for an agent loop (<15 tok/s).
        rows = [
            _row("nemo:12b", 12, 12, "2024-07"),
            _row("qwen:8b", 8, 20, "2025-04"),
        ]
        self.assertEqual(self._recommended(rows), "qwen:8b")

    def test_newer_generation_wins_within_a_capability_tier(self):
        # 8B and 9B sit in the same ~4B tier: parameter count is a poor proxy
        # across generations, the newer model wins.
        rows = [
            _row("ancien:9b", 9, 18, "2024-06"),
            _row("recent:8b", 8, 17, "2025-04"),
        ]
        self.assertEqual(self._recommended(rows), "recent:8b")

    def test_higher_tier_beats_recency_when_fast_enough(self):
        rows = [
            _row("recent:8b", 8, 25, "2026-01"),
            _row("gros:14b", 14, 18, "2024-09"),
        ]
        self.assertEqual(self._recommended(rows), "gros:14b")

    def test_tool_capable_preferred_over_larger_toolless(self):
        rows = [
            _row("sans-outils:14b", 14, 20, "2025-01", tools=False),
            _row("avec-outils:8b", 8, 20, "2025-01", tools=True),
        ]
        self.assertEqual(self._recommended(rows), "avec-outils:8b")

    def test_floor_relaxes_when_nothing_is_fast(self):
        # Everything is slow: still recommend something rather than nothing.
        rows = [_row("seul:8b", 8, 9, "2025-01")]
        self.assertEqual(self._recommended(rows), "seul:8b")

    def test_no_floor_when_speed_unknown(self):
        rows = [_row("inconnu:8b", 8, None, "2025-01")]
        self.assertEqual(self._recommended(rows), "inconnu:8b")

    def test_real_ranking_recommends_a_capable_fast_tool_model(self):
        hardware = {
            "accelerator": "metal",
            "ram_gb": 16.0,
            "budget_gb": 11.2,
            "memory_bandwidth": 120.0,
        }
        rows = rank_models(hardware)
        recommended = [row for row in rows if row.get("recommended")]
        self.assertEqual(len(recommended), 1)
        best = recommended[0]
        self.assertTrue(best["tool_use"])  # agent-centric default
        self.assertGreaterEqual(best["tokens_per_sec"], 15)  # usable in a loop
        self.assertGreaterEqual(best["params_b"], 7)  # not a toy model
        self.assertIn(best["grade"], ("S", "A", "B"))


class EmbeddingDetectionTests(unittest.TestCase):
    def test_catalog_embeddings_are_detected(self):
        self.assertTrue(is_embedding_model("bge-m3"))
        self.assertTrue(is_embedding_model("nomic-embed-text"))

    def test_common_community_embeddings_are_detected_by_name(self):
        for name in ("snowflake-arctic-embed:latest", "mxbai-embed-large", "all-minilm:l6-v2"):
            self.assertTrue(is_embedding_model(name), name)

    def test_chat_models_are_not_flagged(self):
        for name in ("qwen3:8b", "llama3.1:8b", "mistral:7b", "fake-model"):
            self.assertFalse(is_embedding_model(name), name)


class _EngineWithModels(FakeStreamingEngine):
    def __init__(self, models):
        super().__init__()
        self._models = list(models)

    def get_installed_models(self):
        return list(self._models)


def _build_api(tmp, models):
    paths = AppPaths.create(home_override=Path(tmp))
    services = AppServices(
        settings=None,
        engine=_EngineWithModels(models),
        memory=MemoryManager(paths=paths),
        files=FakeFiles(),
        router=FakeRouter(),
        editor=FakeEditor(),
        image_manager=None,
    )
    controller = AgentController(paths=paths, services=services)
    return DesktopApi(controller, emit=lambda event, payload: None)


class ChatSelectorTests(unittest.TestCase):
    def test_embeddings_are_hidden_from_the_chat_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = _build_api(tmp, ["fake-model", "bge-m3", "nomic-embed-text:latest"])
            result = api.list_models()
            self.assertEqual(result["models"], ["fake-model"])
            self.assertEqual(result["selected"], "fake-model")

    def test_only_embeddings_installed_yields_empty_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = _build_api(tmp, ["bge-m3"])
            result = api.list_models()
            self.assertEqual(result["models"], [])
            self.assertEqual(result["selected"], "")  # never the default name

    def test_nothing_installed_yields_empty_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = _build_api(tmp, [])
            result = api.list_models()
            self.assertEqual(result["models"], [])
            self.assertEqual(result["selected"], "")
            self.assertIsNone(result["error"])


class ModelRecommendationApiTests(unittest.TestCase):
    def test_recommendations_include_image_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = _build_api(tmp, [])
            result = api.model_recommendations()

            self.assertIn("image_models", result)
            self.assertTrue(any(row["kind"] == "image" for row in result["image_models"]))
            self.assertTrue(
                any(row["backend"] == "automatic1111" for row in result["image_models"])
            )


class LiveToolCapabilityTests(unittest.TestCase):
    def test_live_probe_overrides_prior_for_installed_models(self):
        # Ollama says this gemma build CAN tool-call — live wins over the prior.
        stub = SimpleNamespace(model_supports_tools=lambda name: name == "gemma-custom:9b")
        rows = [
            {"name": "gemma-custom:9b", "installed": True, "kind": "chat", "tool_use": False},
            {"name": "qwen3:8b", "installed": False, "kind": "chat", "tool_use": True},
            {"name": "bge-m3", "installed": True, "kind": "embed", "tool_use": False},
        ]
        ModelsMixin._apply_live_tool_capability(stub, rows)
        self.assertTrue(rows[0]["tool_use"])  # overridden by the live probe
        self.assertTrue(rows[1]["tool_use"])  # not installed → prior untouched
        self.assertFalse(rows[2]["tool_use"])  # embeddings never tool-call

    def test_unknown_live_capability_keeps_the_prior(self):
        stub = SimpleNamespace(model_supports_tools=lambda name: None)
        rows = [{"name": "qwen3:8b", "installed": True, "kind": "chat", "tool_use": True}]
        ModelsMixin._apply_live_tool_capability(stub, rows)
        self.assertTrue(rows[0]["tool_use"])


if __name__ == "__main__":
    unittest.main()
