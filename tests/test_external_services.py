import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.infrastructure.paths import AppPaths
from lity.services.ai.ollama_engine import AIEngine
from lity.services.image_generation.stable_diffusion import StableDiffusionService


class ExternalServiceTests(unittest.TestCase):
    def test_ai_engine_health_reports_installed_models(self):
        from lity.services.external import ServiceHealth

        engine = AIEngine(model="llama3")
        engine.get_installed_models = lambda: ["llama3"]

        health = engine.check_health()

        self.assertIsInstance(health, ServiceHealth)
        self.assertTrue(health.ok)
        self.assertIn("llama3", health.detail)

    def test_ai_engine_stream_response_uses_ollama_stream_and_keep_alive(self):
        calls = []

        class Message:
            def __init__(self, content):
                self.content = content
                self.thinking = ""

        class Chunk:
            def __init__(self, content):
                self.message = Message(content)

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return [Chunk("Bon"), Chunk("jour")]

        fake_ollama = types.SimpleNamespace(chat=fake_chat)
        engine = AIEngine(model="llama3")

        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            chunks = list(
                engine.stream_response(
                    [{"role": "user", "content": "salut"}],
                    assistant_name="Assistant",
                    keep_alive="15m",
                )
            )

        self.assertEqual(chunks, ["Bon", "jour"])
        self.assertEqual(calls[0]["model"], "llama3")
        self.assertTrue(calls[0]["stream"])
        self.assertEqual(calls[0]["keep_alive"], "15m")

    def test_stable_diffusion_health_fails_with_structured_result(self):
        from lity.services.external import ServiceHealth

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            service = StableDiffusionService(paths, api_url="http://127.0.0.1:9")

            health = service.check_health()

            self.assertIsInstance(health, ServiceHealth)
            self.assertFalse(health.ok)
            self.assertIn("Stable Diffusion", health.name)

    def test_stable_diffusion_get_json_uses_requested_endpoint_without_preflight(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps([{"name": "Euler a"}]).encode("utf-8")

        calls = []

        def fake_urlopen(url, timeout):
            calls.append(str(url))
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            service = StableDiffusionService(paths, api_url="http://sd.local")

            with patch("urllib.request.urlopen", fake_urlopen):
                data = service._get_json("/sdapi/v1/samplers", timeout=2.0)

            self.assertEqual(data, [{"name": "Euler a"}])
            self.assertEqual(calls, ["http://sd.local/sdapi/v1/samplers"])


if __name__ == "__main__":
    unittest.main()
