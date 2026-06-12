import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app.controller import AgentController
from lity.app.services import AppServices
from lity.infrastructure.paths import AppPaths
from lity.interfaces.desktop_web.api import DesktopApi
from lity.services.ai.ollama_engine import AIEngine


class FakeOpenAIClient:
    def __init__(self):
        self.chat_calls = []
        self.stream_calls = []
        self.models = ["qwen2.5-coder-14b-instruct-mlx-4bit", "qwen3-8b-4bit-dwq"]

    def list_models(self):
        return list(self.models)

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if kwargs.get("tools"):
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
        return {
            "choices": [{"message": {"content": "Réponse locale."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }

    def stream_chat(self, **kwargs):
        self.stream_calls.append(kwargs)
        yield {"choices": [{"delta": {"content": "Bon"}}]}
        yield {"choices": [{"delta": {"content": "jour"}}], "usage": {"prompt_tokens": 5}}


class LmStudioEngineTests(unittest.TestCase):
    def test_lmstudio_chat_uses_openai_compatible_client(self):
        engine = AIEngine(model="qwen2.5-coder-14b-instruct-mlx-4bit")
        engine.chat_backend = "lmstudio"
        engine.openai_client = FakeOpenAIClient()

        answer = engine.get_response([{"role": "user", "content": "Salut"}])

        self.assertEqual(answer, "Réponse locale.")
        call = engine.openai_client.chat_calls[0]
        self.assertEqual(call["model"], "qwen2.5-coder-14b-instruct-mlx-4bit")
        self.assertEqual(call["messages"][-1]["content"], "Salut")
        self.assertEqual(engine.last_stats["context_used"], 14)

    def test_lmstudio_tool_calls_are_normalized(self):
        engine = AIEngine(model="qwen3-8b-4bit-dwq")
        engine.chat_backend = "lmstudio"
        engine.openai_client = FakeOpenAIClient()

        result = engine.chat_with_tools(
            [{"role": "user", "content": "Lis README"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )

        self.assertEqual(
            result["tool_calls"],
            [{"name": "read_file", "arguments": {"path": "README.md"}}],
        )

    def test_lmstudio_stream_and_model_listing_are_local(self):
        engine = AIEngine(model="qwen3-8b-4bit-dwq")
        engine.chat_backend = "lmstudio"
        engine.openai_client = FakeOpenAIClient()

        self.assertEqual(
            engine.get_installed_models(),
            ["qwen2.5-coder-14b-instruct-mlx-4bit", "qwen3-8b-4bit-dwq"],
        )
        text = "".join(engine.stream_response([{"role": "user", "content": "Bonjour"}]))

        self.assertEqual(text, "Bonjour")
        self.assertEqual(engine.openai_client.stream_calls[0]["model"], "qwen3-8b-4bit-dwq")


class LmStudioControllerTests(unittest.TestCase):
    def test_settings_switch_engine_to_lmstudio_without_becoming_cli_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            controller = AgentController(paths=paths, services=AppServices.create(paths))

            settings = controller.update_settings(
                {
                    "chat_provider": "lmstudio",
                    "lmstudio_base_url": "http://127.0.0.1:1234/v1",
                    "lmstudio_model": "qwen3-8b-4bit-dwq",
                }
            )

            self.assertEqual(settings["chat_provider"], "lmstudio")
            self.assertEqual(settings["lmstudio_base_url"], "http://127.0.0.1:1234/v1")
            self.assertEqual(settings["lmstudio_model"], "qwen3-8b-4bit-dwq")
            self.assertEqual(controller.engine.chat_backend, "lmstudio")
            self.assertEqual(controller.engine.model, "qwen3-8b-4bit-dwq")
            self.assertFalse(controller.using_cli_provider())

    def test_desktop_api_exposes_lmstudio_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            controller = AgentController(paths=paths, services=AppServices.create(paths))
            controller.update_settings(
                {
                    "chat_provider": "lmstudio",
                    "lmstudio_model": "qwen3-8b-4bit-dwq",
                }
            )
            controller.engine.openai_client = FakeOpenAIClient()
            api = DesktopApi(controller)

            catalog = api.lmstudio_models()

            self.assertTrue(catalog["ok"])
            self.assertEqual(catalog["default_model"], "qwen3-8b-4bit-dwq")
            self.assertEqual(catalog["models"][0]["slug"], "qwen2.5-coder-14b-instruct-mlx-4bit")


if __name__ == "__main__":
    unittest.main()
