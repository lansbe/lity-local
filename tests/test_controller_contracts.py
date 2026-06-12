import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app.controller import AgentController
from lity.infrastructure.paths import AppPaths


class FakeEngine:
    model = "fake-model"

    def __init__(self):
        self.extract_fact_calls = []

    def get_installed_models(self):
        return ["fake-model"]

    def get_response(self, *args, **kwargs):
        return "Réponse fake super utile."

    def extract_fact(self, message):
        self.extract_fact_calls.append(message)
        return None


class FakeMemory:
    assistant_profile = {"nom": "Assistant"}

    def __init__(self):
        self.messages = []

    def add_message(self, role, content, images=None):
        self.messages.append((role, content, images))

    def get_context(self):
        return [
            {"role": role, "content": content, **({"images": images} if images else {})}
            for role, content, images in self.messages
        ]

    def get_user_info_summary(self):
        return ""

    def get_assistant_info_summary(self):
        return ""

    def clear(self):
        self.messages.clear()

    def process_extracted_fact(self, fact):
        return None


class FakeFiles:
    loaded_files = {}
    working_dir = None
    current_file_path = None

    def get_context_for_ai(self):
        return ""

    def set_working_dir(self, path):
        self.working_dir = Path(path)
        return True, f"Répertoire de travail défini sur : {path}"

    def load_file(self, path, user_input=None):
        return True, f"Fichier chargé : {path}"

    def list_files(self):
        return "Aucun fichier"

    def close_file(self, target=None):
        return True, "Fichier fermé."


class FakeRouter:
    model = "fake-model"

    def process_intent(self, user_input, file_manager):
        return {"handled": False, "action": "none", "message": "", "system_context": ""}


class FakeEditor:
    def parse_create_blocks(self, text):
        return []

    def parse_search_replace_blocks(self, text):
        return []


class FakeImageManager:
    def is_active(self):
        return False

    def start_session(self):
        return None


class ControllerContractTests(unittest.TestCase):
    def test_controller_accepts_injected_services_and_returns_typed_result(self):
        from lity.app.results import AiResponseResult
        from lity.app.services import AppServices

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=FakeMemory(),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=FakeImageManager(),
            )
            controller = AgentController(paths=paths, services=services)

            result = controller.process_user_message_sync("bonjour")

            self.assertIsInstance(result, AiResponseResult)
            self.assertEqual(result.to_dict()["type"], "ai_response")
            self.assertEqual(result.content, "Réponse fake super utile.")

    def test_controller_does_not_block_plain_chat_with_broad_file_words(self):
        from lity.app.results import AiResponseResult
        from lity.app.services import AppServices

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=FakeMemory(),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=FakeImageManager(),
            )
            controller = AgentController(paths=paths, services=services)

            for message in ["qu'est-ce qu'un code postal ?", "lis-moi une histoire"]:
                with self.subTest(message=message):
                    result = controller.process_user_message_sync(message)

                    self.assertIsInstance(result, AiResponseResult)
                    self.assertEqual(result.content, "Réponse fake super utile.")

    def test_controller_still_blocks_explicit_file_requests_without_file_context(self):
        from lity.app.results import ErrorResult
        from lity.app.services import AppServices

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=FakeMemory(),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=FakeImageManager(),
            )
            controller = AgentController(paths=paths, services=services)

            result = controller.process_user_message_sync("analyse ce fichier")

            self.assertIsInstance(result, ErrorResult)
            self.assertIn("Aucun fichier", result.message)

    def test_controller_streaming_does_not_block_plain_chat_with_broad_file_words(self):
        from lity.app.results import AiResponseResult
        from lity.app.services import AppServices

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=FakeMemory(),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=FakeImageManager(),
            )
            controller = AgentController(paths=paths, services=services)
            chunks = []

            result = controller.process_user_message_stream(
                "qu'est-ce qu'un code postal ?",
                chunks.append,
            )

            self.assertIsInstance(result, AiResponseResult)
            self.assertEqual(result.content, "Réponse fake super utile.")
            self.assertEqual(chunks, [])

    def test_controller_does_not_create_image_manager_until_image_mode_is_used(self):
        from lity.app.services import AppServices

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=FakeMemory(),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )

            controller = AgentController(paths=paths, services=services)

            self.assertIsNone(controller.services.image_manager)
            self.assertIsNone(controller.image_manager)

    def test_controller_skips_background_fact_extraction_for_small_talk(self):
        from lity.app.controller import should_extract_fact

        self.assertFalse(should_extract_fact("salut"))
        self.assertFalse(should_extract_fact("ok merci"))
        self.assertTrue(should_extract_fact("mon nom est Alex"))
        self.assertTrue(should_extract_fact("je préfère les réponses courtes"))

    def test_image_generation_persists_into_active_conversation(self):
        """A successful render is recorded in the active conversation so it
        shows up in the sidebar and the image redisplays on reload — with no
        caption text on the assistant message."""
        from lity.app.services import AppServices
        from lity.services.memory.json_memory import MemoryManager

        class GeneratingImageManager:
            def __init__(self, image_path):
                self._image_path = image_path

            def is_active(self):
                return True

            def process_user_message(self, user_input, engine):
                return {
                    "type": "image_generation_result",
                    "content": {
                        "status": "success",
                        "params": {"prompt": user_input},
                        "image_path": str(self._image_path),
                    },
                    "message": "Image générée en local avec sd-turbo.",
                }

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            png = Path(tmp) / "dog.png"
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=MemoryManager(paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=GeneratingImageManager(png),
            )
            controller = AgentController(paths=paths, services=services)

            result = controller.process_user_message_sync("Genère un chien")

            self.assertEqual(result["type"], "image_generation_result")
            # The empty draft is now a real, listed conversation.
            conversations = controller.list_conversations()
            self.assertEqual(len(conversations), 1)
            self.assertEqual(conversations[0]["message_count"], 2)
            messages = controller.get_messages()
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[0]["content"], "Genère un chien")
            self.assertEqual(messages[1]["role"], "assistant")
            self.assertEqual(messages[1]["content"], "")  # no caption noise
            self.assertTrue(messages[1]["image"].startswith("data:image/png;base64,"))

    def test_controller_does_not_create_video_manager_until_video_mode_is_used(self):
        from lity.app.services import AppServices

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=FakeMemory(),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
                video_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            self.assertIsNone(controller.services.video_manager)
            self.assertIsNone(controller.video_manager)
            self.assertFalse(controller.is_video_session_active())

    def test_video_generation_persists_into_active_conversation(self):
        """A successful clip is recorded in the active conversation (empty
        assistant text + the video attached) so it redisplays on reload."""
        from lity.app.services import AppServices
        from lity.services.memory.json_memory import MemoryManager

        class GeneratingVideoManager:
            def __init__(self, video_path):
                self._video_path = video_path

            def is_active(self):
                return True

            def process_user_message(self, user_input, engine):
                return {
                    "type": "video_generation_result",
                    "content": {
                        "status": "success",
                        "params": {"prompt": user_input},
                        "video_path": str(self._video_path),
                    },
                    "message": "Vidéo générée en local avec wan21-t2v-1.3b.",
                }

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            mp4 = Path(tmp) / "clip.mp4"
            mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
            services = AppServices(
                settings=None,
                engine=FakeEngine(),
                memory=MemoryManager(paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
                video_manager=GeneratingVideoManager(mp4),
            )
            controller = AgentController(paths=paths, services=services)

            result = controller.process_user_message_sync("Génère un chat qui court")

            self.assertEqual(result["type"], "video_generation_result")
            messages = controller.get_messages()
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[1]["role"], "assistant")
            self.assertEqual(messages[1]["content"], "")  # no caption noise
            self.assertTrue(messages[1]["video"].startswith("data:video/mp4;base64,"))

    def test_controller_syncs_to_first_installed_model_when_saved_model_is_missing(self):
        from lity.app.services import AppServices
        from lity.infrastructure.settings import SettingsStore
        from lity.services.ai.ollama_engine import AIEngine

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("selected_model", "llama3")
            router = FakeRouter()
            services = AppServices(
                settings=settings,
                engine=AIEngine(model="llama3"),
                memory=FakeMemory(),
                files=FakeFiles(),
                router=router,
                editor=FakeEditor(),
                image_manager=FakeImageManager(),
            )
            controller = AgentController(paths=paths, services=services)

            selected = controller.sync_available_models(["mistral:latest", "codellama:latest"])

            self.assertEqual(selected, "mistral:latest")
            self.assertEqual(controller.engine.model, "mistral:latest")
            self.assertEqual(router.model, "mistral:latest")
            self.assertEqual(settings.get("selected_model"), "mistral:latest")


if __name__ == "__main__":
    unittest.main()
