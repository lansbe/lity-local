import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app.controller import AgentController
from lity.app.services import AppServices
from lity.infrastructure.paths import AppPaths
from lity.services.memory.json_memory import MemoryManager


class _InstructionsEngine:
    model = "fake"

    def __init__(self):
        self.system_prompt_extra = ""
        self.temperature = None

    def get_installed_models(self):
        return ["fake"]


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


class ConversationInstructionsTests(unittest.TestCase):
    def _controller(self, tmp):
        paths = AppPaths.create(home_override=Path(tmp))
        services = AppServices(
            settings=None,
            engine=_InstructionsEngine(),
            memory=MemoryManager(paths=paths),
            files=_Files(),
            router=_Router(),
            editor=_Editor(),
            image_manager=None,
        )
        return AgentController(paths=paths, services=services)

    def test_instructions_applied_to_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(tmp)
            self.assertTrue(controller.set_conversation_instructions("Sois très concise.", 0.4))
            controller._apply_conversation_instructions()
            self.assertIn("Sois très concise.", controller.engine.system_prompt_extra)
            self.assertEqual(controller.engine.temperature, 0.4)

    def test_clearing_instructions_resets_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(tmp)
            controller.set_conversation_instructions("X", 0.4)
            controller._apply_conversation_instructions()
            controller.set_conversation_instructions("", None)
            controller._apply_conversation_instructions()
            self.assertIsNone(controller.engine.temperature)
            self.assertEqual(controller.engine.system_prompt_extra, "")

    def test_get_instructions_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(tmp)
            controller.set_conversation_instructions("Réponds en format court.", 0.9)
            data = controller.get_conversation_instructions()
            self.assertEqual(data["instructions"], "Réponds en format court.")
            self.assertEqual(data["temperature"], 0.9)


if __name__ == "__main__":
    unittest.main()
