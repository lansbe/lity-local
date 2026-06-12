import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app._controller_background import BackgroundTaskMixin
from lity.app._controller_conversations import ConversationMixin
from lity.app._controller_health import HealthMixin
from lity.app.controller import AgentController


class ControllerComponentTests(unittest.TestCase):
    def test_controller_keeps_public_surface_after_mixin_splits(self):
        expected_methods = [
            "health",
            "get_settings",
            "update_settings",
            "process_user_message_sync",
            "process_user_message_stream",
            "list_conversations",
            "new_conversation",
            "switch_conversation",
            "get_messages",
        ]

        for name in expected_methods:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(AgentController, name, None)))

    def test_health_lives_in_focused_mixin(self):
        self.assertTrue(issubclass(AgentController, HealthMixin))

    def test_conversations_live_in_focused_mixin(self):
        self.assertTrue(issubclass(AgentController, ConversationMixin))

    def test_background_tasks_live_in_focused_mixin(self):
        self.assertTrue(issubclass(AgentController, BackgroundTaskMixin))


if __name__ == "__main__":
    unittest.main()
