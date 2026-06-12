import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.interfaces.desktop_web._api_conversations import ConversationApiMixin
from lity.interfaces.desktop_web._api_workspace import WorkspaceApiMixin
from lity.interfaces.desktop_web.api import DesktopApi


class DesktopApiComponentTests(unittest.TestCase):
    def test_desktop_api_keeps_public_surface_after_mixin_splits(self):
        expected_methods = [
            "get_state",
            "set_workdir",
            "list_workspace_files",
            "apply_create",
            "apply_edit",
            "list_conversations",
            "send_message",
            "regenerate",
            "edit_and_resend",
            "list_models",
            "get_settings",
        ]

        for name in expected_methods:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(DesktopApi, name, None)))

    def test_workspace_and_conversation_domains_are_mixed_in(self):
        self.assertTrue(issubclass(DesktopApi, WorkspaceApiMixin))
        self.assertTrue(issubclass(DesktopApi, ConversationApiMixin))


if __name__ == "__main__":
    unittest.main()
