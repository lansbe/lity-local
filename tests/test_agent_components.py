import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.receipts import summarize_receipts
from lity.services.ai.tool_specs import build_tool_specs


class AgentReceiptsTests(unittest.TestCase):
    def test_empty_receipts_have_no_summary(self):
        self.assertIsNone(summarize_receipts([]))

    def test_receipts_summary_is_grounded_when_any_tool_succeeds(self):
        summary = summarize_receipts(
            [
                {"name": "read_file", "ok": False, "detail": "missing"},
                {"name": "search", "ok": True, "detail": "match"},
            ]
        )

        self.assertIsNotNone(summary)
        self.assertTrue(summary["grounded"])
        self.assertEqual(summary["tools_used"], ["read_file", "search"])
        self.assertEqual(len(summary["items"]), 2)


class ToolSpecsTests(unittest.TestCase):
    def test_builds_file_web_command_write_and_retrieval_specs(self):
        specs = build_tool_specs(
            allow_files=True,
            allow_commands=True,
            allow_write=True,
            has_editor=True,
            has_web=True,
            retrieval={"project": object(), "memory": object()},
            mcp=None,
        )

        names = [tool["function"]["name"] for tool in specs]
        self.assertIn("list_files", names)
        self.assertIn("web_search", names)
        self.assertIn("retrieve_project", names)
        self.assertIn("recall_memory", names)
        self.assertIn("run_command", names)
        self.assertIn("write_file", names)
        self.assertIn("edit_file", names)

    def test_merges_mcp_tool_specs_best_effort(self):
        class FakeMCP:
            def tool_specs(self):
                return [{"function": {"name": "mcp_tool"}}]

        specs = build_tool_specs(
            allow_files=False,
            allow_commands=False,
            allow_write=False,
            has_editor=False,
            has_web=False,
            retrieval={},
            mcp=FakeMCP(),
        )

        self.assertEqual([tool["function"]["name"] for tool in specs], ["mcp_tool"])


if __name__ == "__main__":
    unittest.main()
