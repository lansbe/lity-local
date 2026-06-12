import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.agent import AgentLoop
from lity.services.mcp import _load_servers, build_mcp_manager
from lity.services.mcp.manager import _format_result, _safe_name


class _ScriptedEngine:
    def __init__(self, script):
        self.script = list(script)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        return self.script.pop(0) if self.script else {"content": "fini", "tool_calls": []}


class _Files:
    working_dir = None

    def get_available_files(self, recursive=False):
        return []

    def read_file_safe(self, path, max_chars=20_000):
        return False, "n/a"


class _FakeMCP:
    def tool_specs(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "mcp_fs_read_file",
                    "description": "Read a file from an MCP server",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

    def call(self, name, args):
        return True, f"MCP:{name}:{args.get('path')}"


def _collect():
    events = []
    return events, lambda kind, payload: events.append((kind, payload))


class LoadServersTests(unittest.TestCase):
    def test_parses_claude_desktop_format(self):
        cfg = (
            '{"mcpServers": {"fs": {"command": "uvx", "args": ["mcp-server-fs"]}, '
            '"bad": {"args": []}, "off": {"command": "x", "disabled": true}}}'
        )
        servers = _load_servers(cfg)
        self.assertEqual([s["name"] for s in servers], ["fs"])  # bad (no cmd) + off skipped
        self.assertEqual(servers[0]["command"], "uvx")
        self.assertEqual(servers[0]["args"], ["mcp-server-fs"])

    def test_malformed_or_empty_returns_empty(self):
        self.assertEqual(_load_servers("not json"), [])
        self.assertEqual(_load_servers("{}"), [])
        self.assertEqual(_load_servers('{"mcpServers": []}'), [])


class BuildManagerTests(unittest.TestCase):
    def test_none_when_config_missing_or_sdk_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(build_mcp_manager(Path(tmp) / "nope.json"))


class SafeNameAndFormatTests(unittest.TestCase):
    def test_safe_name_is_function_call_safe(self):
        self.assertEqual(_safe_name("fs server", "read-file"), "mcp_fs_server_read_file")

    def test_format_result_flattens_text_and_error_flag(self):
        class Block:
            text = "hello"

        class Ok:
            content = [Block()]
            isError = False

        self.assertEqual(_format_result(Ok()), (True, "hello"))

        class Err:
            content = []
            isError = True

        ok, _text = _format_result(Err())
        self.assertFalse(ok)


class AgentMCPIntegrationTests(unittest.TestCase):
    def test_mcp_tools_advertised_and_routed(self):
        engine = _ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "mcp_fs_read_file", "arguments": {"path": "a.txt"}}],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, _Files(), allow_files=False, mcp=_FakeMCP())
        self.assertIn("mcp_fs_read_file", [t["function"]["name"] for t in loop.tool_specs()])

        events, on_event = _collect()
        loop.run([{"role": "user", "content": "lis a.txt via mcp"}], on_event)
        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertTrue(result["ok"])
        self.assertIn("MCP:mcp_fs_read_file:a.txt", result["summary"])

    def test_mcp_missing_required_arg_is_validated(self):
        loop = AgentLoop(_ScriptedEngine([]), _Files(), allow_files=False, mcp=_FakeMCP())
        ok, message = loop._execute("mcp_fs_read_file", {})
        self.assertFalse(ok)
        self.assertIn("path", message)


if __name__ == "__main__":
    unittest.main()
