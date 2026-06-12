import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_loop import FakeEditor, FakeFiles, ScriptedEngine, _collect_events

from lity.services.ai.agent import AgentLoop, _clip_command_output


class ClipCommandOutputTests(unittest.TestCase):
    def test_short_output_is_unchanged(self):
        self.assertEqual(_clip_command_output("ok"), "ok")

    def test_long_output_keeps_head_and_tail(self):
        # pytest puts its verdict at the END — the tail must survive truncation.
        output = "DÉBUT " + ("x" * 10_000) + " FAILED tests/test_x.py — fin du rapport"
        clipped = _clip_command_output(output, head=100, tail=100)
        self.assertIn("DÉBUT", clipped)
        self.assertIn("fin du rapport", clipped)
        self.assertIn("FAILED", clipped)
        self.assertIn("tronquée", clipped)
        self.assertLess(len(clipped), 300)


class VerifyWritesTests(unittest.TestCase):
    def test_verify_flags_broken_json_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bad.json").write_text("{pas du json", encoding="utf-8")
            (Path(tmp) / "good.json").write_text('{"ok": true}', encoding="utf-8")

            class _Files:
                working_dir = Path(tmp)

            loop = AgentLoop(ScriptedEngine([]), _Files())
            self.assertIn("JSON", loop._verify_python_writes(["bad.json"]) or "")
            self.assertIsNone(loop._verify_python_writes(["good.json"]))


class VerifyCommandTests(unittest.TestCase):
    """Definition of done: the project check command gets the last word."""

    def _loop(self, engine, shell_results):
        loop = AgentLoop(
            engine,
            FakeFiles({}),
            allow_write=True,
            editor=FakeEditor(),
            verify_command="pytest -q",
        )
        results = list(shell_results)
        calls = []

        def fake_shell(command, timeout=None):
            calls.append(command)
            return results.pop(0)

        loop._run_shell = fake_shell
        return loop, calls

    def test_red_check_is_reinjected_and_model_fixes(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {"name": "write_file", "arguments": {"path": "a.md", "content": "x"}}
                    ],
                },
                {"content": "C'est terminé.", "tool_calls": []},  # 1st final → verify fails
                {
                    "content": None,
                    "tool_calls": [
                        {"name": "write_file", "arguments": {"path": "a.md", "content": "y"}}
                    ],
                },
                {"content": "Corrigé, terminé.", "tool_calls": []},  # 2nd final → verify passes
            ]
        )
        loop, calls = self._loop(
            engine,
            [(False, "[exit 1]\n1 failed"), (True, "[exit 0]\nok")],
        )
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "répare le bug"}], on_event)

        self.assertEqual(answer, "Corrigé, terminé.")
        self.assertEqual(calls, ["pytest -q", "pytest -q"])
        # The red check was surfaced to the UI and reinjected to the model.
        verify_events = [
            p for k, p in events if k == "tool_result" and p["name"] == "verify_command"
        ]
        self.assertEqual([event["ok"] for event in verify_events], [False, True])
        reinjected = any(
            "VÉRIFICATION PROJET" in str(message.get("content", ""))
            for call in engine.calls
            for message in call["messages"]
        )
        self.assertTrue(reinjected)

    def test_persistent_red_check_is_admitted_in_answer(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {"name": "write_file", "arguments": {"path": "a.md", "content": "x"}}
                    ],
                },
                {"content": "Fini.", "tool_calls": []},
                {"content": "Fini (bis).", "tool_calls": []},
            ]
        )
        loop, _calls = self._loop(
            engine,
            [(False, "[exit 1]\nboom"), (False, "[exit 1]\nboom")],
        )
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "répare"}], on_event)

        self.assertIn("échoue encore", answer)  # honesty: the check is still red

    def test_no_verify_without_writes(self):
        engine = ScriptedEngine([{"content": "Réponse directe.", "tool_calls": []}])
        loop, calls = self._loop(engine, [])
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "question simple"}], on_event)

        self.assertEqual(answer, "Réponse directe.")
        self.assertEqual(calls, [])  # the check never ran


class FailureBudgetResetTests(unittest.TestCase):
    def test_scattered_failures_do_not_kill_the_run(self):
        # fail, succeed, fail, succeed, fail → never 3 CONSECUTIVE failures, so
        # the loop must not force an early answer.
        bad = {
            "content": None,
            "tool_calls": [{"name": "read_file", "arguments": {"path": "nope.py"}}],
        }
        good = {
            "content": None,
            "tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}],
        }
        engine = ScriptedEngine([bad, good, bad, good, bad, {"content": "Fin.", "tool_calls": []}])
        loop = AgentLoop(engine, FakeFiles({"a.py": "x = 1"}), max_steps=10)
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "explore"}], on_event)

        self.assertEqual(answer, "Fin.")
        self.assertEqual(len(engine.calls), 6)  # all steps ran, no forced stop


if __name__ == "__main__":
    unittest.main()
