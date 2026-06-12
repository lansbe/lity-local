import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.agent import AgentLoop, _strip_tool_json


class ScriptedEngine:
    """Returns pre-scripted chat_with_tools results, one per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat_with_tools(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        if self.script:
            return self.script.pop(0)
        return {"content": "réponse finale par défaut", "tool_calls": []}


class FakeFiles:
    working_dir = Path("/tmp/project")

    def __init__(self, files):
        self._files = files

    def get_available_files(self, recursive=False):
        return list(self._files.keys())

    def read_file_safe(self, path, max_chars=20_000):
        if path in self._files:
            return True, self._files[path]
        return False, f"'{path}' introuvable."


class FakeEditor:
    def __init__(self):
        self.created = {}
        self.edits = []

    def create_file(self, path, content, working_dir=None, overwrite=False):
        self.created[path] = content
        return True, f"créé {path}"

    def apply_edit(self, path, search, replace, working_dir=None):
        self.edits.append((path, search, replace))
        return True, f"modifié {path}"


def _collect_events():
    events = []
    return events, lambda kind, payload: events.append((kind, payload))


class AgentLoopTests(unittest.TestCase):
    def test_degrades_to_plain_answer_without_tool_calls(self):
        engine = ScriptedEngine([{"content": "Bonjour, voici ma réponse.", "tool_calls": []}])
        loop = AgentLoop(engine, FakeFiles({}))
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "salut"}], on_event)

        self.assertEqual(answer, "Bonjour, voici ma réponse.")
        self.assertEqual(events, [])
        self.assertEqual(len(engine.calls), 1)

    def test_preserves_think_blocks_for_collapsible_ui(self):
        # Reasoning is kept in the returned content (the frontend renders it as a
        # collapsible block); the loop only uses it to detect a real answer.
        engine = ScriptedEngine(
            [{"content": "<think>je réfléchis…</think>La réponse est 42.", "tool_calls": []}]
        )
        loop = AgentLoop(engine, FakeFiles({}))
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "?"}], on_event)

        self.assertIn("La réponse est 42.", answer)
        self.assertIn("<think>", answer)

    def test_answerless_think_falls_back(self):
        # A reply that is only an unfinished <think> (no answer) must not be
        # returned as the answer; the loop falls back instead.
        engine = ScriptedEngine([{"content": "<think>je réfléchis sans finir", "tool_calls": []}])
        loop = AgentLoop(engine, FakeFiles({}))
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "?"}], on_event)

        self.assertNotIn("<think>", answer)  # not surfaced as the answer

    def test_falls_back_to_plain_chat_when_model_lacks_tool_support(self):
        # Reasoning/distill models (e.g. deepseek-r1) often error on tool-calling.
        class ToolsUnsupportedEngine:
            def __init__(self):
                self.calls = []

            def chat_with_tools(self, messages, tools=None, **kwargs):
                self.calls.append(bool(tools))
                if tools:
                    return {"content": None, "tool_calls": [], "error": "does not support tools"}
                return {"content": "Réponse directe.", "tool_calls": []}

        engine = ToolsUnsupportedEngine()
        loop = AgentLoop(engine, FakeFiles({}))
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "cherche"}], on_event)

        self.assertIn("Réponse directe.", answer)
        self.assertIn("compatible", answer)  # helpful note about model tool support
        self.assertEqual(engine.calls, [True, False])  # retried once without tools

    def test_file_tools_hidden_without_workspace(self):
        loop = AgentLoop(
            ScriptedEngine([]),
            FakeFiles({}),
            allow_files=False,
            web={"searcher": None, "fetcher": None},
        )
        names = [tool["function"]["name"] for tool in loop.tool_specs()]
        self.assertNotIn("list_files", names)
        self.assertNotIn("read_file", names)
        self.assertNotIn("search", names)
        self.assertIn("web_search", names)  # web stays available

    def test_strips_leaked_tool_json_from_answer(self):
        text = (
            'Je reformule ma recherche. {"name": "web_search", "parameters": {"query": "Claude"}}'
        )
        self.assertEqual(_strip_tool_json(text), "Je reformule ma recherche.")

    def test_nudges_when_model_leaks_tool_json_as_text(self):
        # The model writes the tool call as TEXT instead of calling it; the loop
        # nudges it once, then the model produces a real answer.
        engine = ScriptedEngine(
            [
                {
                    "content": 'Je regarde. {"name": "web_search", "parameters": {"query": "x"}}',
                    "tool_calls": [],
                },
                {"content": "Voici la réponse finale.", "tool_calls": []},
            ]
        )
        loop = AgentLoop(
            engine, FakeFiles({}), allow_files=False, web={"searcher": None, "fetcher": None}
        )
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "cherche x"}], on_event)

        self.assertEqual(answer, "Voici la réponse finale.")
        self.assertEqual(len(engine.calls), 2)  # nudged, then answered
        nudged = [
            message
            for message in engine.calls[1]["messages"]
            if "TEXTE/JSON" in str(message.get("content", ""))
        ]
        self.assertTrue(nudged)  # the corrective nudge was injected

    def test_forces_final_answer_after_repeated_tool_failures(self):
        # The model keeps calling a tool that fails; the loop stops early (well
        # before max_steps) and forces a final answer with tools disabled.
        bad_call = {
            "content": None,
            "tool_calls": [{"name": "read_file", "arguments": {"path": "nope.py"}}],
        }
        engine = ScriptedEngine(
            [bad_call, bad_call, bad_call, {"content": "Réponse forcée.", "tool_calls": []}]
        )
        loop = AgentLoop(engine, FakeFiles({}), max_steps=20)
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "lis nope.py"}], on_event)

        self.assertEqual(answer, "Réponse forcée.")
        self.assertEqual(len(engine.calls), 4)  # 3 failed tool steps + 1 forced answer
        self.assertEqual(engine.calls[-1]["tools"], [])  # tools disabled on the forced turn

    def test_executes_tool_then_returns_final_answer(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}],
                },
                {"content": "Le fichier définit foo().", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({"a.py": "def foo():\n    return 1\n"}))
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "que fait a.py ?"}], on_event)

        self.assertEqual(answer, "Le fichier définit foo().")
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds, ["tool_call", "tool_result"])
        self.assertEqual(events[0][1]["name"], "read_file")
        self.assertTrue(events[1][1]["ok"])
        self.assertIn("foo", events[1][1]["summary"])

    def test_unknown_and_failed_tools_report_errors(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "nope.py"}}],
                },
                {"content": "Désolé.", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({}))
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "lis nope.py"}], on_event)

        result_event = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result_event["ok"])

    def test_run_command_disabled_by_default(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({}), allow_commands=False)
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "lance ls"}], on_event)

        result_event = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result_event["ok"])
        self.assertIn("désactivée", result_event["summary"])
        # run_command is not advertised as a tool when disabled.
        self.assertNotIn("run_command", [t["function"]["name"] for t in loop.tool_specs()])

    def test_run_command_advertised_when_allowed(self):
        loop = AgentLoop(ScriptedEngine([]), FakeFiles({}), allow_commands=True)
        self.assertIn("run_command", [t["function"]["name"] for t in loop.tool_specs()])

    def test_edit_failure_suggests_write_file(self):
        class FailEditEditor:
            def create_file(self, *args, **kwargs):
                return True, "ok"

            def apply_edit(self, *args, **kwargs):
                return False, "Le bloc SEARCH apparaît plusieurs fois. Modification annulée."

        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "name": "edit_file",
                            "arguments": {"path": "a.py", "search": "x", "replace": "y"},
                        }
                    ],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({}), allow_write=True, editor=FailEditEditor())
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "modifie"}], on_event)

        result_event = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result_event["ok"])
        self.assertIn("write_file", result_event["summary"])

    def test_command_approval_can_refuse(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "run_command", "arguments": {"command": "ls -la"}}],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({}), allow_commands=True, confirm=lambda command: False)
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "lance"}], on_event)

        result_event = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result_event["ok"])
        self.assertIn("refusée", result_event["summary"])

    def test_write_tools_disabled_by_default(self):
        loop = AgentLoop(ScriptedEngine([]), FakeFiles({}))
        names = [tool["function"]["name"] for tool in loop.tool_specs()]
        self.assertNotIn("write_file", names)
        self.assertNotIn("edit_file", names)

    def test_write_file_handler_refuses_when_write_mode_is_not_autonomous(self):
        loop = AgentLoop(ScriptedEngine([]), FakeFiles({}), editor=FakeEditor())

        ok, message = loop._write_file({"path": "x.py", "content": "print(1)\n"})

        self.assertFalse(ok)
        self.assertIn("désactivée", message.lower())

    def test_yolo_write_file_executes(self):
        editor = FakeEditor()
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {"name": "write_file", "arguments": {"path": "poeme.md", "content": "Oh"}}
                    ],
                },
                {"content": "J'ai créé le fichier.", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({}), allow_write=True, editor=editor)
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "crée poeme.md"}], on_event)

        self.assertEqual(answer, "J'ai créé le fichier.")
        self.assertEqual(editor.created.get("poeme.md"), "Oh")
        self.assertIn("write_file", [tool["function"]["name"] for tool in loop.tool_specs()])
        result_event = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertTrue(result_event["ok"])

    def test_stops_after_max_steps_and_forces_answer(self):
        # Always returns a tool call; loop must stop and force a final answer.
        always_tool = {"content": None, "tool_calls": [{"name": "list_files", "arguments": {}}]}
        engine = ScriptedEngine([always_tool] * 10)
        loop = AgentLoop(engine, FakeFiles({"a.py": "x = 1"}), max_steps=3)
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "explore"}], on_event)

        # 3 stepped tool calls advertised + 1 forced final call (tools disabled).
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual(engine.calls[-1]["tools"], [])
        self.assertIsInstance(answer, str)

    def test_cancellation_stops_loop(self):
        engine = ScriptedEngine(
            [{"content": None, "tool_calls": [{"name": "list_files", "arguments": {}}]}]
        )
        loop = AgentLoop(engine, FakeFiles({"a.py": "x = 1"}))
        events, on_event = _collect_events()

        answer = loop.run([{"role": "user", "content": "go"}], on_event, should_cancel=lambda: True)

        self.assertEqual(engine.calls, [])  # cancelled before first call
        self.assertIsInstance(answer, str)

    def test_detects_and_breaks_a_tool_loop(self):
        # The model calls list_files forever; the same (tool, args, result) is
        # detected as a loop and the run is forced to stop well before max_steps.
        always = {"content": None, "tool_calls": [{"name": "list_files", "arguments": {}}]}
        engine = ScriptedEngine([always] * 12 + [{"content": "fini", "tool_calls": []}])
        loop = AgentLoop(engine, FakeFiles({"a.py": "x"}), max_steps=20)
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "boucle"}], on_event)

        self.assertLess(len(engine.calls), 8)  # stopped early, not 20
        nudged = any(
            "tourn" in str(message.get("content", "")).lower()
            for call in engine.calls
            for message in call["messages"]
        )
        self.assertTrue(nudged)  # the anti-loop nudge was injected

    def test_unknown_tool_name_suggests_a_valid_one(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "read_fil", "arguments": {"path": "a.py"}}],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({"a.py": "x"}))
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "lis a.py"}], on_event)

        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result["ok"])
        self.assertIn("read_file", result["summary"])  # closest-match suggestion

    def test_missing_required_argument_is_reported(self):
        engine = ScriptedEngine(
            [
                {"content": None, "tool_calls": [{"name": "read_file", "arguments": {}}]},
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({"a.py": "x"}))
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "lis"}], on_event)

        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result["ok"])
        self.assertIn("path", result["summary"])

    def test_read_file_windowed_with_line_numbers(self):
        loop = AgentLoop(ScriptedEngine([]), FakeFiles({"a.py": "l1\nl2\nl3\nl4\nl5"}))
        ok, windowed = loop._read_file({"path": "a.py", "offset": 2, "limit": 2})
        self.assertTrue(ok)
        self.assertEqual(windowed, "2: l2\n3: l3")
        _, full = loop._read_file({"path": "a.py"})
        self.assertEqual(full, "l1\nl2\nl3\nl4\nl5")  # no window → plain content

    def test_lint_rejects_a_broken_python_write(self):
        editor = FakeEditor()
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "name": "write_file",
                            "arguments": {"path": "bad.py", "content": "def f(:\n"},
                        }
                    ],
                },
                {"content": "ok", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({}), allow_write=True, editor=editor)
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "écris bad.py"}], on_event)

        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertFalse(result["ok"])
        self.assertIn("SyntaxError", result["summary"])
        self.assertNotIn("bad.py", editor.created)  # never written to disk

    def test_lint_allows_a_valid_python_write(self):
        editor = FakeEditor()
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "name": "write_file",
                            "arguments": {"path": "ok.py", "content": "def f():\n    return 1\n"},
                        }
                    ],
                },
                {"content": "fait", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles({}), allow_write=True, editor=editor)
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "écris ok.py"}], on_event)

        self.assertEqual(editor.created.get("ok.py"), "def f():\n    return 1\n")

    def test_verify_python_writes_flags_a_broken_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.py").write_text("def f(:\n", encoding="utf-8")
            (Path(tmp) / "good.py").write_text("def f():\n    return 1\n", encoding="utf-8")

            class _Files:
                working_dir = Path(tmp)

            loop = AgentLoop(ScriptedEngine([]), _Files())
            self.assertIn("SyntaxError", loop._verify_python_writes(["broken.py"]) or "")
            self.assertIsNone(loop._verify_python_writes(["good.py"]))
            self.assertIsNone(loop._verify_python_writes(["notes.md"]))  # non-.py ignored

    def test_retrieve_project_tool_when_provided(self):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "retrieve_project", "arguments": {"query": "auth"}}],
                },
                {"content": "L'auth est dans auth.py.", "tool_calls": []},
            ]
        )
        hits = [{"path": "auth.py", "text": "def login(): ..."}]
        loop = AgentLoop(
            engine, FakeFiles({}), allow_files=False, retrieval={"project": lambda q, k=5: hits}
        )
        names = [tool["function"]["name"] for tool in loop.tool_specs()]
        self.assertIn("retrieve_project", names)
        self.assertNotIn("recall_memory", names)  # only project source provided

        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "où est l'auth ?"}], on_event)
        result = [payload for kind, payload in events if kind == "tool_result"][0]
        self.assertTrue(result["ok"])
        self.assertIn("auth.py", result["summary"])

    def test_recall_memory_tool_handles_empty_result(self):
        loop = AgentLoop(
            ScriptedEngine([]),
            FakeFiles({}),
            allow_files=False,
            retrieval={"memory": lambda q, k=5: []},
        )
        self.assertIn("recall_memory", [t["function"]["name"] for t in loop.tool_specs()])
        ok, out = loop._recall_memory({"query": "mon prénom"})
        self.assertTrue(ok)
        self.assertIn("aucun extrait", out.lower())


class ReceiptsTests(unittest.TestCase):
    """Tool-call provenance ledger + grounding verdict (anti-hallucination)."""

    def _read_then_answer(self, files):
        engine = ScriptedEngine(
            [
                {
                    "content": None,
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}],
                },
                {"content": "Réponse.", "tool_calls": []},
            ]
        )
        loop = AgentLoop(engine, FakeFiles(files))
        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "lis a.py"}], on_event)
        return loop

    def test_successful_tool_is_grounded(self):
        loop = self._read_then_answer({"a.py": "def foo():\n    return 1\n"})
        summary = loop.receipts_summary()
        self.assertIsNotNone(summary)
        self.assertTrue(summary["grounded"])
        self.assertEqual(summary["tools_used"], ["read_file"])
        self.assertEqual(len(summary["items"]), 1)
        self.assertTrue(summary["items"][0]["ok"])

    def test_all_failed_tools_is_not_grounded(self):
        loop = self._read_then_answer({})  # a.py missing → read_file fails
        summary = loop.receipts_summary()
        self.assertIsNotNone(summary)
        self.assertFalse(summary["grounded"])  # forced answer, unverified
        self.assertFalse(summary["items"][0]["ok"])

    def test_plain_answer_has_no_receipts(self):
        engine = ScriptedEngine([{"content": "Bonjour !", "tool_calls": []}])
        loop = AgentLoop(engine, FakeFiles({}))
        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "salut"}], on_event)
        self.assertIsNone(loop.receipts_summary())  # no tool ran → nothing to attest

    def test_ledger_resets_between_runs(self):
        loop = self._read_then_answer({"a.py": "x = 1\n"})
        self.assertEqual(len(loop.last_receipts), 1)
        # A second, tool-free run must not carry the first run's receipts.
        loop.engine.script = [{"content": "Re-bonjour.", "tool_calls": []}]
        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "salut"}], on_event)
        self.assertEqual(loop.last_receipts, [])
        self.assertIsNone(loop.receipts_summary())


if __name__ == "__main__":
    unittest.main()
