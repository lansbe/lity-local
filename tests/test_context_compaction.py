import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_loop import AgentLoop, FakeFiles, ScriptedEngine, _collect_events

from lity.services.ai.context import (
    clamp_history,
    compact_agent_messages,
    compose_injected_context,
    estimate_messages_chars,
)


class CompactAgentMessagesTests(unittest.TestCase):
    def _work(self, tool_chars=3000, steps=12):
        messages = [
            {"role": "system", "content": "INSTRUCTIONS SYSTÈME"},
            {"role": "user", "content": "TÂCHE INITIALE : répare le bug"},
        ]
        for index in range(steps):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "read_file", "arguments": {}}}],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": "read_file",
                    "content": f"étape {index} " + "x" * tool_chars,
                }
            )
        return messages

    def test_under_budget_is_untouched(self):
        messages = self._work(tool_chars=10, steps=2)
        self.assertIs(compact_agent_messages(messages, max_chars=100_000), messages)

    def test_over_budget_protects_system_task_and_recent(self):
        messages = self._work()
        compacted = compact_agent_messages(messages, max_chars=20_000)
        self.assertLess(estimate_messages_chars(compacted), estimate_messages_chars(messages))
        self.assertEqual(compacted[0]["content"], "INSTRUCTIONS SYSTÈME")  # system intact
        contents = [str(message.get("content", "")) for message in compacted]
        self.assertTrue(any("TÂCHE INITIALE" in content for content in contents))  # task intact
        self.assertTrue(any("HISTORIQUE COMPACTÉ" in content for content in contents))  # visible
        # The most recent observation survives verbatim.
        self.assertIn("étape 11 " + "x" * 3000, contents[-1])

    def test_deep_over_budget_falls_back_to_step_summary(self):
        messages = self._work(tool_chars=5000, steps=30)
        compacted = compact_agent_messages(messages, max_chars=12_000)
        self.assertLessEqual(estimate_messages_chars(compacted), 60_000)
        contents = " ".join(str(message.get("content", "")) for message in compacted)
        self.assertIn("HISTORIQUE COMPACTÉ", contents)


class ClampHistoryTests(unittest.TestCase):
    def test_small_history_is_untouched(self):
        history = [{"role": "user", "content": "salut"}]
        self.assertIs(clamp_history(history, 1000), history)

    def test_keeps_newest_drops_oldest_with_note(self):
        history = [{"role": "user", "content": f"message {i} " + "x" * 500} for i in range(20)]
        clamped = clamp_history(history, 3000)
        self.assertLess(len(clamped), len(history) + 1)
        self.assertIn("retirés", clamped[0]["content"])  # explicit note
        self.assertIn("message 19", clamped[-1]["content"])  # newest survives


class ComposeInjectedContextTests(unittest.TestCase):
    def test_small_sections_survive_a_giant_files_section(self):
        facts = "[FAITS] l'utilisateur s'appelle Alex [/FAITS]\n"
        memory = "[MÉMOIRE] il préfère Python [/MÉMOIRE]\n"
        files = "--- CONTENU DE gros.py ---\n" + ("code " * 50_000)
        composed = compose_injected_context(
            [(files, 0.70), (facts, 0.15), (memory, 0.15)], max_chars=10_000
        )
        self.assertLessEqual(len(composed), 11_000)
        self.assertIn("Alex", composed)  # facts NOT evicted by the big file
        self.assertIn("Python", composed)
        self.assertIn("gros.py", composed)

    def test_rollover_lets_files_use_unused_budget(self):
        files = "F" * 9000
        composed = compose_injected_context([(files, 0.5), ("petit", 0.5)], max_chars=10_000)
        # The small section leaves its slice mostly unused; files reclaim it.
        self.assertIn("petit", composed)
        self.assertGreater(composed.count("F"), 4000)

    def test_empty_sections_yield_empty(self):
        self.assertEqual(compose_injected_context([("", 0.7), ("", 0.3)]), "")


class PlanReminderTests(unittest.TestCase):
    def test_plan_is_reanchored_during_a_long_loop(self):
        always = {"content": None, "tool_calls": [{"name": "list_files", "arguments": {}}]}
        varied = [
            {
                "content": None,
                "tool_calls": [{"name": "read_file", "arguments": {"path": f"f{i}.py"}}],
            }
            for i in range(8)
        ]
        engine = ScriptedEngine([always] + varied + [{"content": "fini", "tool_calls": []}])
        files = FakeFiles({f"f{i}.py": f"x = {i}" for i in range(8)})
        loop = AgentLoop(engine, files, max_steps=10, plan=["Lire", "Modifier", "Vérifier"])
        events, on_event = _collect_events()

        loop.run([{"role": "user", "content": "tâche complexe"}], on_event)

        reminded = any(
            "RAPPEL DU PLAN" in str(message.get("content", ""))
            for call in engine.calls
            for message in call["messages"]
        )
        self.assertTrue(reminded)

    def test_no_reminder_without_plan(self):
        engine = ScriptedEngine([{"content": "ok", "tool_calls": []}])
        loop = AgentLoop(engine, FakeFiles({}))
        events, on_event = _collect_events()
        loop.run([{"role": "user", "content": "salut"}], on_event)
        reminded = any(
            "RAPPEL DU PLAN" in str(message.get("content", ""))
            for call in engine.calls
            for message in call["messages"]
        )
        self.assertFalse(reminded)


if __name__ == "__main__":
    unittest.main()
