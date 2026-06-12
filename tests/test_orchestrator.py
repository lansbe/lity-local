import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.orchestrator import TaskOrchestrator


class FakeLoop:
    """Records the messages it received and returns a scripted answer."""

    def __init__(self, answer, receipts=None, verify_command=None):
        self.answer = answer
        self._receipts = receipts
        self.verify_command = verify_command
        self.seen_messages = None

    def run(self, messages, on_event, should_cancel=None):
        self.seen_messages = list(messages)
        return self.answer

    def receipts_summary(self):
        return self._receipts


class TaskOrchestratorTests(unittest.TestCase):
    def _factory(self, loops):
        created = []

        def factory(**overrides):
            loop = loops[len(created)]
            created.append(loop)
            return loop

        return factory, created

    def test_each_step_runs_in_a_fresh_context(self):
        loops = [FakeLoop("résultat étape 1"), FakeLoop("résultat étape 2")]
        factory, created = self._factory(loops)
        orchestrator = TaskOrchestrator(factory)
        events = []

        answer, receipts = orchestrator.run(
            goal="refactorer le module",
            plan=["Lire le code", "Appliquer le refactor"],
            base_messages=[
                {"role": "system", "content": "SYSTÈME"},
                {"role": "user", "content": "refactorer le module"},
            ],
            on_event=lambda kind, payload: events.append((kind, payload)),
        )

        self.assertEqual(len(created), 2)
        # Fresh context: system + ONE focused user message per sub-task.
        for loop in created:
            self.assertEqual(loop.seen_messages[0]["role"], "system")
            self.assertEqual(len(loop.seen_messages), 2)
        # Step 2 sees the goal, the plan and a digest of step 1's outcome.
        step2_prompt = created[1].seen_messages[1]["content"]
        self.assertIn("OBJECTIF GLOBAL", step2_prompt)
        self.assertIn("résultat étape 1", step2_prompt)
        self.assertIn("étape 2", step2_prompt)
        # The final answer carries the executed plan.
        self.assertIn("résultat étape 2", answer)
        self.assertIn("Plan exécuté", answer)
        # Sub-task progress was surfaced to the UI.
        subtasks = [payload for kind, payload in events if kind == "subtask"]
        self.assertEqual([item["index"] for item in subtasks], [1, 2])

    def test_verify_command_only_gates_the_last_step(self):
        loops = [
            FakeLoop("ok 1", verify_command="pytest"),
            FakeLoop("ok 2", verify_command="pytest"),
        ]
        factory, created = self._factory(loops)
        TaskOrchestrator(factory).run(
            goal="g",
            plan=["a", "b"],
            base_messages=[{"role": "system", "content": "s"}],
            on_event=lambda *args: None,
        )
        self.assertIsNone(created[0].verify_command)  # intermediate: no gate
        self.assertEqual(created[1].verify_command, "pytest")  # final: gated

    def test_receipts_are_aggregated_across_steps(self):
        loops = [
            FakeLoop("r1", receipts={"items": [{"name": "read_file", "ok": True}]}),
            FakeLoop("r2", receipts={"items": [{"name": "write_file", "ok": True}]}),
        ]
        factory, _created = self._factory(loops)
        _answer, receipts = TaskOrchestrator(factory).run(
            goal="g",
            plan=["a", "b"],
            base_messages=[],
            on_event=lambda *args: None,
        )
        self.assertTrue(receipts["grounded"])
        self.assertEqual(receipts["tools_used"], ["read_file", "write_file"])

    def test_cancellation_stops_between_steps(self):
        loops = [FakeLoop("r1"), FakeLoop("r2")]
        factory, created = self._factory(loops)
        cancelled = {"after": 1, "count": 0}

        def should_cancel():
            cancelled["count"] += 1
            return cancelled["count"] > cancelled["after"]

        TaskOrchestrator(factory).run(
            goal="g",
            plan=["a", "b"],
            base_messages=[],
            on_event=lambda *args: None,
            should_cancel=should_cancel,
        )
        self.assertEqual(len(created), 1)  # second step never started


if __name__ == "__main__":
    unittest.main()
