import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.ai.planner import build_plan, format_plan


class BuildPlanTests(unittest.TestCase):
    def test_complex_task_returns_steps(self):
        def gen(prompt, schema):
            return {"complex": True, "steps": ["Lire config.py", "Ajouter le champ", "Tester"]}

        plan = build_plan(gen, "ajoute un champ timeout configurable et teste-le")
        self.assertEqual(plan, ["Lire config.py", "Ajouter le champ", "Tester"])

    def test_simple_task_returns_empty(self):
        def gen(prompt, schema):
            return {"complex": False, "steps": []}

        self.assertEqual(build_plan(gen, "quelle heure est-il ?"), [])

    def test_steps_capped_at_max(self):
        def gen(prompt, schema):
            return {"complex": True, "steps": [f"étape {i}" for i in range(10)]}

        self.assertEqual(len(build_plan(gen, "grosse tâche", max_steps=3)), 3)

    def test_blank_steps_filtered(self):
        def gen(prompt, schema):
            return {"complex": True, "steps": ["  ", "Faire X", ""]}

        self.assertEqual(build_plan(gen, "tâche"), ["Faire X"])

    def test_no_generator_returns_empty(self):
        self.assertEqual(build_plan(None, "peu importe"), [])

    def test_empty_task_returns_empty(self):
        self.assertEqual(build_plan(lambda p, s: {"complex": True, "steps": ["x"]}, "  "), [])

    def test_malformed_output_returns_empty(self):
        self.assertEqual(
            build_plan(lambda p, s: {"complex": True, "steps": "pas une liste"}, "t"), []
        )
        self.assertEqual(build_plan(lambda p, s: None, "t"), [])

    def test_generator_exception_is_swallowed(self):
        def boom(prompt, schema):
            raise RuntimeError("model down")

        self.assertEqual(build_plan(boom, "tâche"), [])


class FormatPlanTests(unittest.TestCase):
    def test_numbered_and_marked_as_proposal(self):
        text = format_plan(["Lire", "Écrire"])
        self.assertIn("1. Lire", text)
        self.assertIn("2. Écrire", text)
        self.assertIn("PLAN PROPOSÉ", text)
        self.assertIn("ne les récite pas", text)  # anti-parrot guard


if __name__ == "__main__":
    unittest.main()
