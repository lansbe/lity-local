import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.core.model_advisor import (
    classify_model,
    estimate_params_from_size,
    rank_models,
    supports_tools,
)

GB = 1024**3


class ClassifyModelTests(unittest.TestCase):
    def test_verdict_thresholds(self):
        self.assertEqual(classify_model(3, 10)[0], "excellent")  # ratio .35
        self.assertEqual(classify_model(7, 10)[0], "bon")  # ratio .80
        self.assertEqual(classify_model(9, 10)[0], "limite")  # ratio 1.03
        self.assertEqual(classify_model(20, 10)[0], "trop_lourd")  # ratio 2.3

    def test_unknown_budget_is_neutral(self):
        self.assertEqual(classify_model(8, 0), ("bon", "inconnu"))

    def test_speed_depends_on_accelerator(self):
        self.assertEqual(classify_model(3, 10, "metal")[1], "rapide (GPU)")
        self.assertEqual(classify_model(3, 10, "cuda")[1], "rapide (GPU)")
        self.assertEqual(classify_model(3, 10, "cpu")[1], "lent (CPU)")
        self.assertEqual(classify_model(20, 10, "metal")[1], "ralenti")

    def test_estimate_params_from_size(self):
        self.assertEqual(estimate_params_from_size(4.8), 8.0)
        self.assertGreater(estimate_params_from_size(0), 0)

    def test_dedicated_gpu_offloads_to_ram(self):
        # 8 GB VRAM card, 32 GB RAM. A ~16 GB model exceeds VRAM but fits RAM →
        # runs with CPU offload ("limite"), not impossible.
        self.assertEqual(classify_model(16, 8, "cuda", ram_gb=32), ("limite", "ralenti (RAM)"))
        # A 40 GB model exceeds even RAM → truly too heavy.
        self.assertEqual(classify_model(40, 8, "cuda", ram_gb=32)[0], "trop_lourd")
        # Without a RAM ceiling, exceeding the budget stays "trop_lourd".
        self.assertEqual(classify_model(16, 8, "cuda")[0], "trop_lourd")


class RankModelsTests(unittest.TestCase):
    def test_ranks_best_to_worst_and_flags_recommended(self):
        hardware = {"budget_gb": 11.2, "accelerator": "metal"}  # ~16 GB Mac
        rows = rank_models(hardware)

        # Best→worst: the first row must be a runnable model.
        self.assertIn(rows[0]["verdict"], ("excellent", "bon"))
        # Exactly one recommended general-purpose pick, and it's runnable.
        recommended = [row for row in rows if row.get("recommended")]
        self.assertEqual(len(recommended), 1)
        self.assertIn(recommended[0]["verdict"], ("excellent", "bon"))

        by_name = {row["name"]: row for row in rows}
        self.assertEqual(by_name["llama3.3:70b"]["verdict"], "trop_lourd")
        # A too-heavy model sinks below a comfortable one.
        self.assertGreater(rows.index(by_name["llama3.3:70b"]), rows.index(by_name["llama3.1:8b"]))

    def test_installed_models_use_real_size_and_flag(self):
        hardware = {"budget_gb": 11.2, "accelerator": "metal"}
        rows = rank_models(
            hardware,
            installed=[
                {"name": "llama3.1:8b", "size": int(4.7 * GB)},
                {"name": "mystery:latest", "size": int(2 * GB)},
            ],
        )
        by_name = {row["name"]: row for row in rows}

        self.assertTrue(by_name["llama3.1:8b"]["installed"])
        self.assertAlmostEqual(by_name["llama3.1:8b"]["size_gb"], 4.7, places=1)
        # Installed model absent from the catalog still appears, flagged installed.
        self.assertIn("mystery:latest", by_name)
        self.assertTrue(by_name["mystery:latest"]["installed"])

    def test_tiny_budget_marks_everything_heavy(self):
        rows = rank_models({"budget_gb": 1.0, "accelerator": "cpu"})
        # A 70B model cannot run on 1 GB.
        big = next(row for row in rows if row["name"] == "llama3.3:70b")
        self.assertEqual(big["verdict"], "trop_lourd")

    def test_rows_carry_tool_use_flag(self):
        rows = rank_models({"budget_gb": 11.2, "accelerator": "metal"})
        by_name = {row["name"]: row for row in rows}
        self.assertTrue(by_name["qwen3:8b"]["tool_use"])
        self.assertTrue(by_name["llama3.1:8b"]["tool_use"])
        self.assertFalse(by_name["deepseek-r1:8b"]["tool_use"])  # reasoning distill
        self.assertFalse(by_name["gemma2:2b"]["tool_use"])
        self.assertFalse(by_name["llama3.2-vision:11b"]["tool_use"])  # vision
        self.assertFalse(by_name["bge-m3"]["tool_use"])  # embeddings


class SupportsToolsTests(unittest.TestCase):
    def test_tool_capable_families(self):
        for name in ("qwen3:8b", "qwen2.5-coder:7b", "llama3.1:8b", "llama3.3:70b", "mistral:7b"):
            self.assertTrue(supports_tools(name, "chat"), name)

    def test_tool_incapable(self):
        self.assertFalse(supports_tools("deepseek-r1:8b", "reasoning"))
        self.assertFalse(supports_tools("gemma2:9b", "chat"))
        self.assertFalse(supports_tools("phi3:3.8b", "chat"))
        self.assertFalse(supports_tools("llava:7b", "vision"))
        self.assertFalse(supports_tools("bge-m3", "embed"))
        # Vision variant of a tool-capable family is still excluded by kind.
        self.assertFalse(supports_tools("llama3.2-vision:11b", "vision"))

    def test_unknown_family_defaults_false(self):
        self.assertFalse(supports_tools("mystery:latest", "chat"))


if __name__ == "__main__":
    unittest.main()
