import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.core.compatibility import (
    HardwareProfile,
    compute_score,
    estimate_tokens_per_second,
    evaluate_model_complete,
    evaluate_status,
    profile_from_hardware,
    score_to_grade,
)
from lity.core.gpu_db import (
    APPLE_DB,
    GPU_DB,
    bandwidth_heuristic,
    match_apple_chip,
    match_gpu,
)
from lity.core.model_advisor import rank_models
from lity.core.model_catalog import (
    FULL_CATALOG,
    find_model,
    installable_models,
    make_quants,
)


class QuantMathTests(unittest.TestCase):
    """Exact port of canirun's makeQuants: same bytes/param, same overheads."""

    def test_8b_quant_sizes_match_canirun(self):
        quants = {q["name"]: q for q in make_quants(8)}
        # vram = params*bpp/2^30 * 1.1 + 0.5 ; disk = params*bpp/2^30 * 1.05
        self.assertEqual(quants["Q4_K_M"]["vram_gb"], 4.6)
        self.assertEqual(quants["Q4_K_M"]["disk_gb"], 3.9)
        self.assertEqual(quants["Q8_0"]["vram_gb"], 8.7)
        self.assertEqual(quants["F16"]["vram_gb"], 16.9)
        self.assertEqual(len(quants), 7)

    def test_70b_q4_matches_canirun(self):
        quants = {q["name"]: q for q in make_quants(70)}
        self.assertEqual(quants["Q4_K_M"]["vram_gb"], 36.4)


class CatalogTests(unittest.TestCase):
    def test_catalog_is_large_and_installables_have_ollama_ids(self):
        # Curated catalog ported from canirun, deduped for Lity. Floor guards
        # against accidental truncation of the shipped model_catalog.json.
        self.assertGreaterEqual(len(FULL_CATALOG), 74)
        installables = installable_models()
        self.assertGreaterEqual(len(installables), 65)
        self.assertTrue(all(model["ollama_id"] for model in installables))

    def test_find_model_matches_exact_then_base(self):
        self.assertEqual(find_model("qwen3:8b")["name"], "Qwen 3 8B")
        self.assertEqual(find_model("mistral:7b-instruct")["family"], "Mistral")
        self.assertIsNone(find_model("inconnu-9000"))

    def test_moe_models_carry_active_params(self):
        moe = find_model("qwen3:30b-a3b")
        self.assertEqual(moe["architecture"], "moe")
        self.assertEqual(moe["active_params_b"], 3.3)


class StatusThresholdTests(unittest.TestCase):
    """Same thresholds as canirun's evaluateModel."""

    def test_apple_silicon_thresholds(self):
        hw = HardwareProfile(total_usable_ram_gb=16, is_apple_silicon=True)
        # usable = 16*0.75 = 12 ; can-run ≤ 8.4 ; tight ≤ 12
        self.assertEqual(evaluate_status(8.0, hw), "can-run")
        self.assertEqual(evaluate_status(10.0, hw), "tight")
        self.assertEqual(evaluate_status(13.0, hw), "cannot-run")

    def test_discrete_gpu_thresholds_and_offload(self):
        hw = HardwareProfile(total_usable_ram_gb=8, estimated_vram_gb=8, system_ram_gb=32)
        # can-run ≤ 6.8 ; tight ≤ 8.8 ; offload ≤ 8 + 32*0.7 = 30.4
        self.assertEqual(evaluate_status(6.0, hw), "can-run")
        self.assertEqual(evaluate_status(8.5, hw), "tight")
        self.assertEqual(evaluate_status(20.0, hw), "can-run-slow")
        self.assertEqual(evaluate_status(35.0, hw), "cannot-run")

    def test_cpu_only_thresholds(self):
        hw = HardwareProfile(total_usable_ram_gb=16)
        # usable = 11.2 ; can-run ≤ 7.84 ; tight ≤ 11.2
        self.assertEqual(evaluate_status(7.0, hw), "can-run")
        self.assertEqual(evaluate_status(10.0, hw), "tight")
        self.assertEqual(evaluate_status(12.0, hw), "cannot-run")

    def test_unknown_hardware(self):
        self.assertEqual(evaluate_status(4.0, HardwareProfile()), "unknown")


class TokensPerSecondTests(unittest.TestCase):
    def test_apple_efficiency(self):
        hw = HardwareProfile(
            total_usable_ram_gb=24, memory_bandwidth_gbs=200, is_apple_silicon=True
        )
        # 200 / 4.6 * 0.65 ≈ 28
        self.assertEqual(estimate_tokens_per_second(4.6, hw), 28)

    def test_discrete_gpu_efficiency(self):
        hw = HardwareProfile(estimated_vram_gb=24, system_ram_gb=32, memory_bandwidth_gbs=1008)
        # 1008 / 4.6 * 0.70 ≈ 153
        self.assertEqual(estimate_tokens_per_second(4.6, hw), 153)

    def test_offload_uses_harmonic_mean_with_penalty(self):
        hw = HardwareProfile(estimated_vram_gb=8, system_ram_gb=32, memory_bandwidth_gbs=300)
        toks = estimate_tokens_per_second(16.0, hw)
        # fraction VRAM 0.5 / RAM 0.5 → bw_eff = 1/(0.5/300 + 0.5/50) ≈ 85.7
        # 85.7/16*0.70*0.85 ≈ 3.2 → ≈3 tok/s (slow but possible)
        self.assertEqual(toks, 3)

    def test_no_bandwidth_means_unknown(self):
        self.assertIsNone(estimate_tokens_per_second(4.6, HardwareProfile(total_usable_ram_gb=16)))


class ScoreAndGradeTests(unittest.TestCase):
    def test_fast_fitting_model_grades_s(self):
        score = compute_score("can-run", 100, 8, 20)
        self.assertGreaterEqual(score, 85)
        self.assertEqual(score_to_grade(score, "can-run"), "S")

    def test_cannot_run_is_f(self):
        self.assertEqual(compute_score("cannot-run", None, 8), 0)
        self.assertEqual(score_to_grade(0, "cannot-run"), "F")

    def test_offload_caps_at_c(self):
        score = compute_score("can-run-slow", 60, 30, 50)
        self.assertIn(score_to_grade(score, "can-run-slow"), ("C", "D"))

    def test_evaluate_model_complete_bundles_everything(self):
        hw = HardwareProfile(
            total_usable_ram_gb=24, memory_bandwidth_gbs=273, is_apple_silicon=True
        )
        result = evaluate_model_complete(4.6, hw, 8)
        self.assertEqual(result["status"], "can-run")
        self.assertEqual(result["tokens_per_sec"], 39)  # 273/4.6*0.65
        self.assertEqual(result["mem_pct"], 19)
        self.assertIn(result["grade"], ("S", "A"))


class HardwareDbTests(unittest.TestCase):
    def test_gpu_db_longest_match_wins(self):
        self.assertEqual(match_gpu("NVIDIA GeForce RTX 4070 Ti SUPER")["bw"], 672)
        self.assertEqual(match_gpu("NVIDIA GeForce RTX 4070")["bw"], 504)
        self.assertIsNone(match_gpu("GPU Mystère 3000"))
        self.assertGreaterEqual(len(GPU_DB), 200)

    def test_apple_chip_match(self):
        key, entry = match_apple_chip("Apple M2 Pro — GPU intégré (mémoire unifiée)")
        self.assertEqual(key, "m2 pro")
        self.assertEqual(entry["bw"], 200)
        self.assertEqual(match_apple_chip("Apple M4 Max")[1]["bw"], 546)
        self.assertIsNone(match_apple_chip("Intel Core i7"))
        self.assertGreaterEqual(len(APPLE_DB), 18)

    def test_bandwidth_heuristic_fallbacks(self):
        self.assertEqual(bandwidth_heuristic("RTX inconnue", 24, "cuda"), 700.0)
        self.assertEqual(bandwidth_heuristic("GTX inconnue", 4, "cuda"), 150.0)
        self.assertEqual(bandwidth_heuristic("Puce inconnue", None, "metal"), 68.0)
        self.assertEqual(bandwidth_heuristic(None, None, "cpu"), 60.0)


class ProfileAdapterTests(unittest.TestCase):
    def test_apple_profile_from_lity_hardware(self):
        profile = profile_from_hardware(
            {
                "accelerator": "metal",
                "ram_gb": 24.0,
                "memory_bandwidth": 200.0,
                "budget_gb": 16.8,
            }
        )
        self.assertTrue(profile.is_apple_silicon)
        self.assertEqual(profile.total_usable_ram_gb, 24.0)
        self.assertEqual(profile.memory_bandwidth_gbs, 200.0)

    def test_budget_only_derives_ram(self):
        profile = profile_from_hardware({"accelerator": "metal", "budget_gb": 11.2})
        self.assertEqual(profile.total_usable_ram_gb, 16.0)

    def test_discrete_gpu_profile(self):
        profile = profile_from_hardware(
            {"accelerator": "cuda", "vram_gb": 8.0, "ram_gb": 32.0, "memory_bandwidth": 448.0}
        )
        self.assertEqual(profile.estimated_vram_gb, 8.0)
        self.assertEqual(profile.system_ram_gb, 32.0)
        self.assertFalse(profile.is_apple_silicon)


class RankModelsCanirunTests(unittest.TestCase):
    """rank_models carries the canirun-aligned fields, per quant."""

    def _hardware(self):
        return {
            "accelerator": "metal",
            "ram_gb": 24.0,
            "budget_gb": 16.8,
            "memory_bandwidth": 273.0,
        }

    def test_rows_carry_grades_scores_and_speed(self):
        rows = rank_models(self._hardware())
        by_name = {row["name"]: row for row in rows}
        small = by_name["llama3.1:8b"]
        self.assertIn(small["grade"], ("S", "A", "B"))
        self.assertGreater(small["score"], 0)
        self.assertEqual(small["status"], "can-run")
        self.assertEqual(small["tokens_per_sec"], 39)  # 273 / 4.6 * 0.65
        self.assertIn("tok/s", small["speed"])
        big = by_name["llama3.1:405b"]
        self.assertEqual(big["grade"], "F")
        self.assertEqual(big["verdict"], "trop_lourd")

    def test_rows_expose_per_quant_evaluations(self):
        rows = rank_models(self._hardware())
        row = next(r for r in rows if r["name"] == "qwen3:14b")
        self.assertEqual(len(row["quants"]), 7)
        q4 = next(q for q in row["quants"] if q["name"] == "Q4_K_M")
        self.assertIn("grade", q4)
        self.assertIn("tokens_per_sec", q4)
        # The best runnable quant is identified (higher bits = better quality).
        self.assertIsNotNone(row["best_quant"])
        f16 = next(q for q in row["quants"] if q["name"] == "F16")
        self.assertEqual(f16["status"], "cannot-run")  # 14B F16 ≈ 29 GB > 18 GB usable

    def test_sorted_best_first_with_single_recommendation(self):
        rows = rank_models(self._hardware())
        self.assertEqual(rows[0]["status"], "can-run")
        scores = [row["score"] for row in rows if row["status"] == "can-run"]
        self.assertEqual(scores, sorted(scores, reverse=True))
        recommended = [row for row in rows if row.get("recommended")]
        self.assertEqual(len(recommended), 1)

    def test_metadata_fields_present(self):
        rows = rank_models(self._hardware())
        row = next(r for r in rows if r["name"] == "qwen3:8b")
        self.assertEqual(row["display_name"], "Qwen 3 8B")
        self.assertEqual(row["context_length"], 131072)
        self.assertTrue(row["thinking"])
        self.assertEqual(row["license"], "Apache 2.0")


if __name__ == "__main__":
    unittest.main()
