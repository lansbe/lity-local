import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.core.image_model_advisor import rank_image_models


class RankImageModelsTests(unittest.TestCase):
    def test_ranks_image_models_for_current_device(self):
        rows = rank_image_models(
            {
                "accelerator": "metal",
                "ram_gb": 16.0,
                "budget_gb": 11.2,
                "memory_bandwidth": 120.0,
            }
        )

        self.assertGreaterEqual(len(rows), 6)
        self.assertEqual(len([row for row in rows if row.get("recommended")]), 1)
        self.assertLess(
            rows.index(next(row for row in rows if row["name"] == "sd15")),
            rows.index(next(row for row in rows if row["name"] == "flux2-dev")),
        )

        by_name = {row["name"]: row for row in rows}
        self.assertEqual(by_name["sdxl-base"]["backend"], "automatic1111")
        # FLUX.2 klein / Z-Image are now one-click MLX (mflux) entries.
        self.assertEqual(by_name["flux2-klein-4b"]["backend"], "mlx")
        self.assertEqual(by_name["z-image-turbo"]["backend"], "mlx")
        self.assertEqual(by_name["z-image-turbo"]["mlx"]["command"], "mflux-generate-z-image-turbo")
        self.assertEqual(by_name["flux2-klein-4b"]["mlx"]["command"], "mflux-generate-flux2")
        self.assertEqual(by_name["flux1-schnell-mlx"]["mlx"]["model_arg"], "model")
        self.assertEqual(by_name["z-image-turbo"]["mlx"]["quantize"], 4)
        self.assertIn(by_name["z-image-turbo"]["license"], ("Apache 2.0", "apache-2.0"))
        self.assertIn(by_name["flux2-dev"]["verdict"], ("limite", "trop_lourd"))


if __name__ == "__main__":
    unittest.main()
