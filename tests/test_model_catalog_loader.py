import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.core.model_catalog_loader import load_model_catalog


class ModelCatalogLoaderTests(unittest.TestCase):
    def test_model_catalog_resource_loads_with_computed_quants(self):
        rows = load_model_catalog()

        self.assertGreaterEqual(len(rows), 20)
        self.assertTrue(all("quants" in row for row in rows))
        self.assertTrue(any(row.get("ollama_id") == "nomic-embed-text" for row in rows))
        self.assertTrue(any(row.get("kind") == "chat" and row.get("tools") for row in rows))


if __name__ == "__main__":
    unittest.main()
