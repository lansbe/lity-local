import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.rag.indexer import ProjectIndexer, chunk_text
from lity.services.rag.vector_index import VectorIndex, cosine_similarity

VOCAB = ["alpha", "beta", "gamma", "delta"]


def fake_embed(text):
    lowered = text.lower()
    vector = [float(lowered.count(word)) for word in VOCAB]
    return vector if any(vector) else None


class FakeFiles:
    def __init__(self, files):
        self._files = files

    def get_available_files(self, recursive=False):
        return list(self._files.keys())

    def read_file_safe(self, path, max_chars=20_000):
        if path in self._files:
            return True, self._files[path]
        return False, "introuvable"


class ChunkAndSimilarityTests(unittest.TestCase):
    def test_chunk_text_splits_long_text_with_overlap(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("court"), ["court"])
        chunks = chunk_text("x" * 2500, max_chars=1000, overlap=100)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(chunk for chunk in chunks))

    def test_cosine_similarity_bounds(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine_similarity([], [1]), 0.0)


class VectorIndexTests(unittest.TestCase):
    def test_add_search_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            index = VectorIndex(path)
            index.add(
                [
                    {
                        "id": "a",
                        "path": "a.txt",
                        "chunk_index": 0,
                        "text": "A",
                        "vector": [1.0, 0.0],
                    },
                    {
                        "id": "b",
                        "path": "b.txt",
                        "chunk_index": 0,
                        "text": "B",
                        "vector": [0.0, 1.0],
                    },
                ]
            )
            results = index.search([1.0, 0.0], top_k=1)
            self.assertEqual(results[0][1]["id"], "a")

            reloaded = VectorIndex(path)
            self.assertEqual(reloaded.count(), 2)

            reloaded.clear()
            self.assertEqual(reloaded.count(), 0)
            self.assertEqual(VectorIndex(path).count(), 0)


class ProjectIndexerTests(unittest.TestCase):
    def test_reindex_and_retrieve(self):
        files = FakeFiles({"a.txt": "alpha alpha alpha", "b.txt": "beta gamma"})
        index = VectorIndex()
        indexer = ProjectIndexer(files, fake_embed, index)

        stats = indexer.reindex()
        self.assertEqual(stats["files"], 2)
        self.assertEqual(stats["chunks"], 2)

        hits = indexer.retrieve("parle moi d'alpha", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["path"], "a.txt")

    def test_reindex_without_embeddings_indexes_nothing(self):
        files = FakeFiles({"a.txt": "contenu sans vocabulaire"})
        index = VectorIndex()
        indexer = ProjectIndexer(files, lambda _text: None, index)

        stats = indexer.reindex()
        self.assertEqual(stats["chunks"], 0)
        self.assertEqual(indexer.retrieve("alpha"), [])


class VectorIndexConcurrencyTests(unittest.TestCase):
    def test_concurrent_add_and_read_do_not_race(self):
        # A background writer mutates the store while the main thread reads it.
        # Without the lock + snapshot(), search/iteration races ("list changed
        # size during iteration"). With them, it must be clean and consistent.
        index = VectorIndex()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(300):
                    index.add([{"id": f"w{i}", "text": "x", "vector": [float(i), 1.0]}])
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            for _ in range(300):
                index.search([1.0, 1.0], top_k=3)
                index.snapshot()
                index.count()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)
        thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(index.count(), 300)


if __name__ == "__main__":
    unittest.main()
