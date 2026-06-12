import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.rag.chunking import chunk_code, chunk_python, chunk_text
from lity.services.rag.indexer import ProjectIndexer
from lity.services.rag.vector_index import VectorIndex

PYTHON_SOURCE = '''"""Module docstring."""

import os


def small_one():
    return 1


def small_two():
    return 2


class Big:
    attribute = "x"

    def method_a(self):
        return "a" * 3

    def method_b(self):
        return "b" * 3
'''


def counting_embed(counter):
    def embed(text):
        counter.append(text)
        return [float(len(text)), 1.0]

    return embed


class MutableFiles:
    def __init__(self, files):
        self.files = dict(files)

    def get_available_files(self, recursive=False):
        return list(self.files.keys())

    def read_file_safe(self, path, max_chars=20_000):
        if path in self.files:
            return True, self.files[path]
        return False, "introuvable"


class ChunkPythonTests(unittest.TestCase):
    def test_functions_stay_whole(self):
        chunks = chunk_python(PYTHON_SOURCE, max_chars=120)
        joined = "\n--\n".join(chunks)
        self.assertIn("def small_one", joined)
        # A definition is never split mid-body at these sizes.
        for chunk in chunks:
            if "def small_one" in chunk:
                self.assertIn("return 1", chunk)

    def test_oversized_class_splits_per_method(self):
        chunks = chunk_python(PYTHON_SOURCE, max_chars=80)
        method_chunks = [c for c in chunks if "def method_a" in c]
        self.assertTrue(method_chunks)
        self.assertIn('return "a" * 3', method_chunks[0])

    def test_small_neighbours_merge(self):
        chunks = chunk_python("def a():\n    return 1\n\n\ndef b():\n    return 2\n", 1000)
        self.assertEqual(len(chunks), 1)  # both fit one chunk

    def test_broken_python_falls_back_to_windows(self):
        broken = "def oops(:\n" + "x" * 3000
        self.assertEqual(chunk_python(broken, 1000), chunk_text(broken, 1000))

    def test_chunk_code_dispatches_by_extension(self):
        text = "def f():\n    return 1\n"
        self.assertEqual(chunk_code("a.py", text), chunk_python(text))
        self.assertEqual(chunk_code("a.txt", text), chunk_text(text))


class IncrementalReindexTests(unittest.TestCase):
    def _indexer(self, files, counter):
        return ProjectIndexer(files, counting_embed(counter), VectorIndex())

    def test_unchanged_files_are_not_reembedded(self):
        files = MutableFiles({"a.txt": "contenu alpha", "b.txt": "contenu beta"})
        counter = []
        indexer = self._indexer(files, counter)

        first = indexer.reindex()
        self.assertEqual(first["files"], 2)
        embed_calls_after_first = len(counter)

        second = indexer.reindex()
        self.assertEqual(second["unchanged"], 2)
        self.assertEqual(second["files"], 0)
        self.assertEqual(len(counter), embed_calls_after_first)  # zero new embeddings

    def test_changed_file_is_reindexed_without_duplicates(self):
        files = MutableFiles({"a.txt": "version un", "b.txt": "stable"})
        counter = []
        indexer = self._indexer(files, counter)
        indexer.reindex()

        files.files["a.txt"] = "version deux"
        stats = indexer.reindex()

        self.assertEqual(stats["files"], 1)  # only the changed file
        self.assertEqual(stats["unchanged"], 1)
        texts = [entry["text"] for entry in indexer.index.snapshot()]
        self.assertIn("version deux", texts)
        self.assertNotIn("version un", texts)  # stale chunks evicted
        self.assertEqual(len(texts), 2)  # no duplicates

    def test_deleted_file_drops_out_of_the_index(self):
        files = MutableFiles({"a.txt": "alpha", "b.txt": "beta"})
        counter = []
        indexer = self._indexer(files, counter)
        indexer.reindex()

        del files.files["b.txt"]
        stats = indexer.reindex()

        self.assertEqual(stats["removed"], 1)
        paths = {entry["path"] for entry in indexer.index.snapshot()}
        self.assertEqual(paths, {"a.txt"})

    def test_full_reindex_still_works(self):
        files = MutableFiles({"a.txt": "alpha"})
        counter = []
        indexer = self._indexer(files, counter)
        indexer.reindex()
        stats = indexer.reindex(full=True)
        self.assertEqual(stats["files"], 1)
        self.assertEqual(indexer.index.count(), 1)


if __name__ == "__main__":
    unittest.main()
