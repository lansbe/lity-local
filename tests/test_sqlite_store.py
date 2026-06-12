import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.memory.memory_index import MemoryIndexer
from lity.services.rag.sqlite_store import SqliteVectorStore

VOCAB = ["alpha", "beta", "gamma", "delta"]


def fake_embed(text):
    lowered = text.lower()
    vector = [float(lowered.count(word)) for word in VOCAB]
    return vector if any(vector) else None


class SqliteVectorStoreTests(unittest.TestCase):
    def test_add_count_snapshot_and_search_round_trip(self):
        store = SqliteVectorStore()
        store.add(
            [
                {"id": "a", "path": "a.txt", "text": "A", "vector": [1.0, 0.0]},
                {"id": "b", "path": "b.txt", "text": "B", "vector": [0.0, 1.0]},
            ]
        )
        self.assertEqual(store.count(), 2)
        self.assertEqual({entry["id"] for entry in store.snapshot()}, {"a", "b"})
        results = store.search([1.0, 0.0], top_k=1)
        self.assertEqual(results[0][1]["id"], "a")
        self.assertEqual(results[0][1]["vector"], [1.0, 0.0])  # vector round-trips

    def test_upsert_replaces_without_duplicating(self):
        store = SqliteVectorStore()
        store.add([{"id": "x", "text": "v1", "vector": [1.0]}])
        store.add([{"id": "x", "text": "v2", "vector": [2.0]}])
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.snapshot()[0]["text"], "v2")

    def test_persists_across_instances_and_clears(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "index.db"
            first = SqliteVectorStore(db)
            first.add([{"id": "a", "text": "hello world", "vector": [1.0]}])
            self.assertEqual(first.count(), 1)
            # A fresh connection to the same file sees the committed row.
            self.assertEqual(SqliteVectorStore(db).count(), 1)
            SqliteVectorStore(db).clear()
            self.assertEqual(SqliteVectorStore(db).count(), 0)

    def test_drop_in_with_memory_indexer(self):
        indexer = MemoryIndexer(fake_embed, SqliteVectorStore())
        msgs = [{"role": "user", "content": "la réunion alpha est prévue lundi"}]
        self.assertEqual(indexer.index_conversation("past", "Vieux fil", msgs), 1)
        self.assertEqual(indexer.index_conversation("past", "Vieux fil", msgs), 0)  # incremental
        hits = indexer.retrieve("réunion alpha", top_k=1, exclude_conversation_id="other")
        self.assertEqual(hits[0]["conversation_id"], "past")

    def test_concurrent_add_and_read_do_not_race(self):
        store = SqliteVectorStore()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(200):
                    store.add([{"id": f"w{i}", "text": "x", "vector": [float(i), 1.0]}])
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            for _ in range(200):
                store.search([1.0, 1.0], top_k=3)
                store.snapshot()
                store.count()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)
        thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(store.count(), 200)


if __name__ == "__main__":
    unittest.main()
