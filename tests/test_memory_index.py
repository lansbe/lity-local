import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.memory.memory_index import MemoryIndexer
from lity.services.rag.vector_index import VectorIndex

VOCAB = ["alpha", "beta", "gamma", "delta", "python", "réunion"]


def fake_embed(text):
    lowered = text.lower()
    vector = [float(lowered.count(word)) for word in VOCAB]
    return vector if any(vector) else None


class MemoryIndexerTests(unittest.TestCase):
    def test_index_is_incremental_and_skips_short_messages(self):
        indexer = MemoryIndexer(fake_embed, VectorIndex())
        messages = [
            {"role": "user", "content": "parle-moi de python alpha"},  # indexed
            {"role": "assistant", "content": "ok"},  # too short → skipped
            {"role": "user", "content": "la réunion beta est demain"},  # indexed
        ]
        self.assertEqual(indexer.index_conversation("c1", "Projet", messages), 2)
        # Re-indexing the same messages adds nothing (incremental, by message id).
        self.assertEqual(indexer.index_conversation("c1", "Projet", messages), 0)
        # A newly appended message is picked up.
        messages.append({"role": "user", "content": "et gamma aussi pour la réunion"})
        self.assertEqual(indexer.index_conversation("c1", "Projet", messages), 1)

    def test_retrieve_excludes_active_conversation(self):
        indexer = MemoryIndexer(fake_embed, VectorIndex())
        indexer.index_conversation(
            "past", "Vieux fil", [{"role": "user", "content": "la réunion gamma est lundi matin"}]
        )
        indexer.index_conversation(
            "current", "Fil actif", [{"role": "user", "content": "réunion gamma de suivi hebdo"}]
        )
        hits = indexer.retrieve(
            "quand est la réunion gamma", top_k=3, exclude_conversation_id="current"
        )
        self.assertTrue(hits)
        self.assertTrue(all(hit["conversation_id"] == "past" for hit in hits))

    def test_retrieve_empty_when_only_active_conversation(self):
        indexer = MemoryIndexer(fake_embed, VectorIndex())
        indexer.index_conversation(
            "current", "Fil actif", [{"role": "user", "content": "python alpha beta gamma"}]
        )
        self.assertEqual(indexer.retrieve("python", exclude_conversation_id="current"), [])

    def test_reranker_reorders_recall(self):
        def reverse_reranker(_query, texts):
            return [float(i) for i in range(len(texts))]

        indexer = MemoryIndexer(fake_embed, VectorIndex(), reranker=reverse_reranker)
        indexer.index_conversation(
            "a", "A", [{"role": "user", "content": "alpha beta gamma delta"}]
        )
        indexer.index_conversation("b", "B", [{"role": "user", "content": "alpha beta python"}])
        hits = indexer.retrieve("alpha beta", top_k=2)
        self.assertEqual(len(hits), 2)  # both conversations recalled
        self.assertEqual(hits[0]["conversation_id"], "a")  # reverse reranker flips fused order


if __name__ == "__main__":
    unittest.main()
