import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.rag.bm25 import BM25, tokenize
from lity.services.rag.hybrid import reciprocal_rank_fusion
from lity.services.rag.indexer import ProjectIndexer
from lity.services.rag.vector_index import VectorIndex

VOCAB = ["alpha", "beta", "gamma", "delta"]


def fake_embed(text):
    """Embeds only the toy VOCAB → mimics an embedder blind to literal tokens."""
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


class BM25Tests(unittest.TestCase):
    def test_tokenize_is_unicode_and_lowercase(self):
        self.assertEqual(tokenize("Été, CODE-42!"), ["été", "code", "42"])

    def test_ranks_document_with_query_term_first(self):
        bm25 = BM25(["le chat dort", "le chien court vite", "rien ici"])
        results = bm25.search("chien", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0][1], 1)  # the "chien" document

    def test_rare_term_outranks_common_term(self):
        bm25 = BM25(["commun commun commun", "commun rare"])
        self.assertEqual(bm25.search("rare")[0][1], 1)  # rare term → doc 1 wins

    def test_empty_corpus_returns_nothing(self):
        self.assertEqual(BM25([]).search("x"), [])
        self.assertEqual(BM25(["", "  "]).search("x"), [])


class RRFTests(unittest.TestCase):
    def test_consensus_id_wins_and_set_is_deduped(self):
        fused = reciprocal_rank_fusion([["b", "a", "c"], ["b", "a", "d"]])
        self.assertEqual(fused[0], "b")  # top of both lists
        self.assertEqual(set(fused), {"a", "b", "c", "d"})

    def test_single_ranking_passthrough(self):
        self.assertEqual(reciprocal_rank_fusion([["x", "y"]]), ["x", "y"])

    def test_empty(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])


class HybridRetrievalTests(unittest.TestCase):
    def test_lexical_recall_when_dense_cannot_embed_query(self):
        # The query term is outside the embedder's vocab → no dense vector, so
        # dense search yields nothing; BM25 must still find the literal token.
        files = FakeFiles({"a.txt": "alpha alpha alpha", "b.txt": "beta xylophone42 reference"})
        indexer = ProjectIndexer(files, fake_embed, VectorIndex())
        stats = indexer.reindex()
        self.assertEqual(stats["chunks"], 2)  # both chunks embedded (have vocab)

        self.assertIsNone(fake_embed("xylophone42"))  # dense is blind here
        hits = indexer.retrieve("xylophone42", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["path"], "b.txt")

    def test_dense_still_used_when_query_embeds(self):
        files = FakeFiles({"a.txt": "alpha alpha alpha", "b.txt": "beta gamma"})
        indexer = ProjectIndexer(files, fake_embed, VectorIndex())
        indexer.reindex()
        hits = indexer.retrieve("parle moi d'alpha", top_k=1)
        self.assertEqual(hits[0]["path"], "a.txt")

    def test_reranker_reorders_candidates(self):
        files = FakeFiles({"a.txt": "beta beta", "b.txt": "beta"})

        def reverse_reranker(_query, texts):
            # Later candidates score higher → flips the fused order.
            return [float(i) for i in range(len(texts))]

        indexer = ProjectIndexer(files, fake_embed, VectorIndex(), reranker=reverse_reranker)
        indexer.reindex()
        hits = indexer.retrieve("beta", top_k=2)
        self.assertEqual([hit["path"] for hit in hits], ["b.txt", "a.txt"])

    def test_reranker_failure_falls_back_to_fused_order(self):
        files = FakeFiles({"a.txt": "beta beta", "b.txt": "beta"})

        def broken_reranker(_query, _texts):
            raise RuntimeError("model exploded")

        indexer = ProjectIndexer(files, fake_embed, VectorIndex(), reranker=broken_reranker)
        indexer.reindex()
        hits = indexer.retrieve("beta", top_k=2)
        self.assertEqual(len(hits), 2)  # still returns results, in fused order


if __name__ == "__main__":
    unittest.main()
