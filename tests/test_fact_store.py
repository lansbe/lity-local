import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app._controller_retrieval import RetrievalMixin
from lity.services.memory.fact_store import FactStore
from lity.services.rag.sqlite_store import SqliteVectorStore

# Deterministic bag-of-words "embedding": cosine then reflects word overlap, so
# recall ordering is predictable WITHOUT a model.
_VOCAB = ["alex", "nom", "projet", "lity", "timeout", "secondes", "postgres", "base", "langue"]


def fake_embed(text):
    if text is None:
        return None
    low = text.lower()
    return [1.0 if word in low else 0.0 for word in _VOCAB]


def new_store(embed=fake_embed, **kw):
    return FactStore(embed, SqliteVectorStore(None), **kw)  # :memory:


class FactStoreTests(unittest.TestCase):
    def test_index_and_count(self):
        store = new_store()
        added = store.index_facts(
            {"db": "Le projet utilise PostgreSQL 16 comme base", "prenom": "Le prénom est Alex"}
        )
        self.assertEqual(added, 2)
        self.assertEqual(store.store.count(), 2)

    def test_index_is_idempotent(self):
        store = new_store()
        store.index_facts({"db": "Le projet utilise PostgreSQL 16"})
        again = store.index_facts({"db": "Le projet utilise PostgreSQL 16"})
        self.assertEqual(again, 0)  # unchanged text → not re-embedded
        self.assertEqual(store.store.count(), 1)

    def test_updating_a_fact_replaces_it(self):
        store = new_store()
        store.index_facts({"db": "Le projet utilise PostgreSQL"})
        store.index_facts({"db": "Le projet utilise PostgreSQL 16 et une base"})
        self.assertEqual(store.store.count(), 1)  # same id replaced, not duplicated

    def test_recall_returns_relevant_fact(self):
        store = new_store()
        store.index_facts(
            {
                "db": "Le projet utilise PostgreSQL 16 comme base",
                "prenom": "Le prénom est Alex",
            }
        )
        hits = store.recall("quelle base de données pour le projet ?")
        self.assertTrue(hits)
        self.assertIn("PostgreSQL", hits[0]["text"])  # most relevant first
        self.assertEqual(hits[0]["key"], "db")

    def test_recall_drops_distractors_below_floor(self):
        store = new_store()
        store.index_facts({"db": "Le projet utilise PostgreSQL 16 comme base"})
        # "langue" shares no vocab with the stored fact → score 0 < min_score.
        self.assertEqual(store.recall("quelle est ta langue ?"), [])

    def test_recall_empty_without_embeddings(self):
        store = new_store(embed=lambda _t: None)
        store.store.add([{"id": "fact::x", "title": "x", "text": "fait", "vector": [1.0]}])
        self.assertEqual(store.recall("peu importe"), [])

    def test_recall_empty_when_store_empty(self):
        self.assertEqual(new_store().recall("quoi que ce soit"), [])

    def test_add_fact_single(self):
        store = new_store()
        self.assertTrue(store.add_fact("projet", "Le projet s'appelle Lity"))
        self.assertEqual(store.store.count(), 1)

    def test_fact_text_handles_dict_record(self):
        # Facts persisted on disk are {value, count, ...}; the store reads .value.
        store = new_store()
        store.index_facts({"db": {"value": "Le projet utilise PostgreSQL", "count": 2}})
        self.assertEqual(store.store.count(), 1)
        self.assertTrue(store.recall("projet postgres ?"))

    def test_blank_value_skipped(self):
        store = new_store()
        self.assertEqual(store.index_facts({"x": "   ", "": "vide"}), 0)


class RetrieveFactsContextTests(unittest.TestCase):
    """Controller-side wiring: gating, the long_term_facts filter, and the
    history-aware query feeding recall."""

    def _stub(self, *, fact_store=None, settings=None, engine=None, context=None):
        memory = SimpleNamespace(
            get_context=lambda: context or [{"role": "user", "content": "quelle base ?"}],
            active_conversation_id="c1",
            get_memory=lambda: {"facts": {}},
        )
        # A real RetrievalMixin instance: _retrieve_facts_context calls sibling
        # methods (_ensure_fact_store, _passive_retrieval_query), so a bare
        # SimpleNamespace would not resolve them.
        obj = RetrievalMixin()
        obj.settings = settings
        obj.engine = (
            engine if engine is not None else SimpleNamespace(embed=lambda t, m=None: [1.0])
        )
        obj.memory = memory
        obj.paths = SimpleNamespace()
        obj._fact_store = fact_store
        obj._embedding_model = lambda: "m"
        return obj

    def test_disabled_returns_empty(self):
        stub = self._stub(settings={"durable_facts": False})
        self.assertEqual(RetrievalMixin._retrieve_facts_context(stub), "")

    def test_formats_recalled_facts(self):
        fake = SimpleNamespace(
            store=SimpleNamespace(count=lambda: 1),
            recall=lambda q, top_k=None: [
                {"key": "db", "text": "Le projet utilise PostgreSQL 16", "score": 0.7}
            ],
        )
        stub = self._stub(fact_store=fake)
        out = RetrievalMixin._retrieve_facts_context(stub)
        self.assertIn("FAITS MÉMORISÉS PERTINENTS", out)
        self.assertIn("PostgreSQL 16", out)

    def test_no_hits_returns_empty(self):
        fake = SimpleNamespace(
            store=SimpleNamespace(count=lambda: 3), recall=lambda q, top_k=None: []
        )
        self.assertEqual(RetrievalMixin._retrieve_facts_context(self._stub(fact_store=fake)), "")

    def test_index_fact_to_store_ignores_non_long_term(self):
        calls = []
        fake = SimpleNamespace(add_fact=lambda k, v: calls.append((k, v)) or True)
        stub = self._stub(fact_store=fake)
        # user_profile / assistant_profile facts must NOT go to the semantic store.
        RetrievalMixin._index_fact_to_store(
            stub, {"categorie": "user_profile", "cle": "nom", "valeur": "Alex"}
        )
        self.assertEqual(calls, [])

    def test_index_fact_to_store_indexes_long_term(self):
        calls = []
        fake = SimpleNamespace(add_fact=lambda k, v: calls.append((k, v)) or True)
        stub = self._stub(fact_store=fake)
        RetrievalMixin._index_fact_to_store(
            stub, {"categorie": "long_term_facts", "cle": "db", "valeur": "PostgreSQL 16"}
        )
        self.assertEqual(calls, [("db", "PostgreSQL 16")])


if __name__ == "__main__":
    unittest.main()
