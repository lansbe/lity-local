import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.rag.agentic import CorrectiveRetriever


def _chunks(*texts):
    return [{"path": f"f{i}.txt", "text": t} for i, t in enumerate(texts)]


class _ScriptedLLM:
    """Scripted generate_fn: grade calls pop from `grades`, rewrite calls pop
    from `rewrites`. Distinguished by the schema's properties (no model needed)."""

    def __init__(self, grades=None, rewrites=None):
        self.grades = list(grades or [])
        self.rewrites = list(rewrites or [])
        self.grade_calls = 0
        self.rewrite_calls = 0

    def __call__(self, prompt, schema):
        props = schema.get("properties", {})
        if "score" in props:
            self.grade_calls += 1
            return self.grades.pop(0) if self.grades else {"relevant": False, "score": 0}
        self.rewrite_calls += 1
        return self.rewrites.pop(0) if self.rewrites else {"query": "reformulation"}


class CorrectiveRetrieverTests(unittest.TestCase):
    def test_good_first_try_is_ok_and_grades_once(self):
        llm = _ScriptedLLM(grades=[{"relevant": True, "score": 5}])
        retr = CorrectiveRetriever(lambda q, k: _chunks("pertinent"), generate_fn=llm)
        result = retr.retrieve("question", top_k=3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["score"], 5)
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(llm.grade_calls, 1)
        self.assertEqual(llm.rewrite_calls, 0)  # no rewrite when first try is good

    def test_weak_then_corrected_after_rewrite(self):
        llm = _ScriptedLLM(
            grades=[{"relevant": False, "score": 1}, {"relevant": True, "score": 4}],
            rewrites=[{"query": "requête reformulée"}],
        )
        seen = []

        def retrieve_fn(query, k):
            seen.append(query)
            return _chunks("contenu")

        retr = CorrectiveRetriever(retrieve_fn, generate_fn=llm)
        result = retr.retrieve("requête initiale", top_k=3)
        self.assertEqual(result["status"], "corrected")
        self.assertEqual(result["query"], "requête reformulée")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(seen, ["requête initiale", "requête reformulée"])

    def test_weak_throughout_returns_best_chunks(self):
        llm = _ScriptedLLM(
            grades=[{"relevant": False, "score": 1}, {"relevant": False, "score": 2}],
            rewrites=[{"query": "autre"}],
        )
        retr = CorrectiveRetriever(lambda q, k: _chunks(q), generate_fn=llm)
        result = retr.retrieve("départ", top_k=3)
        self.assertEqual(result["status"], "weak")
        self.assertEqual(result["score"], 2)  # kept the higher-scoring attempt
        self.assertEqual(result["query"], "autre")

    def test_no_grader_is_passthrough(self):
        retr = CorrectiveRetriever(lambda q, k: _chunks("a", "b"), generate_fn=None)
        result = retr.retrieve("q", top_k=3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(result["attempts"], 1)

    def test_empty_retrieval_is_weak(self):
        llm = _ScriptedLLM(rewrites=[{"query": "x"}])
        retr = CorrectiveRetriever(lambda q, k: [], generate_fn=llm)
        result = retr.retrieve("q", top_k=3)
        self.assertEqual(result["status"], "weak")
        self.assertEqual(result["score"], 0)

    def test_empty_retrieval_without_grader_still_weak(self):
        retr = CorrectiveRetriever(lambda q, k: [], generate_fn=None)
        result = retr.retrieve("q", top_k=3)
        self.assertEqual(result["status"], "weak")

    def test_relevant_true_without_numeric_score_passes(self):
        llm = _ScriptedLLM(grades=[{"relevant": True}])  # no numeric score field
        retr = CorrectiveRetriever(lambda q, k: _chunks("x"), generate_fn=llm)
        result = retr.retrieve("q", top_k=3)
        self.assertEqual(result["status"], "ok")

    def test_bool_score_is_rejected_falls_back_to_relevant(self):
        # score=True must not be read as int 1 (bool is an int subclass).
        llm = _ScriptedLLM(grades=[{"relevant": True, "score": True}])
        retr = CorrectiveRetriever(lambda q, k: _chunks("x"), generate_fn=llm)
        result = retr.retrieve("q", top_k=3)
        self.assertEqual(result["status"], "ok")  # relevant True → min_score, not 1

    def test_grader_exception_does_not_break_retrieval(self):
        def boom(prompt, schema):
            raise RuntimeError("model down")

        retr = CorrectiveRetriever(lambda q, k: _chunks("a"), generate_fn=boom)
        result = retr.retrieve("q", top_k=3)
        self.assertEqual(result["status"], "ok")  # grader failure → trust retrieval

    def test_rewrite_returning_same_query_stops_early(self):
        llm = _ScriptedLLM(
            grades=[{"relevant": False, "score": 1}],
            rewrites=[{"query": "départ"}],  # identical to input → no point retrying
        )
        retr = CorrectiveRetriever(lambda q, k: _chunks("x"), generate_fn=llm)
        result = retr.retrieve("départ", top_k=3)
        self.assertEqual(result["status"], "weak")
        self.assertEqual(result["attempts"], 1)  # did not retry on identical rewrite
        self.assertEqual(llm.grade_calls, 1)


if __name__ == "__main__":
    unittest.main()
