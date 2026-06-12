from __future__ import annotations

from collections.abc import Callable
from typing import Any

RetrieveFn = Callable[[str, int], "list[dict[str, Any]]"]
# generate_fn(prompt, schema) -> parsed JSON object (or None). The LLM transport
# is injected so the corrective logic below is fully unit-testable WITHOUT a model.
GenerateFn = Callable[[str, "dict[str, Any]"], "dict[str, Any] | None"]

# Constrained-decoding schemas (Ollama ``format=``): force a clean, parseable
# verdict instead of fishing JSON out of free text.
_GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
    },
    "required": ["relevant", "score"],
}
_REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

_GRADE_PROMPT = (
    "Tu évalues si des extraits récupérés permettent de répondre à une question.\n"
    "Question : {query}\n\n"
    "Extraits :\n{chunks}\n\n"
    "Donne un score de pertinence de 0 (totalement hors-sujet) à 5 (répond "
    "directement), et dis si au moins un extrait est pertinent. Sois strict."
)
_REWRITE_PROMPT = (
    "Une recherche documentaire locale n'a pas donné d'extraits pertinents. "
    "Reformule la requête pour mieux retrouver l'information : ajoute des "
    "synonymes ou des termes-clés probables, garde-la courte, en français.\n"
    "Requête initiale : {query}\n"
    "Reformulation :"
)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…"


class CorrectiveRetriever:
    """CRAG-light (Corrective RAG): grade what retrieval returned; if it's weak,
    rewrite the query once and retry; if it's still weak, flag a web fallback.

    Rationale: dense+BM25 retrieval can return confidently-irrelevant chunks
    (wrong file, stale section). Feeding those to the model invites grounded-
    sounding hallucination. A cheap relevance gate + a single query rewrite
    recovers recall on bad phrasings, and a clear "weak" signal lets the caller
    decide to search the web instead.

    100% local: grading/rewriting use the SAME Ollama model via constrained
    decoding (``format=schema``), passed in as ``generate_fn``. When no grader is
    available the retriever is a pure pass-through — it never makes plain
    retrieval worse. Bounded cost: 1 grade call for good retrievals; at most
    ``1 + 2 * max_rewrites`` small calls in the worst case.
    """

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        generate_fn: GenerateFn | None = None,
        *,
        min_score: int = 3,
        max_rewrites: int = 1,
        grade_chunks: int = 4,
    ):
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.min_score = min_score
        self.max_rewrites = max_rewrites
        self.grade_chunks = grade_chunks

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Return ``{chunks, status, query, score, attempts}``.

        status: ``ok`` (first try good), ``corrected`` (a rewrite fixed it),
        ``weak`` (still poor — caller may fall back to the web).
        """
        chunks = list(self.retrieve_fn(query, top_k) or [])
        score = self._grade(query, chunks)
        attempts = 1
        if score >= self.min_score:
            return self._result(chunks, "ok", query, score, attempts)

        best = (chunks, score, query)
        current = query
        for _ in range(self.max_rewrites):
            new_query = self._rewrite(current)
            if not new_query or new_query == current:
                break
            current = new_query
            retried = list(self.retrieve_fn(current, top_k) or [])
            rscore = self._grade(current, retried)
            attempts += 1
            if rscore > best[1]:
                best = (retried, rscore, current)
            if rscore >= self.min_score:
                return self._result(retried, "corrected", current, rscore, attempts)

        return self._result(best[0], "weak", best[2], best[1], attempts)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _result(chunks, status, query, score, attempts) -> dict[str, Any]:
        return {
            "chunks": chunks,
            "status": status,
            "query": query,
            "score": score,
            "attempts": attempts,
        }

    def _grade(self, query: str, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0  # nothing retrieved is unambiguously weak
        if self.generate_fn is None:
            return self.min_score  # no grader → trust retrieval, never make it worse
        joined = "\n---\n".join(
            _truncate(str(chunk.get("text", "")).strip(), 500)
            for chunk in chunks[: self.grade_chunks]
        )
        prompt = _GRADE_PROMPT.format(query=query, chunks=joined)
        try:
            data = self.generate_fn(prompt, _GRADE_SCHEMA) or {}
        except Exception:
            return self.min_score
        raw = data.get("score")
        if isinstance(raw, bool):  # bool is an int subclass — reject it as a score
            raw = None
        if isinstance(raw, int):
            return max(0, min(5, raw))
        return self.min_score if data.get("relevant") else 0

    def _rewrite(self, query: str) -> str | None:
        if self.generate_fn is None:
            return None
        try:
            data = self.generate_fn(_REWRITE_PROMPT.format(query=query), _REWRITE_SCHEMA) or {}
        except Exception:
            return None
        new_query = str(data.get("query", "")).strip()
        return new_query or None
