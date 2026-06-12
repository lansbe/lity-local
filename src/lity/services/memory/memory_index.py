from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lity.services.rag.bm25 import BM25
from lity.services.rag.hybrid import Reranker, reciprocal_rank_fusion
from lity.services.rag.vector_index import VectorIndex, cosine_similarity

EmbedFn = Callable[[str], "list[float] | None"]

# Skip trivial messages ("ok", "merci", "bonjour") — they add noise, not memory.
_MIN_LEN = 16


class MemoryIndexer:
    """Cross-session memory: a hybrid index over PAST conversation messages.

    Same retrieval stack as the project RAG (dense embedding + BM25 → RRF →
    optional cross-encoder rerank), but the corpus is the user's own
    conversations. Indexing is INCREMENTAL (only new messages are embedded) and
    retrieval EXCLUDES the active conversation, since that one is already in the
    live context. The embedding function and reranker are injected, so this is
    fully testable without a running model.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        index: VectorIndex,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
    ):
        self.embed = embed_fn
        self.index = index
        self.reranker = reranker
        self.candidate_k = candidate_k

    def index_conversation(
        self, conversation_id: str, title: str, messages: list[dict[str, Any]]
    ) -> int:
        """Embed any not-yet-indexed messages of a conversation. Returns # added."""
        existing = {
            entry.get("id")
            for entry in self.index.snapshot()
            if entry.get("conversation_id") == conversation_id
        }
        new_entries: list[dict[str, Any]] = []
        for position, message in enumerate(messages):
            entry_id = f"{conversation_id}#{position}"
            if entry_id in existing:
                continue
            text = str(message.get("content", "")).strip()
            if len(text) < _MIN_LEN:
                continue
            vector = self.embed(text)
            if not vector:
                continue
            new_entries.append(
                {
                    "id": entry_id,
                    "conversation_id": conversation_id,
                    "title": title,
                    "role": message.get("role", ""),
                    "text": text,
                    "vector": list(vector),
                }
            )
        if new_entries:
            self.index.add(new_entries)  # locked extend + atomic save
        return len(new_entries)

    def retrieve(
        self, query: str, top_k: int = 3, exclude_conversation_id: str | None = None
    ) -> list[dict[str, Any]]:
        entries = [
            entry
            for entry in self.index.snapshot()
            if entry.get("conversation_id") != exclude_conversation_id
        ]
        if not entries:
            return []
        by_id = {entry["id"]: entry for entry in entries}

        rankings: list[list[Any]] = []
        vector = self.embed(query)
        if vector:
            scored = [
                (cosine_similarity(vector, entry.get("vector", [])), entry)
                for entry in entries
                if entry.get("vector")
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            rankings.append(
                [entry["id"] for score, entry in scored[: self.candidate_k] if score > 0]
            )

        bm25 = BM25([entry.get("text", "") for entry in entries])
        lexical = bm25.search(query, top_k=self.candidate_k)
        rankings.append([entries[index].get("id") for _score, index in lexical])

        fused = reciprocal_rank_fusion(rankings)
        candidates = [by_id[doc_id] for doc_id in fused if doc_id in by_id][: self.candidate_k]
        if not candidates:
            return []
        return self._rerank(query, candidates)[:top_k]

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.reranker or len(candidates) <= 1:
            return candidates
        try:
            scores = self.reranker(query, [candidate.get("text", "") for candidate in candidates])
        except Exception:
            return candidates
        if not scores or len(scores) != len(candidates):
            return candidates
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order]
