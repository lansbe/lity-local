from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A reranker scores (query, [texts]) -> [scores] aligned with the input order
# (higher = more relevant). Implementations live in rerank.py and are optional.
Reranker = Callable[[str, "list[str]"], "list[float]"]


def reciprocal_rank_fusion(rankings: list[list[Any]], k: int = 60) -> list[Any]:
    """Fuse several ranked id-lists into one via Reciprocal Rank Fusion (RRF).

    Each input is a list of ids ordered best→worst (e.g. a dense/embedding
    ranking and a BM25 ranking). RRF score(id) = Σ 1/(k + rank) over the lists in
    which it appears. Returns unique ids ordered by fused score (high→low). It is
    rank-based, so it is robust to the incomparable score scales of cosine
    similarity and BM25 — the reason it is the durable default for hybrid search.
    """
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
