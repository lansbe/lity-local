from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from lity.services.rag.bm25 import BM25
from lity.services.rag.chunking import chunk_code, chunk_text
from lity.services.rag.hybrid import Reranker, reciprocal_rank_fusion
from lity.services.rag.vector_index import VectorIndex

EmbedFn = Callable[[str], "list[float] | None"]

# chunk_text moved to chunking.py; re-exported for backward-compatible imports.
__all__ = ["ProjectIndexer", "chunk_text", "chunk_code"]


class ProjectIndexer:
    """Embeds workspace files into a :class:`VectorIndex` and retrieves chunks.

    Retrieval is HYBRID: dense (embedding cosine) + lexical (BM25), fused with
    Reciprocal Rank Fusion, then optionally refined by a local cross-encoder
    reranker. Dense alone misses literal terms (codes, identifiers, versions,
    negations) and struggles in French; BM25 recovers that recall, and RRF makes
    the two incomparable score scales combine robustly. The embedding function
    and optional reranker are injected, keeping indexing/retrieval testable
    without a running model.
    """

    def __init__(
        self,
        files: Any,
        embed_fn: EmbedFn,
        index: VectorIndex,
        max_files: int = 2000,
        max_chars: int = 1000,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
    ):
        self.files = files
        self.embed = embed_fn
        self.index = index
        self.max_files = max_files
        self.max_chars = max_chars
        self.reranker = reranker
        # How many fused candidates to consider before reranking / truncation.
        self.candidate_k = candidate_k
        self._bm25: BM25 | None = None
        self._bm25_count = -1

    def reindex(self, full: bool = False) -> dict[str, Any]:
        """Refresh the index. INCREMENTAL by default: each file's content hash
        is stored with its chunks, so unchanged files are neither re-read into
        the embedder nor re-inserted — only new/changed files pay the embedding
        cost, and deleted files drop out. ``full=True`` (or a store without
        ``delete_paths``) rebuilds from scratch."""
        getter = getattr(self.files, "get_available_files", None)
        candidates = (getter(recursive=True) if callable(getter) else [])[: self.max_files]

        deleter = getattr(self.index, "delete_paths", None)
        incremental = not full and callable(deleter)
        existing: dict[str, str] = {}
        if incremental:
            for entry in self.index.snapshot():
                path = entry.get("path")
                if path:
                    existing.setdefault(path, str(entry.get("title") or ""))
        else:
            self.index.clear()

        entries: list[dict[str, Any]] = []
        indexed = 0
        skipped = 0
        unchanged = 0
        seen: set[str] = set()
        stale: list[str] = []
        for rel in candidates:
            ok, content = self.files.read_file_safe(rel, max_chars=200_000)
            if not ok:
                skipped += 1
                continue
            seen.add(rel)
            digest = hashlib.sha1(content.encode("utf-8", "ignore")).hexdigest()
            if incremental and existing.get(rel) == digest:
                unchanged += 1
                continue
            if incremental and rel in existing:
                stale.append(rel)
            file_had_chunk = False
            for chunk_index, chunk in enumerate(chunk_code(rel, content, self.max_chars)):
                vector = self.embed(chunk)
                if not vector:
                    continue
                entries.append(
                    {
                        "id": f"{rel}#{chunk_index}",
                        "path": rel,
                        "chunk_index": chunk_index,
                        "text": chunk,
                        "vector": list(vector),
                        "title": digest,  # file content hash → incremental skip
                    }
                )
                file_had_chunk = True
            if file_had_chunk:
                indexed += 1
        removed = [path for path in existing if path not in seen]
        if incremental and (stale or removed):
            deleter(stale + removed)
        self.index.add(entries)
        self._bm25 = None  # invalidate; rebuilt lazily on next retrieve
        return {
            "files": indexed,
            "chunks": self.index.count(),
            "skipped": skipped,
            "unchanged": unchanged,
            "removed": len(removed),
        }

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        entries = self.index.snapshot()
        if not entries:
            return []
        by_id = {entry.get("id"): entry for entry in entries}

        rankings: list[list[Any]] = []
        vector = self.embed(query)
        if vector:
            dense = self.index.search(vector, top_k=self.candidate_k)
            rankings.append([entry.get("id") for score, entry in dense if score > 0])
        lexical = self._ensure_bm25(entries).search(query, top_k=self.candidate_k)
        rankings.append([entries[index].get("id") for _score, index in lexical])

        fused = reciprocal_rank_fusion(rankings)
        candidates = [by_id[doc_id] for doc_id in fused if doc_id in by_id][: self.candidate_k]
        if not candidates:
            return []
        return self._rerank(query, candidates)[:top_k]

    def _ensure_bm25(self, entries: list[dict[str, Any]]) -> BM25:
        if self._bm25 is None or self._bm25_count != len(entries):
            self._bm25 = BM25([entry.get("text", "") for entry in entries])
            self._bm25_count = len(entries)
        return self._bm25

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
