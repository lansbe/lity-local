from __future__ import annotations

from collections.abc import Callable
from typing import Any

# embed_fn(text) -> embedding vector (or None when embeddings are unavailable).
# Injected so the store is fully unit-testable without a running model.
EmbedFn = Callable[[str], "list[float] | None"]


class FactStore:
    """Mem0-light: a semantic index over DURABLE facts (stable preferences,
    project constants, decisions) so they resurface by RELEVANCE in very long
    conversations — long after the raw turns scrolled out of the window and were
    compressed into the lossy rolling summary.

    Why this beats the existing flat fact dump: today durable facts are injected
    wholesale only when seen more than once (``count > 1``), so a detail stated
    ONCE early on is dropped, and irrelevant facts pad every prompt (distractors,
    which the context-rot research shows hurt more than filler). Here each fact is
    embedded individually and only the few RELEVANT to the current question are
    recalled — a fact mentioned once, 200 turns ago, can still come back when it
    matters.

    100% local: embeddings come from the same Ollama model via the injected
    ``embed_fn``. Persistence + dense search are delegated to the same
    ``SqliteVectorStore`` that backs project RAG and cross-session memory. Recall
    quality tracks the embedding model — a capable multilingual model (e.g.
    bge-m3) discriminates far better than a small one (e.g. nomic-embed-text).
    Degrades to a no-op (``recall`` returns ``[]``) without embeddings; never
    raises.
    """

    def __init__(self, embed_fn: EmbedFn, store: Any, *, min_score: float = 0.4, top_k: int = 4):
        self.embed = embed_fn
        self.store = store
        self.min_score = min_score
        self.top_k = top_k

    def index_facts(self, facts: dict[str, Any]) -> int:
        """Upsert ``{key: value}`` facts (``value`` a str, or a ``{'value': ...}``
        record as stored on disk). Idempotent: a fact already indexed with the
        SAME text is skipped, so re-seeding the whole set each session is cheap.
        Returns the number of facts (re)embedded."""
        existing = {entry.get("id"): str(entry.get("text", "")) for entry in self.store.snapshot()}
        new_entries: list[dict[str, Any]] = []
        for key, raw in (facts or {}).items():
            text = self._fact_text(raw)
            if not text or not str(key).strip():
                continue
            fid = f"fact::{key}"
            if existing.get(fid) == text:
                continue  # already indexed, unchanged → no re-embed
            vector = self.embed(text)
            if not vector:
                continue
            new_entries.append({"id": fid, "title": str(key), "text": text, "vector": list(vector)})
        if new_entries:
            self.store.add(new_entries)
        return len(new_entries)

    def add_fact(self, key: str, value: str) -> bool:
        """Index or refresh a single fact. Returns True if it was (re)embedded."""
        return self.index_facts({key: value}) > 0

    def recall(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the durable facts most relevant to ``query``, each as
        ``{key, text, score}``. Facts below ``min_score`` are dropped as
        distractors. Empty when there is no query, no facts, or no embeddings."""
        cleaned = (query or "").strip()
        if not cleaned or self.store.count() == 0:
            return []
        vector = self.embed(cleaned)
        if not vector:
            return []
        hits = self.store.search(vector, top_k=top_k or self.top_k)
        return [
            {"key": entry.get("title", ""), "text": entry.get("text", ""), "score": score}
            for score, entry in hits
            if score >= self.min_score and str(entry.get("text", "")).strip()
        ]

    @staticmethod
    def _fact_text(raw: Any) -> str:
        if isinstance(raw, dict):
            raw = raw.get("value", "")
        return str(raw or "").strip()
