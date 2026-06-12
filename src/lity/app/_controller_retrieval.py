from __future__ import annotations

import contextlib
import threading
from typing import Any


class RetrievalMixin:
    """RAG (project) + cross-session memory retrieval/indexing for AgentController.

    Relies on attributes set in ``AgentController.__init__`` (engine, paths,
    settings, memory, files, ``_indexer``, ``_reranker``, ``_reranker_tried``,
    ``_memory_indexer``, ``_rag_enabled``) and the ``_embedding_model()`` helper.
    Split out of the controller to keep that god-object smaller.
    """

    def _ensure_indexer(self) -> Any:
        if not hasattr(self.engine, "embed"):
            return None
        model = self._embedding_model()

        def embed_fn(text: str, _model: str = model) -> list[float] | None:
            return self.engine.embed(text, _model)

        if self._indexer is None:
            from lity.services.rag.indexer import ProjectIndexer
            from lity.services.rag.sqlite_store import SqliteVectorStore

            index = SqliteVectorStore(self.paths.vector_index_file.with_suffix(".db"))
            self._indexer = ProjectIndexer(
                self.files, embed_fn, index, reranker=self._ensure_reranker()
            )
        else:
            self._indexer.embed = embed_fn
        return self._indexer

    def _ensure_reranker(self) -> Any:
        """Build the optional local cross-encoder reranker once (None if absent).

        Hybrid dense+BM25 retrieval works without it; the reranker only refines
        the final order. Degrades gracefully when the ``rerank`` extra (fastembed)
        is not installed.
        """
        if not self._reranker_tried:
            self._reranker_tried = True
            from lity.services.rag.rerank import build_reranker

            self._reranker = build_reranker()
        return self._reranker

    def _cross_session_enabled(self) -> bool:
        if self.settings is None:
            return True
        return bool(self.settings.get("cross_session_memory", True))

    def _ensure_memory_indexer(self) -> Any:
        if not hasattr(self.engine, "embed"):
            return None
        model = self._embedding_model()

        def embed_fn(text: str, _model: str = model) -> list[float] | None:
            return self.engine.embed(text, _model)

        if self._memory_indexer is None:
            from lity.services.memory.memory_index import MemoryIndexer
            from lity.services.rag.sqlite_store import SqliteVectorStore

            index = SqliteVectorStore(self.paths.memory_index_file.with_suffix(".db"))
            self._memory_indexer = MemoryIndexer(embed_fn, index, reranker=self._ensure_reranker())
        else:
            self._memory_indexer.embed = embed_fn
        return self._memory_indexer

    def _passive_retrieval_query(self) -> str:
        """Standalone, history-aware query for the per-turn passive retrieval.

        The latest user message is often elliptical ("et le deuxième ?",
        "corrige ça") — retrieving on it alone returns off-topic chunks, so
        relevance fades on follow-ups. We rewrite it into a self-contained query
        using the recent turns (constrained decoding, 100% local). Memoized per
        turn (keyed by conversation + window length + message) so project RAG and
        cross-session memory share ONE rewrite call instead of two. Gated by the
        ``history_aware_retrieval`` setting; falls back to the raw last user
        message when off, unsupported, or on any failure — never empty, never
        worse than before.
        """
        context = self.memory.get_context()
        last_user = next(
            (msg.get("content") for msg in reversed(context) if msg.get("role") == "user"), None
        )
        if not last_user:
            return ""
        conv_id = getattr(self.memory, "active_conversation_id", None)
        key = (conv_id, len(context), last_user)
        cache = getattr(self, "_retrieval_query_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]

        query = last_user
        history_aware = self.settings is None or self.settings.get("history_aware_retrieval", True)
        if history_aware and hasattr(self.engine, "generate_structured"):
            from lity.services.rag.contextualize import contextualize_query

            def generate_fn(prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
                # A query rewrite is a cheap verdict: route it to the small
                # utility model when one is configured.
                return self.engine.generate_structured(
                    prompt, schema, think=False, prefer_utility=True
                )

            query = contextualize_query(generate_fn, context, last_user)

        self._retrieval_query_cache = (key, query)
        return query

    def _compose_injected_context(self, include_rag: bool = False) -> str:
        """Compose the per-turn injected context with per-TYPE budgets.

        The loaded code gets the dominant share: before this, facts + memory
        were concatenated first and a big loaded file was the one to get
        truncated — exactly backwards for a coding task. Each section is
        clamped to its own slice (unused space rolls over), so no section can
        evict another."""
        from lity.services.ai.context import compose_injected_context

        sections: list[tuple[str, float]] = [(self.files.get_context_for_ai(), 0.70)]
        if include_rag:
            sections.append((self._retrieve_rag_context(), 0.15))
        sections.append((self._retrieve_facts_context(), 0.08))
        sections.append((self._retrieve_memory_context(), 0.07))
        return compose_injected_context(sections)

    def _reindex_after_writes(self, receipts: dict[str, Any] | None) -> None:
        """Background: refresh the project index after the agent changed files,
        so retrieve_project sees the agent's own modifications immediately.
        Incremental — only changed files are re-embedded."""
        if not receipts or not self._rag_enabled or self._indexer is None:
            return
        wrote = any(
            item.get("ok") and item.get("name") in ("write_file", "edit_file")
            for item in receipts.get("items", [])
        )
        if not wrote:
            return

        def task() -> None:
            with contextlib.suppress(Exception):
                self._indexer.reindex()

        threading.Thread(target=task, daemon=True).start()

    def _retrieve_memory_context(self) -> str:
        """Recall the most relevant snippets from OTHER past conversations."""
        if not self._cross_session_enabled():
            return ""
        indexer = self._ensure_memory_indexer()
        if indexer is None or indexer.index.count() == 0:
            return ""
        query = self._passive_retrieval_query()
        if not query:
            return ""
        conv_id = getattr(self.memory, "active_conversation_id", None)
        hits = indexer.retrieve(query, top_k=3, exclude_conversation_id=conv_id)
        if not hits:
            return ""
        blocks = [
            f"--- {hit.get('title') or 'conversation'} ({hit.get('role', '')}) ---\n{hit['text']}"
            for hit in hits
        ]
        return (
            "[MÉMOIRE DES CONVERSATIONS PASSÉES — rappel, n'invente rien]\n"
            + "\n\n".join(blocks)
            + "\n[/MÉMOIRE]\n"
        )

    def _index_active_conversation_to_memory(self) -> None:
        """Background: embed the active conversation's new messages for recall."""
        if not self._cross_session_enabled():
            return
        indexer = self._ensure_memory_indexer()
        if indexer is None:
            return
        conv_id = getattr(self.memory, "active_conversation_id", None)
        if not conv_id:
            return
        try:
            messages = list(self.memory.get_active_messages())
        except Exception:
            return
        title = ""
        store = getattr(self.memory, "conversations", None)
        if store is not None and hasattr(store, "get_meta"):
            meta = store.get_meta(conv_id) or {}
            title = meta.get("title", "") if isinstance(meta, dict) else ""

        def task() -> None:
            try:
                indexer.index_conversation(conv_id, title, messages)
            except Exception:
                return

        threading.Thread(target=task, daemon=True).start()

    # ----------------------------------------------------- durable facts (Mem0)
    def _facts_enabled(self) -> bool:
        if self.settings is None:
            return True
        return bool(self.settings.get("durable_facts", True))

    def _ensure_fact_store(self) -> Any:
        """Build the durable-fact semantic index once (None when embeddings are
        unavailable). On first build it seeds from the facts already on disk so
        recall works immediately in a continued session."""
        if not self._facts_enabled() or not hasattr(self.engine, "embed"):
            return None
        model = self._embedding_model()

        def embed_fn(text: str, _model: str = model) -> list[float] | None:
            return self.engine.embed(text, _model)

        if self._fact_store is None:
            from lity.services.memory.fact_store import FactStore
            from lity.services.rag.sqlite_store import SqliteVectorStore

            store = SqliteVectorStore(self.paths.fact_index_file.with_suffix(".db"))
            self._fact_store = FactStore(embed_fn, store)
            with contextlib.suppress(Exception):
                self._fact_store.index_facts(self._durable_facts_seed())
        else:
            self._fact_store.embed = embed_fn
        return self._fact_store

    def _durable_facts_seed(self) -> dict[str, Any]:
        """The long-term facts already persisted by the memory manager."""
        if not hasattr(self.memory, "get_memory"):
            return {}
        try:
            return dict((self.memory.get_memory() or {}).get("facts", {}))
        except Exception:
            return {}

    def _index_fact_to_store(self, fact: dict[str, Any] | None) -> None:
        """Embed a freshly extracted long-term fact so it is recallable at once."""
        if not fact or fact.get("categorie") != "long_term_facts":
            return
        store = self._ensure_fact_store()
        if store is None:
            return
        key, value = fact.get("cle"), fact.get("valeur")
        if key and value:
            try:
                store.add_fact(str(key), str(value))
            except Exception:
                return

    def _retrieve_facts_context(self) -> str:
        """Recall the durable facts relevant to the current turn (Mem0-light).

        Uses the same history-aware standalone query as the other passive
        retrievals, so a follow-up still recalls the right fact. Injects only the
        relevant few (relevance-gated), so a detail mentioned once long ago can
        resurface without padding every prompt with the whole fact set."""
        store = self._ensure_fact_store()
        if store is None or store.store.count() == 0:
            return ""
        query = self._passive_retrieval_query()
        if not query:
            return ""
        hits = store.recall(query)
        if not hits:
            return ""
        lines = "\n".join(f"- {hit['text']}" for hit in hits)
        return (
            "[FAITS MÉMORISÉS PERTINENTS — rappelle-toi, n'invente rien]\n" + lines + "\n[/FAITS]\n"
        )

    def index_project(self) -> dict[str, Any]:
        indexer = self._ensure_indexer()
        if indexer is None:
            return {"ok": False, "chunks": 0, "message": "Embeddings non supportés par ce moteur."}
        stats = indexer.reindex()
        ok = stats["chunks"] > 0
        if ok:
            self._rag_enabled = True
            message = f"Projet indexé : {stats['files']} fichier(s), {stats['chunks']} extraits."
        else:
            message = (
                "Aucun extrait indexé. Installe un modèle d'embedding "
                "(ex. : ollama pull nomic-embed-text) et choisis un dossier de travail."
            )
        return {"ok": ok, "message": message, **stats}

    @property
    def rag_enabled(self) -> bool:
        return self._rag_enabled

    def set_rag(self, enabled: bool) -> bool:
        self._rag_enabled = bool(enabled)
        return self._rag_enabled

    def index_size(self) -> int:
        return self._indexer.index.count() if self._indexer is not None else 0

    def _agent_retrieval(self) -> dict[str, Any]:
        """Expose the project RAG + cross-session memory as agent TOOLS (agentic
        retrieval): the model decides when to query its local knowledge, instead
        of a single passive injection. Only includes a source when it has content.
        """
        retrieval: dict[str, Any] = {}
        if self._rag_enabled:
            indexer = self._ensure_indexer()
            if indexer is not None and indexer.index.count() > 0:
                retrieval["project"] = self._corrective_project_fn(indexer)
        if self._cross_session_enabled():
            memory = self._ensure_memory_indexer()
            if memory is not None and memory.index.count() > 0:
                conv_id = getattr(self.memory, "active_conversation_id", None)
                retrieval["memory"] = lambda query, k=5: memory.retrieve(
                    query, top_k=k, exclude_conversation_id=conv_id
                )
        return retrieval

    def _corrective_project_fn(self, indexer: Any) -> Any:
        """Wrap project retrieval with CRAG-light (grade → rewrite+retry → flag).

        Applied ONLY here, on the agentic ``retrieve_project`` tool the model
        invokes deliberately — never on the per-turn passive injection — so the
        extra grading call is bounded, not a tax on every message. Disable via
        the ``corrective_rag`` setting; falls back to plain retrieval when off or
        when the engine has no structured-generation primitive. Returns a list of
        chunks (the AgentLoop contract is unchanged), appending one clearly
        labelled "(piste)" hint toward web search when retrieval stays weak.
        """
        plain = lambda query, k=5: indexer.retrieve(query, top_k=k)  # noqa: E731
        if self.settings is not None and not self.settings.get("corrective_rag", True):
            return plain
        if not hasattr(self.engine, "generate_structured"):
            return plain

        from lity.services.rag.agentic import CorrectiveRetriever

        def generate_fn(prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
            # Relevance grading is a cheap verdict → small utility model if set.
            return self.engine.generate_structured(prompt, schema, think=False, prefer_utility=True)

        corrective = CorrectiveRetriever(
            lambda query, k: indexer.retrieve(query, top_k=k), generate_fn=generate_fn
        )

        def retrieve(query: str, k: int = 5) -> list[dict[str, Any]]:
            result = corrective.retrieve(query, top_k=k)
            chunks = list(result.get("chunks") or [])
            if result.get("status") == "weak":
                chunks.append(
                    {
                        "path": "(piste)",
                        "text": (
                            "Peu d'extraits vraiment pertinents, même après reformulation. "
                            "Si la question le justifie, envisage une recherche web."
                        ),
                    }
                )
            return chunks

        return retrieve

    def _retrieve_rag_context(self) -> str:
        if not (
            self._rag_enabled and self._indexer is not None and self._indexer.index.count() > 0
        ):
            return ""
        query = self._passive_retrieval_query()
        if not query:
            return ""
        chunks = self._indexer.retrieve(query, top_k=4)
        if not chunks:
            return ""
        blocks = [f"--- EXTRAIT {chunk['path']} ---\n{chunk['text']}" for chunk in chunks]
        return "[CONTEXTE PROJET PERTINENT]\n" + "\n\n".join(blocks) + "\n[/CONTEXTE PROJET]\n"
