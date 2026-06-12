from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from lity.services.web.fetch import DEFAULT_MAX_CHARS, query_coverage, select_relevant


class WebResearcher:
    """Search and read several sources in parallel for one fast agent tool call."""

    def __init__(
        self,
        searcher: Any,
        fetcher: Any,
        *,
        fetch_limit: int = 3,
        max_workers: int = 3,
        fetch_chars: int = DEFAULT_MAX_CHARS,
        passage_chars: int = 3600,
    ):
        self.searcher = searcher
        self.fetcher = fetcher
        self.fetch_limit = max(1, fetch_limit)
        self.max_workers = max(1, max_workers)
        self.fetch_chars = max(1000, fetch_chars)
        self.passage_chars = max(1000, passage_chars)

    def research(
        self,
        query: str,
        *,
        max_results: int = 6,
        fetch_limit: int | None = None,
    ) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"ok": False, "error": "Requête vide.", "sources": [], "failed": []}

        outcome = self.searcher.search(query, max_results=max_results)
        if not outcome.get("ok"):
            return {
                "ok": False,
                "query": query,
                "provider": outcome.get("provider"),
                "error": outcome.get("error", "Recherche sans résultat."),
                "sources": [],
                "failed": [],
                "searched": 0,
                "fetched": 0,
            }

        results = [dict(item) for item in outcome.get("results", []) if item.get("url")]
        limit = max(1, fetch_limit or self.fetch_limit)
        candidates = results[:limit]
        sources: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []

        workers = min(self.max_workers, len(candidates)) or 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self._read_source, index, item, query): index
                for index, item in enumerate(candidates)
            }
            for future in as_completed(future_map):
                read = future.result()
                if read.get("ok"):
                    sources.append(read)
                else:
                    failed.append(read)

        sources.sort(key=lambda item: int(item.get("index", 0)))
        failed.sort(key=lambda item: int(item.get("index", 0)))
        return {
            "ok": bool(sources),
            "query": query,
            "provider": outcome.get("provider"),
            "sources": sources,
            "failed": failed,
            "search_results": results,
            "searched": len(results),
            "fetched": len(sources),
            "error": None if sources else "Aucune source lisible extraite.",
        }

    def _read_source(self, index: int, item: dict[str, Any], query: str) -> dict[str, Any]:
        url = str(item.get("url", ""))
        try:
            page = self.fetcher.fetch(url, max_chars=self.fetch_chars)
        except Exception as exc:  # pragma: no cover - defensive boundary
            return {"ok": False, "index": index, "url": url, "error": str(exc)}
        if not page.get("ok"):
            return {
                "ok": False,
                "index": index,
                "url": url,
                "title": item.get("title", ""),
                "error": str(page.get("error", "Lecture impossible.")),
            }

        full_text = str(page.get("text", ""))
        passage = select_relevant(
            full_text,
            query,
            lambda _text: None,
            max_chars=self.passage_chars,
        )
        return {
            "ok": True,
            "index": index,
            "title": page.get("title") or item.get("title") or url,
            "url": url,
            "snippet": item.get("snippet", ""),
            "text": passage,
            "coverage": query_coverage(full_text, query),
        }
