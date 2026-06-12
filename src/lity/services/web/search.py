from __future__ import annotations

import logging
import re
import urllib.parse
from collections import OrderedDict
from typing import Any, Protocol

from lity.services.web.http_util import get_json

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 6


class SearchProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def search(self, query: str, max_results: int) -> list[dict[str, str]]: ...


class SearxngProvider:
    """Self-hosted SearXNG metasearch (JSON API). No key, you control it.

    The instance must allow the JSON format (``search.formats: [html, json]``
    in its settings.yml). Returns an empty list on any failure so the chain
    falls through to the next provider.
    """

    name = "searxng"

    def __init__(self, base_url: str, timeout: int = 8, language: str = "fr"):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.language = language

    def available(self) -> bool:
        return bool(self.base_url)

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "language": self.language, "safesearch": 1}
        )
        data = get_json(f"{self.base_url}/search?{params}", timeout=self.timeout)
        results: list[dict[str, str]] = []
        for item in (data or {}).get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            results.append(
                {
                    "title": item.get("title", "") or url,
                    "url": url,
                    "snippet": item.get("content", "") or "",
                }
            )
            if len(results) >= max_results:
                break
        return results


class DuckDuckGoProvider:
    """DuckDuckGo via the optional ``ddgs`` / ``duckduckgo_search`` lib.

    Zero key, but unofficial — used as a fallback. Unavailable (returns []) if
    the library is not installed.
    """

    name = "duckduckgo"

    def __init__(self, timeout: int = 8):
        self.timeout = timeout

    def available(self) -> bool:
        return _ddgs_class() is not None

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        ddgs_class = _ddgs_class()
        if ddgs_class is None:
            return []
        try:
            with ddgs_class() as ddgs:
                hits = ddgs.text(query, max_results=max_results)
        except Exception as exc:
            logger.info("DuckDuckGo search failed: %s", exc)
            return []
        results: list[dict[str, str]] = []
        for hit in hits or []:
            url = hit.get("href") or hit.get("url") or ""
            if not url:
                continue
            results.append(
                {
                    "title": hit.get("title", "") or url,
                    "url": url,
                    "snippet": hit.get("body", "") or hit.get("snippet", "") or "",
                }
            )
        return results


class WikipediaProvider:
    """Official Wikipedia search API — free, no key, great for factual queries."""

    name = "wikipedia"

    def __init__(self, language: str = "fr", timeout: int = 8):
        self.language = language
        self.timeout = timeout

    def available(self) -> bool:
        return True

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results,
            }
        )
        url = f"https://{self.language}.wikipedia.org/w/api.php?{params}"
        data = get_json(url, timeout=self.timeout)
        results: list[dict[str, str]] = []
        for item in (data or {}).get("query", {}).get("search", []):
            title = item.get("title", "")
            page = urllib.parse.quote(title.replace(" ", "_"))
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "") or "")
            results.append(
                {
                    "title": title,
                    "url": f"https://{self.language}.wikipedia.org/wiki/{page}",
                    "snippet": re.sub(r"\s+", " ", snippet).strip(),
                }
            )
        return results


class WebSearcher:
    """Run a query through a priority chain of providers.

    Returns the first provider that yields results; records errors from the
    ones that failed so the caller can surface a useful message.
    """

    def __init__(self, providers: list[SearchProvider], cache_size: int = 32):
        self.providers = providers
        self._cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"ok": False, "error": "Requête vide.", "provider": None, "results": []}
        cache_key = (query.lower(), max_results)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return _copy_outcome(cached)

        errors: list[str] = []
        for provider in self.providers:
            try:
                if not provider.available():
                    continue
                results = provider.search(query, max_results)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{provider.name}: {exc}")
                continue
            if results:
                outcome = {
                    "ok": True,
                    "provider": provider.name,
                    "results": results,
                    "error": None,
                }
                self._remember(cache_key, outcome)
                return _copy_outcome(outcome)

        message = "; ".join(errors) if errors else "Aucun résultat (aucun fournisseur disponible)."
        return {"ok": False, "error": message, "provider": None, "results": []}

    def _remember(self, key: tuple[str, int], outcome: dict[str, Any]) -> None:
        self._cache[key] = _copy_outcome(outcome)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _ddgs_class() -> Any | None:
    try:
        from ddgs import DDGS

        return DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS

            return DDGS
        except Exception:
            return None


def _copy_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    copied = dict(outcome)
    copied["results"] = [dict(item) for item in outcome.get("results", [])]
    return copied
