from __future__ import annotations

import html as _html
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from lity.services.web.http_util import get_text

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 6000


class PageFetcher:
    """Download a URL and return clean, readable text.

    Uses trafilatura when available (strips nav/ads/boilerplate); otherwise a
    crude regex HTML strip. Results are cached (LRU) so re-fetching the same URL
    within a session is free.
    """

    def __init__(self, timeout: int = 8, cache_size: int = 64):
        self.timeout = timeout
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._cache_lock = threading.Lock()

    def fetch(self, url: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "url": url, "error": "URL invalide (http/https requis)."}

        with self._cache_lock:
            page = self._cache.get(url)
            if page is not None:
                self._cache.move_to_end(url)
        if page is None:
            page = self._download(url)
            self._remember(url, page)

        if not page.get("ok"):
            return dict(page)

        text = page["text"]
        clipped = text[:max_chars] + ("\n…[tronqué]" if len(text) > max_chars else "")
        return {**page, "text": clipped}

    def _download(self, url: str) -> dict[str, Any]:
        html = get_text(url, timeout=self.timeout)
        if html is None:
            return {"ok": False, "url": url, "error": "Téléchargement impossible (réseau/blocage)."}
        title, text = _extract(html, url)
        if not text:
            return {"ok": False, "url": url, "error": "Aucun contenu lisible extrait."}
        return {"ok": True, "url": url, "title": title, "text": text}

    def _remember(self, url: str, value: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[url] = value
            self._cache.move_to_end(url)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)


def _extract(html: str, url: str) -> tuple[str, str]:
    """Return (title, main_text) — trafilatura if present, else a crude strip."""
    title = _crude_title(html)
    try:
        import trafilatura

        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if text and text.strip():
            return title, text.strip()
    except Exception as exc:
        logger.info("trafilatura extraction failed (%s): %s", url, exc)

    return title, _crude_text(html)


def _crude_title(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _crude_text(html: str) -> str:
    # Focus on the main content region when the page marks one (article/main),
    # so boilerplate-heavy pages don't drown the real text. Then drop non-content
    # blocks, strip tags, decode entities (keep &amp;/&#233; etc.), collapse ws.
    region = _main_region(html) or html
    region = re.sub(r"(?is)<(script|style|noscript|template|svg)[^>]*>.*?</\1>", " ", region)
    region = re.sub(r"(?is)<(head|nav|footer|header|aside)[^>]*>.*?</\1>", " ", region)
    text = re.sub(r"(?s)<[^>]+>", " ", region)
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _main_region(html: str) -> str | None:
    """The first substantial <article>/<main> region, if the page marks one."""
    for tag in ("article", "main"):
        match = re.search(rf"(?is)<{tag}[^>]*>(.*?)</{tag}>", html)
        if match and len(match.group(1)) > 200:
            return match.group(1)
    return None


# Query tokens worth matching: words of 3+ letters/digits (drops "de", "la",
# "the"…). Language-agnostic, never hardcoded to a topic.
def query_terms(query: str) -> set[str]:
    return {token for token in re.findall(r"\w{3,}", (query or "").lower())}


def query_coverage(text: str, query: str) -> float:
    """Fraction of the query's distinct terms that appear in ``text`` (0..1).

    A general, topic-agnostic signal: ~0 means the page does not even mention
    what the user is looking for, so the agent should try another source."""
    terms = query_terms(query)
    if not terms:
        return 1.0
    low = (text or "").lower()
    return sum(1 for term in terms if term in low) / len(terms)


def select_relevant(
    text: str,
    query: str,
    embed: Callable[[str], list[float] | None],
    *,
    top_k: int = 4,
    chunk_chars: int = 700,
    max_chunks: int = 16,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Keep the passages of ``text`` most relevant to ``query``.

    Dense (embeddings) when an embedder is available; otherwise a LEXICAL
    fallback that ranks passages by query-term overlap. The old behaviour was to
    return the document HEAD when embeddings were missing — which, on a long
    page whose answer sits below the fold (or when no embedding model is pulled),
    handed the model the menu instead of the answer. Lexical ranking surfaces the
    passages that actually discuss the query, for any topic, with zero models.
    Never raises; returns the head only when there is nothing better to do.
    """
    head = text[:max_chars]
    query = (query or "").strip()
    if not query:
        return head

    chunks = _chunk(text, chunk_chars)[:max_chunks]
    if len(chunks) <= 1:
        return head

    ranked = _dense_rank(chunks, query, embed, top_k)
    if ranked is None:
        ranked = _lexical_rank(chunks, query, top_k)
    if not ranked:
        return head

    kept = sorted(ranked, key=lambda item: item[0])  # restore document order
    return "\n…\n".join(chunk for _, chunk in kept)[:max_chars]


def _dense_rank(
    chunks: list[str],
    query: str,
    embed: Callable[[str], list[float] | None],
    top_k: int,
) -> list[tuple[int, str]] | None:
    """Embedding-cosine ranking, or None when the embedder is unavailable."""
    query_vector = _safe_embed(embed, query)
    if query_vector is None:
        return None
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        vector = _safe_embed(embed, chunk)
        if vector is None:
            return None  # embedder unreliable mid-pass → let lexical take over
        scored.append((_cosine(query_vector, vector), index, chunk))
    scored.sort(key=lambda item: -item[0])
    return [(index, chunk) for _, index, chunk in scored[:top_k]]


def _lexical_rank(chunks: list[str], query: str, top_k: int) -> list[tuple[int, str]] | None:
    """Rank passages by how many query terms (and how often) they contain."""
    terms = query_terms(query)
    if not terms:
        return None
    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        low = chunk.lower()
        score = sum(low.count(term) for term in terms)
        if score:
            scored.append((score, index, chunk))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    return [(index, chunk) for _, index, chunk in scored[:top_k]]


def _chunk(text: str, size: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > size and current:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def _safe_embed(embed: Callable[[str], list[float] | None], text: str) -> list[float] | None:
    try:
        vector = embed(text)
    except Exception:
        return None
    return vector if vector else None


def _cosine(a: list[float], b: list[float]) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = sum(value * value for value in a[:length]) ** 0.5
    norm_b = sum(value * value for value in b[:length]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
