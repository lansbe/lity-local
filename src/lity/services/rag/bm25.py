from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, Unicode-aware (keeps accented French words)."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25:
    """In-memory Okapi BM25 lexical ranker over a fixed list of documents.

    Pure Python, no dependencies. Complements dense (embedding) search: BM25
    catches exact/literal terms embeddings miss — error codes, identifiers,
    function names, versions, negations. Built once per corpus; ``search``
    returns ``(score, doc_index)`` pairs sorted high→low. Sized for a local
    project (hundreds of files / thousands of chunks), not a web-scale corpus.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.freqs: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        document_frequency: dict[str, int] = {}
        for document in documents:
            tokens = tokenize(document)
            self.doc_len.append(len(tokens))
            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.freqs.append(term_freq)
            for term in term_freq:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        self.doc_count = len(documents)
        self.avg_len = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        # BM25+ idf: log(1 + (N - df + 0.5)/(df + 0.5)) — always positive, so a
        # term present in every document still contributes a little.
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, int]]:
        if not self.doc_count or self.avg_len == 0:
            return []
        query_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, term_freq in enumerate(self.freqs):
            length = self.doc_len[index] or 1
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += idf * (freq * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]
