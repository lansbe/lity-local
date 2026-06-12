from __future__ import annotations

import json
import math
from pathlib import Path
from threading import RLock
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorIndex:
    """A tiny JSON-persisted vector store with cosine-similarity search.

    Entries are ``{"id", "path", "chunk_index", "text", "vector"}``. Suitable for
    a single local project; not meant to scale to huge corpora.

    Thread-safe: a re-entrant lock guards every read/write, and ``snapshot()``
    hands readers a private copy so a background indexing thread can mutate the
    store while the main thread iterates results without a data race.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[dict[str, Any]] = []
        self._lock = RLock()
        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        except Exception:
            data = []
        with self._lock:
            self.entries = data if isinstance(data, list) else []

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            payload = json.dumps(self.entries, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(self.path)

    def clear(self) -> None:
        with self._lock:
            self.entries = []
        self.save()

    def add(self, entries: list[dict[str, Any]]) -> None:
        with self._lock:
            self.entries.extend(entries)
        self.save()

    def delete_paths(self, paths: list[str]) -> None:
        """Drop every chunk of the given source paths (incremental reindexing)."""
        targets = set(paths)
        if not targets:
            return
        with self._lock:
            self.entries = [entry for entry in self.entries if entry.get("path") not in targets]
        self.save()

    def count(self) -> int:
        with self._lock:
            return len(self.entries)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the entries for safe iteration by readers."""
        with self._lock:
            return list(self.entries)

    def search(
        self, query_vector: list[float], top_k: int = 4
    ) -> list[tuple[float, dict[str, Any]]]:
        scored = [
            (cosine_similarity(query_vector, entry.get("vector", [])), entry)
            for entry in self.snapshot()
            if entry.get("vector")
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]
