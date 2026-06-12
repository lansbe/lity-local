from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

from lity.services.rag.vector_index import cosine_similarity

_COLUMNS = ("id", "conversation_id", "path", "chunk_index", "role", "title", "text", "vector")


class SqliteVectorStore:
    """SQLite-backed drop-in for :class:`VectorIndex` (incremental + durable).

    Exposes the exact surface the indexers use — ``snapshot`` / ``add`` /
    ``clear`` / ``count`` / ``search`` — so :class:`ProjectIndexer` and
    :class:`MemoryIndexer` work against it unchanged. It replaces the JSON
    full-file rewrite (write amplification that grew with every turn) with
    incremental ``INSERT OR REPLACE`` upserts inside a WAL-mode transaction, and
    is thread-safe via an ``RLock`` over a shared connection. Vectors are stored
    as JSON text; ranking stays brute-force cosine (fine at local scale — a
    persistent FTS5 / ANN index is the next step for very large corpora).
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = RLock()
        target = str(self.path) if self.path else ":memory:"
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + an RLock lets the background indexing thread
        # and the main thread share one connection safely.
        self._conn = sqlite3.connect(target, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "id TEXT PRIMARY KEY, conversation_id TEXT, path TEXT, chunk_index INTEGER, "
                "role TEXT, title TEXT, text TEXT, vector TEXT)"
            )
            self._conn.commit()

    def add(self, entries: list[dict[str, Any]]) -> None:
        rows = [
            (
                entry.get("id"),
                entry.get("conversation_id"),
                entry.get("path"),
                entry.get("chunk_index"),
                entry.get("role"),
                entry.get("title"),
                entry.get("text"),
                json.dumps(entry.get("vector") or []),
            )
            for entry in entries
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO entries ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                rows,
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()

    def delete_paths(self, paths: list[str]) -> None:
        """Drop every chunk of the given source paths (incremental reindexing)."""
        rows = [(path,) for path in paths if path]
        if not rows:
            return
        with self._lock:
            self._conn.executemany("DELETE FROM entries WHERE path = ?", rows)
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM entries").fetchall()
        entries: list[dict[str, Any]] = []
        for row in rows:
            entry = dict(zip(_COLUMNS, row, strict=False))
            try:
                entry["vector"] = json.loads(entry["vector"] or "[]")
            except Exception:
                entry["vector"] = []
            entries.append(entry)
        return entries

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
