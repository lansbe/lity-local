from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from lity.infrastructure.paths import AppPaths
from lity.services.rag.bm25 import BM25

_MAX_TEXT_CHARS = 1200


def search_codex_rag(
    query: str,
    *,
    paths: AppPaths | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search Lity's local context for Codex without calling a local model.

    This is intentionally lexical-only. It lets Codex decide when to query Lity's
    stored context, while avoiding the hidden Ollama/embedding work that made
    Codex turns feel heavy.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return []
    app_paths = paths or AppPaths.create()
    documents = _collect_documents(app_paths)
    if not documents:
        return []
    ranker = BM25([document["text"] for document in documents])
    results = []
    for score, index in ranker.search(cleaned, top_k=max(1, int(top_k or 5))):
        document = dict(documents[index])
        document["score"] = round(float(score), 4)
        results.append(document)
    return results


def _collect_documents(paths: AppPaths) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    documents.extend(_load_facts(paths.facts_file))
    documents.extend(_load_conversations(paths.conversations_dir))
    documents.extend(
        _load_sqlite_entries(paths.memory_index_file.with_suffix(".db"), "memory_index")
    )
    documents.extend(
        _load_sqlite_entries(paths.vector_index_file.with_suffix(".db"), "project_rag")
    )
    return [document for document in documents if document.get("text", "").strip()]


def _load_facts(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    documents = []
    for key, raw in data.items():
        value = raw.get("value") if isinstance(raw, dict) else raw
        text = str(value or "").strip()
        if text:
            documents.append(
                {
                    "source": "facts",
                    "title": str(key),
                    "text": _clip(text),
                }
            )
    return documents


def _load_conversations(directory: Path) -> list[dict[str, Any]]:
    documents = []
    try:
        files = sorted(directory.glob("*.json"))
    except Exception:
        return documents
    for path in files:
        if path.name == "index.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = str(data.get("title") or path.stem)
        messages = data.get("messages") if isinstance(data, dict) else []
        if not isinstance(messages, list):
            continue
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            text = str(message.get("content") or "").strip()
            if not text:
                continue
            documents.append(
                {
                    "source": "conversation",
                    "title": title,
                    "role": str(message.get("role") or ""),
                    "path": f"{path.name}#{position}",
                    "text": _clip(text),
                }
            )
    return documents


def _load_sqlite_entries(path: Path, source: str) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    documents = []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:
        return []
    try:
        rows = conn.execute("SELECT path, role, title, text FROM entries").fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    for rel_path, role, title, text in rows:
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        documents.append(
            {
                "source": source,
                "title": str(title or rel_path or source),
                "role": str(role or ""),
                "path": str(rel_path or ""),
                "text": _clip(cleaned),
            }
        )
    return documents


def _clip(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= _MAX_TEXT_CHARS:
        return cleaned
    return cleaned[: _MAX_TEXT_CHARS - 1].rstrip() + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search Lity RAG without local LLM calls.")
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--query", dest="query_opt", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    query = args.query_opt or args.query
    print(json.dumps(search_codex_rag(query, top_k=args.top_k), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
