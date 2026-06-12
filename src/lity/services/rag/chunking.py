from __future__ import annotations

import ast


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 120) -> list[str]:
    """Plain sliding-window chunking (the fallback for non-Python content)."""
    text = text or ""
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return [chunk for chunk in chunks if chunk.strip()]


def chunk_code(path: str, text: str, max_chars: int = 1000) -> list[str]:
    """Dispatch: symbol-aware chunking for Python, sliding window otherwise."""
    if str(path).lower().endswith(".py"):
        return chunk_python(text, max_chars)
    return chunk_text(text, max_chars)


def chunk_python(text: str, max_chars: int = 1000) -> list[str]:
    """Symbol-aware chunking for Python: one chunk per function/class.

    Character windows cut definitions in half — the retrieved fragment then
    shows the middle of a function with no signature, and the embedding mixes
    two unrelated symbols. Here the AST gives natural boundaries: top-level
    functions and classes stay whole (with their decorators), oversized classes
    split per method, small neighbours merge, and anything still oversized
    falls back to the sliding window. Unparseable text (work in progress,
    Python 2…) falls back entirely — never returns less than chunk_text would.
    """
    try:
        tree = ast.parse(text or "")
    except (SyntaxError, ValueError):
        return chunk_text(text, max_chars)
    lines = (text or "").splitlines()
    if not lines:
        return []

    segments: list[tuple[int, int]] = []
    previous_end = 0
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = _node_span(node)
        if start > previous_end:
            segments.append((previous_end, start))  # imports / constants between symbols
        if isinstance(node, ast.ClassDef) and _segment_size(lines, start, end) > max_chars:
            segments.extend(_class_segments(node, start, end))
        else:
            segments.append((start, end))
        previous_end = end
    if previous_end < len(lines):
        segments.append((previous_end, len(lines)))
    if not segments:
        return chunk_text(text, max_chars)

    return _merge_segments(lines, segments, max_chars)


def _node_span(node: ast.stmt) -> tuple[int, int]:
    """Line span of a definition INCLUDING its decorators (0-based, end-exclusive)."""
    start = node.lineno - 1
    for decorator in getattr(node, "decorator_list", []) or []:
        start = min(start, decorator.lineno - 1)
    return start, int(node.end_lineno or node.lineno)


def _segment_size(lines: list[str], start: int, end: int) -> int:
    return sum(len(line) + 1 for line in lines[start:end])


def _class_segments(node: ast.ClassDef, start: int, end: int) -> list[tuple[int, int]]:
    """Split an oversized class into header + one segment per method."""
    segments: list[tuple[int, int]] = []
    previous = start
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            child_start, child_end = _node_span(child)
            if child_start > previous:
                segments.append((previous, child_start))  # class header / attributes
            segments.append((child_start, child_end))
            previous = child_end
    if previous < end:
        segments.append((previous, end))
    return segments


def _merge_segments(lines: list[str], segments: list[tuple[int, int]], max_chars: int) -> list[str]:
    """Merge small adjacent segments up to the budget; window oversized ones."""
    chunks: list[str] = []
    buffer = ""
    for start, end in segments:
        segment = "\n".join(lines[start:end]).strip("\n")
        if not segment.strip():
            continue
        if buffer and len(buffer) + len(segment) + 1 <= max_chars:
            buffer += "\n" + segment
            continue
        if buffer:
            chunks.append(buffer)
            buffer = ""
        if len(segment) <= max_chars:
            buffer = segment
        else:
            chunks.extend(chunk_text(segment, max_chars))
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk.strip()]
