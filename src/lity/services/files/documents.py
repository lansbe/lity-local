from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cap extracted text so a huge document can't blow up the prompt.
MAX_CHARS = 40_000


def extract_document(filename: str, data: bytes) -> dict[str, Any]:
    """Extract readable text from an uploaded document.

    Supports PDF (pypdf) and DOCX (python-docx) when those optional libs are
    present; otherwise decodes as plain text. Never raises — returns
    ``{ok, text, error}``.
    """
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            result = _extract_pdf(data)
        elif name.endswith(".docx"):
            result = _extract_docx(data)
        else:
            result = {"ok": True, "text": data.decode("utf-8", errors="replace"), "error": None}
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("Document extraction failed (%s): %s", filename, exc)
        return {"ok": False, "text": "", "error": str(exc)}

    text = (result.get("text") or "").strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n…[document tronqué]"
    return {"ok": result.get("ok", bool(text)), "text": text, "error": result.get("error")}


def _extract_pdf(data: bytes) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception:
        return {"ok": False, "text": "", "error": "pypdf absent (installe l'extra 'documents')."}
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if not text:
        return {"ok": False, "text": "", "error": "PDF sans texte extractible (scanné/image ?)."}
    return {"ok": True, "text": text, "error": None}


def _extract_docx(data: bytes) -> dict[str, Any]:
    try:
        import docx
    except Exception:
        return {
            "ok": False,
            "text": "",
            "error": "python-docx absent (installe l'extra 'documents').",
        }
    document = docx.Document(io.BytesIO(data))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    if not text:
        return {"ok": False, "text": "", "error": "Document vide."}
    return {"ok": True, "text": text, "error": None}
