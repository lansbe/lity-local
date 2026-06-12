from __future__ import annotations

import importlib.util
import logging
from typing import Any

from lity.services.rag.hybrid import Reranker

logger = logging.getLogger(__name__)

# Small, fast cross-encoder; downloaded once as ONNX on first real use.
_DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def build_reranker(model_name: str = _DEFAULT_RERANK_MODEL) -> Reranker | None:
    """Return a local cross-encoder reranker ``(query, texts) -> [scores]``, or None.

    Uses fastembed (ONNX Runtime — already pulled in by piper-tts, so no torch
    and no API key) and ships by default. 100% local. Returns None only if
    fastembed is not importable at all; otherwise it returns a callable that
    loads the model LAZILY on the first real rerank (so merely indexing a project
    never triggers the one-time ~80 MB model download). Any load/scoring failure
    degrades to an empty score list, and the caller keeps the RRF ordering.
    """
    if importlib.util.find_spec("fastembed") is None:
        logger.info("Reranker disabled: fastembed not installed.")
        return None

    state: dict[str, Any] = {}  # {"encoder": TextCrossEncoder | False}

    def rerank(query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        encoder = state.get("encoder")
        if encoder is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                encoder = TextCrossEncoder(model_name=model_name)
            except Exception as exc:  # download / load failed → stop retrying
                logger.info("Reranker disabled (model load failed): %s", exc)
                encoder = False
            state["encoder"] = encoder
        if not encoder:
            return []
        try:
            return [float(score) for score in encoder.rerank(query, texts)]
        except Exception as exc:  # runtime failure → caller keeps RRF order
            logger.info("Reranker scoring failed: %s", exc)
            return []

    return rerank
