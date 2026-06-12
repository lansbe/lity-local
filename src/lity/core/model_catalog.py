from __future__ import annotations

from typing import Any

from lity.core.model_catalog_loader import (
    DEFAULT_QUANT,
    RUNTIME_OVERHEAD_GB,
    load_model_catalog,
    make_quants,
)

__all__ = [
    "DEFAULT_QUANT",
    "FULL_CATALOG",
    "RUNTIME_OVERHEAD_GB",
    "default_quant",
    "find_model",
    "installable_models",
    "is_embedding_model",
    "is_vision_model",
    "make_quants",
]

# Full open-weight model catalog, ported from canirun.ai (packages/models) and
# completed with Lity-specific entries (Ollama embeddings, vision and reasoning
# tags the tests rely on). Static rows live in lity.resources; quantized
# VRAM/disk estimates are computed by model_catalog_loader.
FULL_CATALOG: list[dict[str, Any]] = load_model_catalog()


def default_quant(model: dict[str, Any]) -> dict[str, Any]:
    """The quant Ollama installs by default (Q4_K_M)."""
    for quant in model["quants"]:
        if quant["name"] == DEFAULT_QUANT:
            return quant
    return model["quants"][0]


def installable_models() -> list[dict[str, Any]]:
    """Models Lity can actually pull through Ollama."""
    return [model for model in FULL_CATALOG if model.get("ollama_id")]


def find_model(name: str) -> dict[str, Any] | None:
    """Look a model up by Ollama tag (exact, then base-name match)."""
    cleaned = (name or "").strip().lower()
    if not cleaned:
        return None
    for model in FULL_CATALOG:
        if (model.get("ollama_id") or "").lower() == cleaned:
            return model
    base = cleaned.split(":", 1)[0]
    for model in FULL_CATALOG:
        ollama_id = (model.get("ollama_id") or "").lower()
        if ollama_id and ollama_id.split(":", 1)[0] == base:
            return model
    return None


# Embedding models advertise themselves with these markers even when absent
# from the catalog (Ollama hosts many community embedding models).
_EMBED_NAME_MARKERS = (
    "embed",
    "bge-",
    "bge:",
    "e5-",
    "gte-",
    "minilm",
    "arctic-embed",
    "mxbai",
    "paraphrase-",
)


def is_embedding_model(name: str) -> bool:
    """True when an Ollama tag designates an EMBEDDING model (not chat-capable).

    Catalog kind wins when known; otherwise common naming markers decide. Used
    to keep embeddings out of the chat-model selector and the LLM ranking."""
    cleaned = (name or "").strip().lower()
    if not cleaned:
        return False
    model = find_model(cleaned)
    if model is not None:
        return model["kind"] == "embed"
    return any(marker in cleaned for marker in _EMBED_NAME_MARKERS)


# Vision (multimodal) models advertise themselves with these markers even when
# absent from the catalog. This is only a FALLBACK heuristic for when Ollama's
# authoritative ``capabilities`` can't be queried — Gemma 3/4 and most recent
# families are multimodal from the ground up.
_VISION_NAME_MARKERS = (
    "llava",
    "bakllava",
    "vision",
    "moondream",
    "minicpm-v",
    "pixtral",
    "gemma3",
    "gemma4",
    "qwen2-vl",
    "qwen2.5vl",
    "qwen2.5-vl",
    "llama4",
    "llama-4",
    "mistral-small3",
    "granite3.2-vision",
)


def is_vision_model(name: str) -> bool:
    """True when an Ollama tag designates a VISION (multimodal) model.

    Catalog ``kind == "vision"`` wins when known; otherwise common naming
    markers decide. This is a NAME-based fallback: the authoritative check is
    Ollama's reported ``capabilities`` (see ``AIEngine.supports_vision``), used
    whenever the live engine is reachable."""
    cleaned = (name or "").strip().lower()
    if not cleaned:
        return False
    model = find_model(cleaned)
    if model is not None:
        return model["kind"] == "vision"
    return any(marker in cleaned for marker in _VISION_NAME_MARKERS)
