from __future__ import annotations

import json
from importlib import resources
from typing import Any

# ~0.5 GB constant overhead for KV cache + inference runtime (llama.cpp/Metal).
RUNTIME_OVERHEAD_GB = 0.5

DEFAULT_QUANT = "Q4_K_M"  # what `ollama pull <model>` ships by default

_QUANT_LEVELS: tuple[tuple[str, int, float, str], ...] = (
    ("Q2_K", 2, 0.3125, "low"),
    ("Q3_K_M", 3, 0.4375, "moderate"),
    ("Q4_K_M", 4, 0.5, "good"),
    ("Q5_K_M", 5, 0.625, "good"),
    ("Q6_K", 6, 0.75, "excellent"),
    ("Q8_0", 8, 1.0, "excellent"),
    ("F16", 16, 2.0, "lossless"),
)


def make_quants(params_b: float) -> list[dict[str, Any]]:
    """VRAM/disk needs per quantization level (exact canirun makeQuants port)."""
    total_params = params_b * 1_000_000_000
    quants: list[dict[str, Any]] = []
    for name, bits, bpp, quality in _QUANT_LEVELS:
        raw = (total_params * bpp) / (1024**3)
        quants.append(
            {
                "name": name,
                "bits": bits,
                "vram_gb": round(max(raw * 1.1 + RUNTIME_OVERHEAD_GB, 0.5), 1),
                "disk_gb": round(max(raw * 1.05, 0.1), 1),
                "quality": quality,
            }
        )
    return quants


def load_model_catalog() -> list[dict[str, Any]]:
    payload = resources.files("lity.resources").joinpath("model_catalog.json")
    rows = json.loads(payload.read_text(encoding="utf-8"))
    catalog: list[dict[str, Any]] = []
    for row in rows:
        model = dict(row)
        model["quants"] = make_quants(float(model["params_b"]))
        catalog.append(model)
    return catalog
