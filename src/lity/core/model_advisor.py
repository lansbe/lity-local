from __future__ import annotations

from typing import Any

from lity.core.compatibility import (
    GRADES,
    evaluate_model_complete,
    profile_from_hardware,
)
from lity.core.model_catalog import (
    RUNTIME_OVERHEAD_GB,
    default_quant,
    find_model,
    installable_models,
)

# Backward-compatible flat view of the catalog (name / params_b / size_gb /
# kind). The full data — quants, context, licence, MoE — lives in
# model_catalog.FULL_CATALOG; the matching logic in core.compatibility.
MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "name": model["ollama_id"],
        "params_b": model["params_b"],
        "size_gb": default_quant(model)["disk_gb"],
        "kind": model["kind"],
    }
    for model in installable_models()
]

VERDICTS = ("excellent", "bon", "limite", "trop_lourd")
_CONTEXT_OVERHEAD = 1.15  # KV cache / context headroom on top of the weights

# Tool-calling support is a FAMILY trait on Ollama. This local prior GATES the
# per-model `tools` flag from the catalog: canirun reports what the WEIGHTS can
# do, but e.g. gemma3 or llama3.2-vision do not reliably tool-call through
# Ollama, which is what matters for Lity's agent mode.
_TOOL_CAPABLE = (
    "qwen3",
    "qwen2.5",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral",
    "mixtral",
    "command-r",
    "firefunction",
)
_TOOL_INCAPABLE = ("vision", "llava", "deepseek-r1", "gemma", "phi")


def supports_tools(name: str, kind: str = "") -> bool:
    """Offline prior: does this model family reliably tool-call on Ollama?"""
    lowered = (name or "").lower()
    if kind in ("embed", "vision"):
        return False
    if any(tag in lowered for tag in _TOOL_INCAPABLE):
        return False
    return any(tag in lowered for tag in _TOOL_CAPABLE)


def _tool_use(name: str, kind: str, catalog_tools: bool) -> bool:
    """Catalog `tools` flag gated by the Ollama family prior."""
    lowered = (name or "").lower()
    if kind in ("embed", "vision"):
        return False
    if any(tag in lowered for tag in _TOOL_INCAPABLE):
        return False
    if catalog_tools:
        return True
    return any(tag in lowered for tag in _TOOL_CAPABLE)


def estimate_params_from_size(size_gb: float) -> float:
    """Rough param count (in billions) from a Q4 size (~0.6 GB per B params)."""
    return round(max(size_gb, 0.1) / 0.6, 1)


def recommend_num_ctx(budget_gb: float, model_size_gb: float = 0.0) -> int:
    """Hardware-aware context window for the chat/agent engine.

    The KV cache competes with the model weights for the same memory budget, so
    the window a machine can afford depends on what is left AFTER loading the
    model. Returns 0 when the hardware is unknown ("keep the default")."""
    if budget_gb <= 0:
        return 0
    free_gb = budget_gb - (model_size_gb or 0.0) * _CONTEXT_OVERHEAD
    if free_gb >= 16:
        return 32768
    if free_gb >= 8:
        return 24576
    if free_gb >= 4:
        return 16384
    return 8192


def catalog_size_gb(model_name: str) -> float:
    """Best-effort default-quant weight size for a model name (0.0 if unknown)."""
    model = find_model(model_name)
    if model is None:
        return 0.0
    return float(default_quant(model)["disk_gb"])


def classify_model(
    size_gb: float,
    budget_gb: float,
    accelerator: str = "cpu",
    ram_gb: float | None = None,
) -> tuple[str, str]:
    """Classic (verdict, speed) classification kept for compatibility.

    New code should go through :func:`rank_models` /
    :mod:`lity.core.compatibility`, which carry the canirun.ai grades.
    """
    if budget_gb <= 0:
        return "bon", "inconnu"  # unknown hardware → stay neutral, don't scare

    required = size_gb * _CONTEXT_OVERHEAD
    ratio = required / budget_gb
    if ratio <= 0.6:
        verdict = "excellent"
    elif ratio <= 0.9:
        verdict = "bon"
    elif ratio <= 1.1:
        verdict = "limite"
    elif accelerator in ("cuda", "rocm") and ram_gb and required <= ram_gb * 0.8:
        verdict = "limite"  # exceeds VRAM but fits RAM → CPU offload, slower
    else:
        verdict = "trop_lourd"

    if accelerator in ("cuda", "rocm", "metal"):
        if verdict in ("excellent", "bon"):
            speed = "rapide (GPU)"
        elif verdict == "limite":
            speed = "ralenti (RAM)"
        else:
            speed = "ralenti"
    else:
        speed = "lent (CPU)"
    return verdict, speed


# canirun statuses → Lity's historical four-value verdict.
_STATUS_TIER = {"can-run": 0, "tight": 1, "can-run-slow": 2, "unknown": 3, "cannot-run": 4}


def _verdict(evaluation: dict[str, Any]) -> str:
    status = evaluation["status"]
    if status == "cannot-run":
        return "trop_lourd"
    if status in ("tight", "can-run-slow"):
        return "limite"
    if status == "unknown":
        return "bon"  # stay neutral on unknown hardware
    return "excellent" if evaluation["score"] >= 70 else "bon"


def _speed_label(evaluation: dict[str, Any], accelerator: str) -> str:
    toks = evaluation.get("tokens_per_sec")
    if toks:
        suffix = " (déchargé en RAM)" if evaluation["status"] == "can-run-slow" else ""
        return f"≈{toks} tok/s{suffix}"
    status = evaluation["status"]
    if status == "unknown":
        return "inconnu"
    if accelerator in ("cuda", "rocm", "metal"):
        if status == "can-run":
            return "rapide (GPU)"
        if status in ("tight", "can-run-slow"):
            return "ralenti (RAM)"
        return "ralenti"
    return "lent (CPU)"


def _best_quant(quants: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest-quality quant that still runs comfortably (then tightly)."""
    for wanted in ("can-run", "tight"):
        runnable = [quant for quant in quants if quant["status"] == wanted]
        if runnable:
            best = max(runnable, key=lambda quant: quant["bits"])
            return {"name": best["name"], "vram_gb": best["vram_gb"], "grade": best["grade"]}
    return None


def _sort_key(row: dict[str, Any]) -> tuple[int, float, float]:
    tier = _STATUS_TIER.get(row["status"], 4)
    if row["status"] == "cannot-run":
        # Unrunnable models: smallest first (closest to becoming runnable).
        return (tier, 0.0, row["params_b"])
    return (tier, -float(row["score"]), -row["params_b"])


def _parse_installed(installed: list[Any] | None) -> tuple[dict[str, float], set[str]]:
    sizes: dict[str, float] = {}
    names: set[str] = set()
    for item in installed or []:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            if not name:
                continue
            names.add(name)
            size = item.get("size")
            if size:
                sizes[name] = float(size) / (1024**3)
        elif item:
            names.add(str(item))
    return sizes, names


def rank_models(
    hardware: dict[str, Any], installed: list[Any] | None = None
) -> list[dict[str, Any]]:
    """Rank every Ollama-installable catalog model for THIS device.

    Same pipeline as canirun.ai: per-quant VRAM needs → run status → tokens/s
    from memory bandwidth → memory headroom → 0-100 score → S-F grade. Each row
    keeps Lity's historical keys (verdict/speed/tool_use/…) and adds the
    canirun-aligned ones (grade, score, status, tokens_per_sec, mem_pct,
    quants, best_quant, context_length, thinking, license).
    """
    profile = profile_from_hardware(hardware)
    accelerator = str(hardware.get("accelerator", "cpu"))
    installed_sizes, installed_names = _parse_installed(installed)

    rows: list[dict[str, Any]] = []
    catalog_names: set[str] = set()
    for model in installable_models():
        name = str(model["ollama_id"])
        catalog_names.add(name)
        headline = default_quant(model)
        evaluation = evaluate_model_complete(headline["vram_gb"], profile, model["params_b"])
        quants = [
            {
                "name": quant["name"],
                "bits": quant["bits"],
                "vram_gb": quant["vram_gb"],
                "disk_gb": quant["disk_gb"],
                "quality": quant["quality"],
                **evaluate_model_complete(quant["vram_gb"], profile, model["params_b"]),
            }
            for quant in model["quants"]
        ]
        rows.append(
            {
                "name": name,
                "display_name": model["name"],
                "provider": model["provider"],
                "params_b": model["params_b"],
                "size_gb": round(installed_sizes.get(name, headline["disk_gb"]), 1),
                "kind": model["kind"],
                "installed": name in installed_names,
                "verdict": _verdict(evaluation),
                "speed": _speed_label(evaluation, accelerator),
                "tool_use": _tool_use(name, model["kind"], bool(model["tools"])),
                "grade": evaluation["grade"],
                "grade_label": GRADES[evaluation["grade"]]["label"],
                "score": evaluation["score"],
                "status": evaluation["status"],
                "tokens_per_sec": evaluation["tokens_per_sec"],
                "mem_pct": evaluation["mem_pct"],
                "context_length": model["context_length"],
                "thinking": bool(model["thinking"]),
                "license": model["license"],
                "release_date": model.get("release_date", ""),
                "architecture": model["architecture"],
                "active_params_b": model["active_params_b"],
                "quants": quants,
                "best_quant": _best_quant(quants),
            }
        )

    # Installed models that aren't in the catalog get their own row.
    for name in sorted(installed_names - catalog_names):
        size_gb = installed_sizes.get(name, 0.0)
        params_b = estimate_params_from_size(size_gb) if size_gb else 0.0
        vram_needed = max(size_gb * 1.1 + RUNTIME_OVERHEAD_GB, 0.5)
        evaluation = evaluate_model_complete(vram_needed, profile, params_b)
        rows.append(
            {
                "name": name,
                "display_name": name,
                "provider": "",
                "params_b": params_b,
                "size_gb": round(size_gb, 1),
                "kind": "chat",
                "installed": True,
                "verdict": _verdict(evaluation),
                "speed": _speed_label(evaluation, accelerator),
                "tool_use": supports_tools(name, "chat"),
                "grade": evaluation["grade"],
                "grade_label": GRADES[evaluation["grade"]]["label"],
                "score": evaluation["score"],
                "status": evaluation["status"],
                "tokens_per_sec": evaluation["tokens_per_sec"],
                "mem_pct": evaluation["mem_pct"],
                "context_length": 0,
                "thinking": False,
                "license": "",
                "release_date": "",
                "architecture": "dense",
                "active_params_b": None,
                "quants": [],
                "best_quant": None,
            }
        )

    rows.sort(key=_sort_key)
    _flag_recommended(rows)
    return rows


# Below this decode speed an agent loop (many model calls per task) feels
# broken; a model this slow is never the right DEFAULT even if more capable.
_RECOMMEND_MIN_TOKS = 15
# Capability tier width (in B params): models within the same ~4B band are
# considered comparable, and the NEWER generation wins (an 8B from this year
# beats a 9B from last year — parameter count is a poor proxy across
# generations).
_CAPABILITY_TIER_B = 4


def _flag_recommended(rows: list[dict[str, Any]]) -> None:
    """Flag the single best DEFAULT pick: the most capable model that runs
    comfortably — tool-capable preferred (Lity is agent-centric), fast enough
    for an agent loop, newest generation within a capability tier."""
    candidates = [
        row
        for row in rows
        if row["status"] == "can-run"
        and row["grade"] in ("S", "A", "B")
        and row["kind"] in ("chat", "code", "reasoning")
    ]
    fast = [
        row
        for row in candidates
        if row["tokens_per_sec"] is None or row["tokens_per_sec"] >= _RECOMMEND_MIN_TOKS
    ]
    pool = fast or candidates
    if pool:
        best = max(
            pool,
            key=lambda row: (
                bool(row["tool_use"]),
                round(row["params_b"] / _CAPABILITY_TIER_B),
                row.get("release_date") or "",
                row["score"],
            ),
        )
        best["recommended"] = True
        return
    # Nothing runs comfortably: fall back to the first runnable row.
    for row in rows:
        if row["verdict"] in ("excellent", "bon") and row["kind"] in (
            "chat",
            "code",
            "reasoning",
        ):
            row["recommended"] = True
            break
