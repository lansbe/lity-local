from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Exact port of canirun.ai's scoring pipeline (packages/compatibility):
# status → tokens/s → memory headroom → score → S–F grade. Same thresholds,
# same formulas, so Lity and canirun.ai give the SAME verdict for the same
# device. Only the hardware source differs: canirun probes the browser, Lity
# probes the OS (more precise).

# can-run / tight / can-run-slow / cannot-run / unknown
ModelStatus = str
Grade = str

# DDR5 dual-channel system RAM bandwidth assumed for CPU-offloaded layers.
SYSTEM_RAM_BW_GBS = 50.0


@dataclass
class HardwareProfile:
    """The minimal device facts the evaluation needs."""

    total_usable_ram_gb: float | None = None  # unified/system memory (GB)
    estimated_vram_gb: float | None = None  # dedicated VRAM (GB), discrete GPUs
    system_ram_gb: float | None = None  # system RAM behind a discrete GPU
    memory_bandwidth_gbs: float | None = None  # GB/s — drives tokens/s
    is_apple_silicon: bool = False
    is_mobile: bool = False
    platform: str | None = None


def profile_from_hardware(hardware: dict[str, Any]) -> HardwareProfile:
    """Adapt Lity's ``detect_hardware()`` dict into a HardwareProfile."""
    accelerator = str(hardware.get("accelerator", "cpu"))
    ram = float(hardware.get("ram_gb") or 0) or None
    if ram is None:
        budget = float(hardware.get("budget_gb") or 0)
        ram = round(budget / 0.7, 1) if budget else None  # budget is 70% of RAM
    vram = float(hardware.get("vram_gb") or 0) or None
    bandwidth = float(hardware.get("memory_bandwidth") or 0) or None
    is_apple = accelerator == "metal"
    if accelerator in ("cuda", "rocm") and vram:
        return HardwareProfile(
            total_usable_ram_gb=vram,
            estimated_vram_gb=vram,
            system_ram_gb=ram,
            memory_bandwidth_gbs=bandwidth,
            platform=hardware.get("os"),
        )
    return HardwareProfile(
        total_usable_ram_gb=ram,
        memory_bandwidth_gbs=bandwidth,
        is_apple_silicon=is_apple,
        platform=hardware.get("os"),
    )


def evaluate_status(vram_needed_gb: float, hw: HardwareProfile) -> ModelStatus:
    """Run status for a model needing ``vram_needed_gb`` (canirun evaluateModel)."""
    if hw.is_mobile and not hw.is_apple_silicon and hw.total_usable_ram_gb:
        factor = 0.50 if hw.platform == "iOS" else 0.55
        usable = hw.total_usable_ram_gb * factor
        if vram_needed_gb <= usable * 0.7:
            return "can-run"
        if vram_needed_gb <= usable:
            return "tight"
        return "cannot-run"
    if hw.is_apple_silicon and hw.total_usable_ram_gb:
        usable = hw.total_usable_ram_gb * 0.75
        if vram_needed_gb <= usable * 0.7:
            return "can-run"
        if vram_needed_gb <= usable:
            return "tight"
        return "cannot-run"
    if hw.estimated_vram_gb:
        if vram_needed_gb <= hw.estimated_vram_gb * 0.85:
            return "can-run"
        if vram_needed_gb <= hw.estimated_vram_gb * 1.1:
            return "tight"
        # Doesn't fit in VRAM — check CPU offloading via system RAM.
        if hw.system_ram_gb and hw.system_ram_gb > hw.estimated_vram_gb:
            total_offload = hw.estimated_vram_gb + hw.system_ram_gb * 0.70
            if vram_needed_gb <= total_offload:
                return "can-run-slow"
        return "cannot-run"
    if hw.total_usable_ram_gb:
        usable = hw.total_usable_ram_gb * 0.7
        if vram_needed_gb <= usable * 0.7:
            return "can-run"
        if vram_needed_gb <= usable:
            return "tight"
        return "cannot-run"
    return "unknown"


def estimate_tokens_per_second(model_vram_gb: float, hw: HardwareProfile) -> int | None:
    """Decode speed ≈ bandwidth / model size × efficiency (canirun formula)."""
    if not hw.memory_bandwidth_gbs or model_vram_gb <= 0:
        return None
    if hw.is_mobile and not hw.is_apple_silicon:
        efficiency = 0.40
    elif hw.is_apple_silicon:
        efficiency = 0.65
    else:
        efficiency = 0.70

    # Offloaded models: harmonic mean of the VRAM and system-RAM paths — the
    # slower path bottlenecks — plus a PCIe transfer penalty.
    if hw.estimated_vram_gb and model_vram_gb > hw.estimated_vram_gb and hw.system_ram_gb:
        fraction_vram = min(1.0, hw.estimated_vram_gb / model_vram_gb)
        fraction_ram = 1.0 - fraction_vram
        effective_bw = 1.0 / (
            fraction_vram / hw.memory_bandwidth_gbs + fraction_ram / SYSTEM_RAM_BW_GBS
        )
        return max(1, round((effective_bw / model_vram_gb) * efficiency * 0.85))

    return round((hw.memory_bandwidth_gbs / model_vram_gb) * efficiency)


def memory_percentage(vram_needed_gb: float, hw: HardwareProfile) -> int | None:
    if hw.is_mobile or hw.is_apple_silicon:
        if not hw.total_usable_ram_gb:
            return None
        return round((vram_needed_gb / hw.total_usable_ram_gb) * 100)
    reference = hw.estimated_vram_gb or hw.total_usable_ram_gb
    if not reference:
        return None
    return round((vram_needed_gb / reference) * 100)  # >100% when offloading


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


def compute_score(
    status: ModelStatus,
    toks_per_sec: int | None,
    params_billions: float,
    mem_pct: int | None = None,
) -> int:
    """0–100 score: speed (55%) + memory headroom (35%) + quality bonus."""
    if status in ("cannot-run", "unknown"):
        return 0

    if toks_per_sec is not None:
        t = float(toks_per_sec)
        if t >= 80:
            speed_score = 100.0
        elif t >= 40:
            speed_score = _lerp(t, 40, 80, 80, 100)
        elif t >= 20:
            speed_score = _lerp(t, 20, 40, 55, 80)
        elif t >= 10:
            speed_score = _lerp(t, 10, 20, 35, 55)
        elif t >= 5:
            speed_score = _lerp(t, 5, 10, 15, 35)
        else:
            speed_score = _lerp(max(t, 0.0), 0, 5, 0, 15)
    else:
        speed_score = 45.0 if status == "can-run" else 20.0

    headroom_score = 45.0
    if mem_pct is not None:
        p = float(mem_pct)
        if p <= 20:
            headroom_score = 100.0
        elif p <= 40:
            headroom_score = _lerp(p, 20, 40, 100, 75)
        elif p <= 60:
            headroom_score = _lerp(p, 40, 60, 75, 45)
        elif p <= 80:
            headroom_score = _lerp(p, 60, 80, 45, 20)
        else:
            headroom_score = _lerp(min(p, 100.0), 80, 100, 20, 0)

    quality_bonus = min(12.0, math.log2(params_billions + 1) * 2)
    fit_multiplier = 0.60 if status == "can-run-slow" else 0.75 if status == "tight" else 1.0
    return round((speed_score * 0.55 + headroom_score * 0.35 + quality_bonus) * fit_multiplier)


def score_to_grade(score: int, status: ModelStatus) -> Grade:
    if status == "cannot-run":
        return "F"
    if status == "unknown":
        return "?"
    if status == "can-run-slow":
        return "C" if score >= 40 else "D"
    if score >= 85:
        return "S"
    if score >= 70:
        return "A"
    if score >= 55:
        return "B"
    if score >= 40:
        return "C"
    if score >= 20:
        return "D"
    return "F"


GRADES: dict[Grade, dict[str, str]] = {
    "S": {"letter": "S", "label": "Tourne parfaitement", "color": "#22c55e"},
    "A": {"letter": "A", "label": "Tourne très bien", "color": "#4ade80"},
    "B": {"letter": "B", "label": "Correct", "color": "#a3e635"},
    "C": {"letter": "C", "label": "Juste", "color": "#f59e0b"},
    "D": {"letter": "D", "label": "Tourne à peine", "color": "#f97316"},
    "F": {"letter": "F", "label": "Trop lourd", "color": "#ef4444"},
    "?": {"letter": "?", "label": "Inconnu", "color": "#56565f"},
}


def evaluate_model_complete(
    vram_gb: float, hw: HardwareProfile, params_billions: float
) -> dict[str, Any]:
    """status + tokens/s + memory % + score + grade in one call."""
    status = evaluate_status(vram_gb, hw)
    toks = estimate_tokens_per_second(vram_gb, hw)
    mem_pct = memory_percentage(vram_gb, hw)
    score = compute_score(status, toks, params_billions, mem_pct)
    return {
        "status": status,
        "tokens_per_sec": toks,
        "mem_pct": mem_pct,
        "score": score,
        "grade": score_to_grade(score, status),
    }
