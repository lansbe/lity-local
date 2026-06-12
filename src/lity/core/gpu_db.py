from __future__ import annotations

import re

# Ported VERBATIM from canirun.ai (packages/compatibility/src/index.ts) so the
# two projects share one source of truth for hardware capabilities. Values are
# official vendor specs: VRAM in GB, memory bandwidth in GB/s, shader cores.
# Apple entries are the top-bin configuration of each chip family; RAM there is
# only a fallback — the real installed RAM (detected) always wins.

GPU_DB: dict[str, dict[str, float]] = {
    "RTX 5090": {"vram": 32, "bw": 1792, "cores": 21760},
    "RTX 5080": {"vram": 16, "bw": 960, "cores": 10752},
    "RTX 5070 Ti": {"vram": 16, "bw": 896, "cores": 8960},
    "RTX 5070": {"vram": 12, "bw": 672, "cores": 6144},
    "RTX 5060 Ti 16GB": {"vram": 16, "bw": 448, "cores": 4608},
    "RTX 5060 Ti": {"vram": 8, "bw": 448, "cores": 4608},
    "RTX 5060": {"vram": 8, "bw": 448, "cores": 3840},
    "RTX 5050": {"vram": 8, "bw": 320, "cores": 2560},
    "RTX 4090": {"vram": 24, "bw": 1008, "cores": 16384},
    "RTX 4080 SUPER": {"vram": 16, "bw": 736, "cores": 10240},
    "RTX 4080": {"vram": 16, "bw": 717, "cores": 9728},
    "RTX 4070 Ti SUPER": {"vram": 16, "bw": 672, "cores": 8448},
    "RTX 4070 Ti": {"vram": 12, "bw": 504, "cores": 7680},
    "RTX 4070 SUPER": {"vram": 12, "bw": 504, "cores": 7168},
    "RTX 4070": {"vram": 12, "bw": 504, "cores": 5888},
    "RTX 4060 Ti 16GB": {"vram": 16, "bw": 288, "cores": 4352},
    "RTX 4060 Ti": {"vram": 8, "bw": 288, "cores": 4352},
    "RTX 4060": {"vram": 8, "bw": 272, "cores": 3072},
    "RTX 3090 Ti": {"vram": 24, "bw": 1008, "cores": 10752},
    "RTX 3090": {"vram": 24, "bw": 936, "cores": 10496},
    "RTX 3080 Ti": {"vram": 12, "bw": 912, "cores": 10240},
    "RTX 3080 12GB": {"vram": 12, "bw": 912, "cores": 8960},
    "RTX 3080": {"vram": 10, "bw": 760, "cores": 8704},
    "RTX 3070 Ti": {"vram": 8, "bw": 608, "cores": 6144},
    "RTX 3070": {"vram": 8, "bw": 448, "cores": 5888},
    "RTX 3060 Ti": {"vram": 8, "bw": 448, "cores": 4864},
    "RTX 3060": {"vram": 12, "bw": 360, "cores": 3584},
    "RTX 3050": {"vram": 8, "bw": 224, "cores": 2560},
    "RTX 5090 Laptop": {"vram": 24, "bw": 896, "cores": 10496},
    "RTX 5080 Laptop": {"vram": 16, "bw": 896, "cores": 7680},
    "RTX 5070 Ti Laptop": {"vram": 12, "bw": 672, "cores": 5888},
    "RTX 5070 Laptop": {"vram": 8, "bw": 384, "cores": 4608},
    "RTX 5060 Laptop": {"vram": 8, "bw": 384, "cores": 3328},
    "RTX 5050 Laptop": {"vram": 8, "bw": 384, "cores": 2560},
    "RTX 4090 Laptop": {"vram": 16, "bw": 576, "cores": 9728},
    "RTX 4080 Laptop": {"vram": 12, "bw": 432, "cores": 7424},
    "RTX 4070 Laptop": {"vram": 8, "bw": 256, "cores": 4608},
    "RTX 4060 Laptop": {"vram": 8, "bw": 256, "cores": 3072},
    "RTX 4050 Laptop": {"vram": 6, "bw": 192, "cores": 2560},
    "RTX 3080 Ti Laptop": {"vram": 16, "bw": 512, "cores": 7424},
    "RTX 3080 Laptop": {"vram": 16, "bw": 448, "cores": 6144},
    "RTX 3070 Ti Laptop": {"vram": 8, "bw": 448, "cores": 5888},
    "RTX 3070 Laptop": {"vram": 8, "bw": 448, "cores": 5120},
    "RTX 3060 Laptop": {"vram": 6, "bw": 336, "cores": 3840},
    "RTX 3050 Ti Laptop": {"vram": 4, "bw": 192, "cores": 2560},
    "RTX 3050 Laptop": {"vram": 4, "bw": 192, "cores": 2048},
    "RTX PRO 6000": {"vram": 96, "bw": 1792, "cores": 24064},
    "RTX 6000 Ada": {"vram": 48, "bw": 960, "cores": 18176},
    "RTX 5880 Ada": {"vram": 48, "bw": 960, "cores": 14080},
    "RTX 5000 Ada": {"vram": 32, "bw": 800, "cores": 12800},
    "RTX 4500 Ada": {"vram": 24, "bw": 432, "cores": 7680},
    "RTX 4000 SFF Ada": {"vram": 20, "bw": 320, "cores": 6144},
    "RTX 4000 Ada": {"vram": 20, "bw": 360, "cores": 6144},
    "RTX 3500 Ada": {"vram": 12, "bw": 432, "cores": 5120},
    "RTX 2000 Ada": {"vram": 16, "bw": 224, "cores": 2816},
    "RTX A6000": {"vram": 48, "bw": 768, "cores": 10752},
    "RTX A5500": {"vram": 24, "bw": 768, "cores": 10240},
    "RTX A5000": {"vram": 24, "bw": 768, "cores": 8192},
    "RTX A4500": {"vram": 20, "bw": 640, "cores": 7168},
    "RTX A4000": {"vram": 16, "bw": 448, "cores": 6144},
    "RTX A2000": {"vram": 6, "bw": 288, "cores": 3328},
    "RTX 2080 Ti": {"vram": 11, "bw": 616, "cores": 4352},
    "RTX 2080 SUPER": {"vram": 8, "bw": 496, "cores": 3072},
    "RTX 2080": {"vram": 8, "bw": 448, "cores": 2944},
    "RTX 2070 SUPER": {"vram": 8, "bw": 448, "cores": 2560},
    "RTX 2070": {"vram": 8, "bw": 448, "cores": 2304},
    "RTX 2060 SUPER": {"vram": 8, "bw": 448, "cores": 2176},
    "RTX 2060": {"vram": 6, "bw": 336, "cores": 1920},
    "RTX 2060 12GB": {"vram": 12, "bw": 336, "cores": 2176},
    "RTX 3050 6GB": {"vram": 6, "bw": 168, "cores": 2304},
    "A100": {"vram": 80, "bw": 2039, "cores": 6912},
    "H100": {"vram": 80, "bw": 3350, "cores": 14592},
    "GH200": {"vram": 96, "bw": 4000, "cores": 16896},
    "DGX Spark": {"vram": 128, "bw": 273, "cores": 6144},
    "L40S": {"vram": 48, "bw": 864, "cores": 18176},
    "L4": {"vram": 24, "bw": 300, "cores": 7424},
    "T4": {"vram": 16, "bw": 300, "cores": 2560},
    "Tesla P40": {"vram": 24, "bw": 346, "cores": 3840},
    "RX 7900 XTX": {"vram": 24, "bw": 960, "cores": 6144},
    "RX 7900 XT": {"vram": 20, "bw": 800, "cores": 5376},
    "RX 7800 XT": {"vram": 16, "bw": 624, "cores": 3840},
    "RX 7700 XT": {"vram": 12, "bw": 432, "cores": 3456},
    "RX 7600 XT": {"vram": 16, "bw": 288, "cores": 2048},
    "RX 7600": {"vram": 8, "bw": 288, "cores": 2048},
    "RX 6900 XT": {"vram": 16, "bw": 512, "cores": 5120},
    "RX 6800 XT": {"vram": 16, "bw": 512, "cores": 4608},
    "RX 6800": {"vram": 16, "bw": 512, "cores": 3840},
    "RX 6750 XT": {"vram": 12, "bw": 432, "cores": 2560},
    "RX 6700 XT": {"vram": 12, "bw": 384, "cores": 2560},
    "RX 6650 XT": {"vram": 8, "bw": 280, "cores": 2048},
    "RX 6600 XT": {"vram": 8, "bw": 256, "cores": 2048},
    "RX 6600": {"vram": 8, "bw": 224, "cores": 1792},
    "RX 6500 XT": {"vram": 4, "bw": 144, "cores": 1024},
    "Arc A770": {"vram": 16, "bw": 560, "cores": 4096},
    "Arc A750": {"vram": 8, "bw": 512, "cores": 3584},
    "Arc A580": {"vram": 8, "bw": 512, "cores": 3072},
    "Arc A380": {"vram": 6, "bw": 186, "cores": 1024},
    "GTX 1660 Ti": {"vram": 6, "bw": 288, "cores": 1536},
    "GTX 1660 SUPER": {"vram": 6, "bw": 336, "cores": 1408},
    "GTX 1660": {"vram": 6, "bw": 192, "cores": 1408},
    "GTX 1650 SUPER": {"vram": 4, "bw": 192, "cores": 1280},
    "GTX 1650 Ti": {"vram": 4, "bw": 192, "cores": 1024},
    "GTX 1650": {"vram": 4, "bw": 128, "cores": 896},
    "GTX 1630": {"vram": 4, "bw": 96, "cores": 512},
    "GTX 1080 Ti": {"vram": 11, "bw": 484, "cores": 3584},
    "GTX 1080": {"vram": 8, "bw": 320, "cores": 2560},
    "GTX 1070 Ti": {"vram": 8, "bw": 256, "cores": 2432},
    "GTX 1070": {"vram": 8, "bw": 256, "cores": 1920},
    "GTX 1060 6GB": {"vram": 6, "bw": 192, "cores": 1280},
    "GTX 1060 3GB": {"vram": 3, "bw": 192, "cores": 1152},
    "GTX 1060": {"vram": 6, "bw": 192, "cores": 1280},
    "GTX 1050 Ti": {"vram": 4, "bw": 112, "cores": 768},
    "GTX 1050": {"vram": 2, "bw": 112, "cores": 640},
    "GTX 980 Ti": {"vram": 6, "bw": 336, "cores": 2816},
    "GTX 980": {"vram": 4, "bw": 224, "cores": 2048},
    "GTX 970": {"vram": 4, "bw": 224, "cores": 1664},
    "GTX 960": {"vram": 2, "bw": 112, "cores": 1024},
    "GTX 950": {"vram": 2, "bw": 105, "cores": 768},
    "Quadro RTX 8000": {"vram": 48, "bw": 672, "cores": 4608},
    "Quadro RTX 6000": {"vram": 24, "bw": 672, "cores": 4608},
    "Quadro RTX 5000": {"vram": 16, "bw": 448, "cores": 3072},
    "Quadro RTX 4000": {"vram": 8, "bw": 416, "cores": 2304},
    "Quadro RTX 3000": {"vram": 6, "bw": 336, "cores": 1920},
    "Quadro T2000": {"vram": 4, "bw": 128, "cores": 1024},
    "Quadro T1000": {"vram": 4, "bw": 128, "cores": 896},
    "T1200": {"vram": 4, "bw": 192, "cores": 1024},
    "NVIDIA T600": {"vram": 4, "bw": 192, "cores": 896},
    "NVIDIA T550": {"vram": 4, "bw": 112, "cores": 1024},
    "NVIDIA T500": {"vram": 4, "bw": 80, "cores": 896},
    "Quadro P5200": {"vram": 16, "bw": 230, "cores": 2560},
    "Quadro P5000": {"vram": 16, "bw": 288, "cores": 2560},
    "Quadro P4200": {"vram": 8, "bw": 224, "cores": 1792},
    "Quadro P4000": {"vram": 8, "bw": 192, "cores": 1792},
    "Quadro P3000": {"vram": 6, "bw": 168, "cores": 1280},
    "Quadro P3200": {"vram": 6, "bw": 192, "cores": 1792},
    "Quadro P2000": {"vram": 5, "bw": 140, "cores": 1024},
    "Quadro P1000": {"vram": 4, "bw": 82, "cores": 640},
    "Quadro P620": {"vram": 4, "bw": 96, "cores": 512},
    "Quadro P600": {"vram": 2, "bw": 64, "cores": 384},
    "Quadro P520": {"vram": 2, "bw": 48, "cores": 384},
    "Quadro P500": {"vram": 2, "bw": 64, "cores": 256},
    "Quadro M5500": {"vram": 8, "bw": 211, "cores": 2048},
    "Quadro M5000M": {"vram": 8, "bw": 160, "cores": 1536},
    "Quadro M4000M": {"vram": 4, "bw": 160, "cores": 1024},
    "Quadro M3000M": {"vram": 4, "bw": 160, "cores": 1024},
    "Quadro M2200": {"vram": 4, "bw": 140, "cores": 1024},
    "Quadro M2000M": {"vram": 4, "bw": 80, "cores": 640},
    "Quadro M1200": {"vram": 4, "bw": 128, "cores": 640},
    "Quadro M1000M": {"vram": 2, "bw": 80, "cores": 512},
    "Quadro M620": {"vram": 2, "bw": 80, "cores": 512},
    "Quadro M600M": {"vram": 2, "bw": 64, "cores": 384},
    "Quadro M520": {"vram": 1, "bw": 40, "cores": 384},
    "Quadro M500M": {"vram": 2, "bw": 16, "cores": 384},
    "Quadro K5100M": {"vram": 8, "bw": 160, "cores": 1536},
    "Quadro K5000M": {"vram": 4, "bw": 173, "cores": 1344},
    "Quadro K4100M": {"vram": 4, "bw": 115, "cores": 1152},
    "Quadro K4000M": {"vram": 4, "bw": 134, "cores": 960},
    "Quadro K3100M": {"vram": 4, "bw": 80, "cores": 768},
    "Quadro K3000M": {"vram": 2, "bw": 80, "cores": 576},
    "Quadro K2100M": {"vram": 2, "bw": 48, "cores": 576},
    "Quadro K2000M": {"vram": 2, "bw": 64, "cores": 384},
    "Quadro K1100M": {"vram": 2, "bw": 64, "cores": 384},
    "Quadro K1000M": {"vram": 2, "bw": 64, "cores": 384},
    "Quadro K620M": {"vram": 2, "bw": 16, "cores": 384},
    "Quadro K610M": {"vram": 1, "bw": 29, "cores": 192},
    "Quadro K510M": {"vram": 1, "bw": 19.2, "cores": 192},
    "Quadro K500M": {"vram": 2, "bw": 28.8, "cores": 192},
    "RTX A3000": {"vram": 6, "bw": 192, "cores": 4096},
    "RTX A3000 12GB": {"vram": 12, "bw": 336, "cores": 4096},
    "RTX A2000 8GB": {"vram": 8, "bw": 224, "cores": 2560},
    "RTX A1000": {"vram": 4, "bw": 224, "cores": 2048},
    "RTX A500": {"vram": 4, "bw": 112, "cores": 2048},
    "RX 5700 XT": {"vram": 8, "bw": 448, "cores": 2560},
    "RX 5700": {"vram": 8, "bw": 448, "cores": 2304},
    "RX 5600 XT": {"vram": 6, "bw": 288, "cores": 2304},
    "RX 5500 XT": {"vram": 8, "bw": 224, "cores": 1408},
    "RX 590": {"vram": 8, "bw": 256, "cores": 2304},
    "RX 580": {"vram": 8, "bw": 256, "cores": 2304},
    "RX 570": {"vram": 4, "bw": 224, "cores": 2048},
    "RX 560": {"vram": 4, "bw": 112, "cores": 1024},
    "Radeon VII": {"vram": 16, "bw": 1024, "cores": 3840},
    "Vega 64": {"vram": 8, "bw": 484, "cores": 4096},
    "Vega 56": {"vram": 8, "bw": 410, "cores": 3584},
    "RX 9070 XT": {"vram": 16, "bw": 640, "cores": 4096},
    "RX 9070": {"vram": 16, "bw": 640, "cores": 3584},
    "RX 7900M": {"vram": 16, "bw": 720, "cores": 4608},
    "RX 7700S": {"vram": 8, "bw": 288, "cores": 2048},
    "RX 7600M XT": {"vram": 8, "bw": 288, "cores": 2048},
    "RX 7600M": {"vram": 8, "bw": 288, "cores": 1792},
    "RX 7600S": {"vram": 8, "bw": 288, "cores": 1792},
    "RX 6800M": {"vram": 12, "bw": 384, "cores": 2560},
    "RX 6700M": {"vram": 10, "bw": 320, "cores": 2304},
    "RX 6600M": {"vram": 8, "bw": 224, "cores": 1792},
    "RX 6500M": {"vram": 4, "bw": 144, "cores": 1024},
    "Ryzen AI MAX+ 395": {"vram": 96, "bw": 256, "cores": 2560},
    "Radeon 890M": {"vram": 0, "bw": 89, "cores": 1024},
    "Radeon 880M": {"vram": 0, "bw": 89, "cores": 768},
    "Radeon 780M": {"vram": 0, "bw": 89, "cores": 768},
    "Radeon 760M": {"vram": 0, "bw": 89, "cores": 512},
    "Radeon 680M": {"vram": 0, "bw": 77, "cores": 768},
    "Radeon 660M": {"vram": 0, "bw": 77, "cores": 384},
    "Vega 8": {"vram": 0, "bw": 51, "cores": 512},
    "Vega 7": {"vram": 0, "bw": 51, "cores": 448},
    "Arc A770M": {"vram": 16, "bw": 512, "cores": 4096},
    "Arc A550M": {"vram": 8, "bw": 224, "cores": 2048},
    "Arc A370M": {"vram": 4, "bw": 112, "cores": 1024},
    "Iris Xe": {"vram": 0, "bw": 68, "cores": 96},
    "Iris Plus": {"vram": 0, "bw": 50, "cores": 64},
    "UHD 770": {"vram": 0, "bw": 76, "cores": 32},
    "UHD 730": {"vram": 0, "bw": 76, "cores": 24},
    "UHD Graphics 630": {"vram": 0, "bw": 42, "cores": 24},
    "UHD Graphics 620": {"vram": 0, "bw": 34, "cores": 24},
}

APPLE_DB: dict[str, dict[str, float]] = {
    "m5 max": {"ram": 36, "bw": 614, "cpu_cores": 18, "gpu_cores": 40},
    "m5 pro": {"ram": 24, "bw": 307, "cpu_cores": 18, "gpu_cores": 20},
    "m5": {"ram": 16, "bw": 153, "cpu_cores": 10, "gpu_cores": 10},
    "m4 max": {"ram": 36, "bw": 546, "cpu_cores": 16, "gpu_cores": 40},
    "m4 pro": {"ram": 24, "bw": 273, "cpu_cores": 14, "gpu_cores": 20},
    "m4": {"ram": 16, "bw": 120, "cpu_cores": 10, "gpu_cores": 10},
    "m3 ultra": {"ram": 96, "bw": 819, "cpu_cores": 32, "gpu_cores": 80},
    "m3 max": {"ram": 36, "bw": 400, "cpu_cores": 16, "gpu_cores": 40},
    "m3 pro": {"ram": 18, "bw": 150, "cpu_cores": 12, "gpu_cores": 18},
    "m3": {"ram": 8, "bw": 100, "cpu_cores": 8, "gpu_cores": 10},
    "m2 ultra": {"ram": 64, "bw": 800, "cpu_cores": 24, "gpu_cores": 76},
    "m2 max": {"ram": 32, "bw": 400, "cpu_cores": 12, "gpu_cores": 38},
    "m2 pro": {"ram": 16, "bw": 200, "cpu_cores": 12, "gpu_cores": 19},
    "m2": {"ram": 8, "bw": 100, "cpu_cores": 8, "gpu_cores": 10},
    "m1 ultra": {"ram": 64, "bw": 800, "cpu_cores": 20, "gpu_cores": 64},
    "m1 max": {"ram": 32, "bw": 400, "cpu_cores": 10, "gpu_cores": 32},
    "m1 pro": {"ram": 16, "bw": 200, "cpu_cores": 10, "gpu_cores": 16},
    "m1": {"ram": 8, "bw": 68, "cpu_cores": 8, "gpu_cores": 8},
}


def match_gpu(name: str | None) -> dict[str, float] | None:
    """Longest-name match against GPU_DB (same rule as canirun's matchGPU)."""
    if not name:
        return None
    upper = re.sub(r"\s+", " ", name.upper().replace("(TM)", "")).strip()
    best: dict[str, float] | None = None
    best_len = 0
    for entry_name, data in GPU_DB.items():
        if entry_name.upper() in upper and len(entry_name) > best_len:
            best = data
            best_len = len(entry_name)
    return best


def match_apple_chip(text: str | None) -> tuple[str, dict[str, float]] | None:
    """Identify an Apple Silicon chip key ("m2 pro", "m4 max"…) inside a brand
    string like "Apple M2 Pro 12-core" and return its APPLE_DB entry."""
    if not text:
        return None
    lower = text.lower()
    match = re.search(r"\bm(\d)\b(?:\s+(pro|max|ultra))?", lower)
    if not match:
        return None
    key = f"m{match.group(1)}" + (f" {match.group(2)}" if match.group(2) else "")
    entry = APPLE_DB.get(key)
    return (key, entry) if entry else None


def bandwidth_heuristic(
    gpu_name: str | None,
    vram_gb: float | None,
    accelerator: str,
) -> float | None:
    """Vendor/name/VRAM-based bandwidth estimate when the GPU is not in the DB
    (port of canirun's estimateBandwidthHeuristic)."""
    upper = (gpu_name or "").upper()
    if accelerator == "metal":
        return 68.0  # unknown Apple chip → M1 baseline
    is_nvidia = any(tag in upper for tag in ("NVIDIA", "GEFORCE", "RTX", "GTX"))
    is_amd = "AMD" in upper or "RADEON" in upper or accelerator == "rocm"
    is_intel = "INTEL" in upper
    if is_intel and any(tag in upper for tag in ("UHD", "IRIS", "HD GRAPHICS")):
        return 50.0
    if is_nvidia:
        if "RTX" in upper:
            if vram_gb and vram_gb >= 20:
                return 700.0
            if vram_gb and vram_gb >= 12:
                return 450.0
            if vram_gb and vram_gb >= 8:
                return 300.0
            return 250.0
        if "GTX" in upper:
            if vram_gb and vram_gb >= 8:
                return 250.0
            if vram_gb and vram_gb >= 4:
                return 150.0
            return 112.0
        if vram_gb and vram_gb >= 8:
            return 300.0
        return 150.0
    if is_amd:
        if "RADEON GRAPHICS" in upper:
            return 55.0
        if vram_gb and vram_gb >= 16:
            return 500.0
        if vram_gb and vram_gb >= 8:
            return 300.0
        if vram_gb and vram_gb >= 4:
            return 180.0
        return 150.0
    # Generic desktop fallback: DDR5 dual-channel.
    return 60.0
