from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GB = 1024**3


def detect_hardware() -> dict[str, Any]:
    """Best-effort, cross-platform hardware probe to rank models for this device.

    Never raises: any field it cannot determine comes back as 0/None. Detection
    is layered per vendor/OS (NVIDIA → AMD → Intel → Apple) and uses the most
    authoritative source available, falling back gracefully.
    """
    system = platform.system()
    arch = platform.machine()
    cores = os.cpu_count() or 0

    ram_bytes = _total_ram_bytes(system)
    ram_gb = round(ram_bytes / GB, 1) if ram_bytes else 0.0

    gpu_name, vram_gb, accelerator = _detect_gpu(system, arch)
    budget_gb = round(_budget_gb(accelerator, ram_gb, vram_gb), 1)
    bandwidth, gpu_cores, chip = _bandwidth_info(gpu_name, vram_gb, accelerator)

    return {
        "os": system,
        "arch": arch,
        "cpu_cores": cores,
        "ram_gb": ram_gb,
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "accelerator": accelerator,
        "budget_gb": budget_gb,
        # canirun.ai-aligned extras: bandwidth drives the tokens/s estimate.
        "memory_bandwidth": bandwidth,
        "gpu_cores": gpu_cores,
        "chip": chip,
    }


def _bandwidth_info(
    gpu_name: str | None, vram_gb: float | None, accelerator: str
) -> tuple[float | None, int | None, str | None]:
    """Memory bandwidth (GB/s) + GPU cores from the shared canirun.ai databases.

    Apple Silicon resolves through the chip family (brand string → APPLE_DB);
    discrete GPUs through the longest-name GPU_DB match; everything else falls
    back to the vendor/VRAM heuristic. Never raises."""
    try:
        from lity.core.gpu_db import bandwidth_heuristic, match_apple_chip, match_gpu

        if accelerator == "metal":
            matched = match_apple_chip(gpu_name)
            if matched:
                key, entry = matched
                return float(entry["bw"]), int(entry["gpu_cores"]), key
            return bandwidth_heuristic(gpu_name, vram_gb, accelerator), None, None
        entry = match_gpu(gpu_name)
        if entry:
            return float(entry["bw"]), int(entry["cores"]), None
        return bandwidth_heuristic(gpu_name, vram_gb, accelerator), None, None
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("Bandwidth lookup failed: %s", exc)
        return None, None, None


# --------------------------------------------------------------------- RAM
def _total_ram_bytes(system: str) -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    try:
        if system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return int(out.strip()) if out else 0
        if system == "Linux":
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        if system == "Windows":  # pragma: no cover - platform-specific
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
    except Exception as exc:
        logger.info("RAM detection failed: %s", exc)
    return 0


# --------------------------------------------------------------------- GPU
def _detect_gpu(system: str, arch: str) -> tuple[str | None, float | None, str]:
    # 1. NVIDIA on any OS — the most precise source.
    nvidia = _parse_nvidia_smi(
        _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    )
    if nvidia is not None:
        return nvidia[0], nvidia[1], "cuda"

    # 2. Platform-specific probes for AMD / Intel / Apple.
    if system == "Darwin":
        return _macos_gpu(arch)
    if system == "Linux":
        return _linux_gpu()
    if system == "Windows":  # pragma: no cover - platform-specific
        return _windows_gpu()
    return None, None, "cpu"


def _macos_gpu(arch: str) -> tuple[str | None, float | None, str]:
    # Apple Silicon: integrated GPU, unified memory, Metal. The chip brand string
    # (fast) is enough — no need for the slow system_profiler probe here.
    if arch in ("arm64", "aarch64"):
        chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or "Apple Silicon"
        return f"{chip} — GPU intégré (mémoire unifiée)", None, "metal"

    # Intel Mac: may have a discrete GPU with dedicated VRAM.
    parsed = _parse_macos_profiler(_run_json(["system_profiler", "SPDisplaysDataType", "-json"]))
    if parsed is not None:
        return parsed[0] or "GPU (Metal)", parsed[1], "metal"
    return "GPU (Metal)", None, "metal"


def _linux_gpu() -> tuple[str | None, float | None, str]:
    name = _parse_lspci(_run(["lspci"]))
    vram = _amd_sysfs_vram()
    low = (name or "").lower()

    if "amd" in low or "radeon" in low or "advanced micro devices" in low:
        accelerator = "rocm" if _has_cmd("rocm-smi") else "cpu"
        return name, vram, accelerator
    if "intel" in low:
        # Integrated Intel GPUs share system RAM → treat as CPU for the budget.
        return name, None, "cpu"
    if name:
        return name, vram, "cpu"
    return None, None, "cpu"


def _windows_gpu() -> tuple[str | None, float | None, str]:  # pragma: no cover - platform-specific
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
    )
    names = [line.strip() for line in (out or "").splitlines() if line.strip()]
    name = _pick_windows_gpu(names)
    vram = _windows_vram()
    return name, vram, _windows_accelerator(name)


# ----------------------------------------------------------- pure parsers
def _parse_nvidia_smi(output: str | None) -> tuple[str, float | None] | None:
    if not output or not output.strip():
        return None
    line = output.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if not parts or not parts[0]:
        return None
    vram_gb: float | None = None
    if len(parts) >= 2:
        try:
            vram_gb = round(int(float(parts[1])) / 1024, 1)  # MiB → GB
        except (TypeError, ValueError):
            vram_gb = None
    return parts[0], vram_gb


def _parse_macos_profiler(data: Any) -> tuple[str | None, float | None] | None:
    if not isinstance(data, dict):
        return None
    displays = data.get("SPDisplaysDataType")
    if not isinstance(displays, list) or not displays:
        return None
    gpu = displays[0] if isinstance(displays[0], dict) else {}
    name = gpu.get("sppci_model") or gpu.get("_name")
    vram = _vram_from_text(gpu.get("spdisplays_vram") or gpu.get("spdisplays_vram_shared"))
    return name, vram


def _parse_lspci(output: str | None) -> str | None:
    if not output:
        return None
    gpus: list[str] = []
    for line in output.splitlines():
        low = line.lower()
        if (
            "vga compatible controller" in low
            or "3d controller" in low
            or "display controller" in low
        ):
            name = line.split(": ", 1)[-1].strip()
            if name:
                gpus.append(name)
    if not gpus:
        return None
    # Prefer a discrete GPU over an integrated one (order isn't guaranteed).
    for keyword in ("nvidia", "radeon", "amd", "arc"):
        for name in gpus:
            if keyword in name.lower():
                return name
    return gpus[0]


def _pick_windows_gpu(names: list[str]) -> str | None:
    if not names:
        return None
    # Prefer a discrete GPU over an integrated one.
    for keyword in ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd", "arc"):
        for name in names:
            if keyword in name.lower():
                return name
    return names[0]


def _windows_accelerator(name: str | None) -> str:
    low = (name or "").lower()
    if any(k in low for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
        return "cuda"
    if any(k in low for k in ("radeon", "amd", "advanced micro devices")):
        return "rocm"
    return "cpu"


def _vram_from_text(text: Any) -> float | None:
    if not text:
        return None
    match = re.search(r"([\d.]+)\s*(gib|gb|go|mib|mb|mo)", str(text).lower())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit in ("mib", "mb", "mo"):
        return round(value / 1024, 1)
    return round(value, 1)


def _amd_sysfs_vram() -> float | None:
    import glob

    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        try:
            return round(int(Path(path).read_text(encoding="utf-8").strip()) / GB, 1)
        except Exception:
            continue
    return None


def _windows_vram() -> float | None:  # pragma: no cover - platform-specific
    try:
        import winreg

        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as key:
            value, _ = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")
            return round(int(value) / GB, 1) if value else None
    except Exception:
        return None


# --------------------------------------------------------------- helpers
def _budget_gb(accelerator: str, ram_gb: float, vram_gb: float | None) -> float:
    # Dedicated GPU (NVIDIA/AMD): a model runs fast only if it fits in VRAM.
    if accelerator in ("cuda", "rocm") and vram_gb:
        return float(vram_gb)
    # Apple unified memory or CPU/iGPU: the model lives in system RAM. Leave ~30%
    # headroom for the OS, the app and the KV cache.
    return ram_gb * 0.7 if ram_gb else 0.0


def _run(cmd: list[str], timeout: int = 4) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except Exception:
        return None


def _run_json(cmd: list[str], timeout: int = 6) -> Any | None:
    raw = _run(cmd, timeout=timeout)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None
