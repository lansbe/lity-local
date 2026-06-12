import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.core.hardware import (
    _budget_gb,
    _parse_lspci,
    _parse_macos_profiler,
    _parse_nvidia_smi,
    _pick_windows_gpu,
    _vram_from_text,
    _windows_accelerator,
    detect_hardware,
)


class NvidiaParseTests(unittest.TestCase):
    def test_name_and_vram(self):
        self.assertEqual(
            _parse_nvidia_smi("NVIDIA GeForce RTX 4070, 12282\n"),
            ("NVIDIA GeForce RTX 4070", 12.0),
        )

    def test_name_without_vram(self):
        self.assertEqual(_parse_nvidia_smi("NVIDIA RTX A6000\n"), ("NVIDIA RTX A6000", None))

    def test_empty(self):
        self.assertIsNone(_parse_nvidia_smi(""))
        self.assertIsNone(_parse_nvidia_smi(None))


class MacProfilerParseTests(unittest.TestCase):
    def test_intel_mac_discrete_vram(self):
        data = {
            "SPDisplaysDataType": [
                {"sppci_model": "AMD Radeon Pro 5500M", "spdisplays_vram": "8 GB"}
            ]
        }
        self.assertEqual(_parse_macos_profiler(data), ("AMD Radeon Pro 5500M", 8.0))

    def test_apple_silicon_no_vram(self):
        data = {"SPDisplaysDataType": [{"sppci_model": "Apple M2 Pro"}]}
        self.assertEqual(_parse_macos_profiler(data), ("Apple M2 Pro", None))

    def test_garbage(self):
        self.assertIsNone(_parse_macos_profiler(None))
        self.assertIsNone(_parse_macos_profiler({"SPDisplaysDataType": []}))


class LspciParseTests(unittest.TestCase):
    def test_prefers_discrete_over_integrated(self):
        output = (
            "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 630\n"
            "01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31\n"
        )
        self.assertIn("AMD", _parse_lspci(output))

    def test_single_intel(self):
        output = "00:02.0 VGA compatible controller: Intel Corporation Iris Xe Graphics\n"
        self.assertIn("Intel", _parse_lspci(output))

    def test_no_gpu(self):
        self.assertIsNone(_parse_lspci("00:1f.0 ISA bridge: Intel Corporation\n"))


class VramTextTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(_vram_from_text("8 GB"), 8.0)
        self.assertEqual(_vram_from_text("8192 MB"), 8.0)
        self.assertEqual(_vram_from_text("1536 MiB"), 1.5)
        self.assertIsNone(_vram_from_text("shared"))
        self.assertIsNone(_vram_from_text(None))


class WindowsHelperTests(unittest.TestCase):
    def test_pick_prefers_discrete(self):
        self.assertEqual(
            _pick_windows_gpu(["Intel(R) UHD Graphics 770", "NVIDIA GeForce RTX 4060"]),
            "NVIDIA GeForce RTX 4060",
        )
        self.assertEqual(_pick_windows_gpu(["Intel(R) UHD Graphics"]), "Intel(R) UHD Graphics")
        self.assertIsNone(_pick_windows_gpu([]))

    def test_accelerator_from_name(self):
        self.assertEqual(_windows_accelerator("NVIDIA GeForce RTX 4060"), "cuda")
        self.assertEqual(_windows_accelerator("AMD Radeon RX 7800 XT"), "rocm")
        self.assertEqual(_windows_accelerator("Intel Arc A770"), "cpu")
        self.assertEqual(_windows_accelerator(None), "cpu")


class BudgetTests(unittest.TestCase):
    def test_dedicated_gpu_uses_vram(self):
        self.assertEqual(_budget_gb("cuda", 64, 12.0), 12.0)
        self.assertEqual(_budget_gb("rocm", 32, 16.0), 16.0)

    def test_unified_and_cpu_use_ram_fraction(self):
        self.assertAlmostEqual(_budget_gb("metal", 16, None), 11.2, places=1)
        self.assertAlmostEqual(_budget_gb("cpu", 8, None), 5.6, places=1)
        self.assertEqual(_budget_gb("cpu", 0, None), 0.0)


class DetectHardwareSmokeTests(unittest.TestCase):
    def test_returns_expected_shape_without_crashing(self):
        hardware = detect_hardware()
        for key in (
            "os",
            "arch",
            "cpu_cores",
            "ram_gb",
            "gpu",
            "vram_gb",
            "accelerator",
            "budget_gb",
        ):
            self.assertIn(key, hardware)
        self.assertIn(hardware["accelerator"], ("metal", "cuda", "rocm", "cpu"))
        self.assertGreaterEqual(hardware["budget_gb"], 0)


if __name__ == "__main__":
    unittest.main()
