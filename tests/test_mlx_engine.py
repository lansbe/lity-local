import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.infrastructure.paths import AppPaths
from lity.services.image_generation.mlx_engine import (
    MlxImageEngine,
    mlx_supported_platform,
)

_CLI = "lity.services.image_generation.mlx_engine.resolve_mflux_cli"


class MlxEngineTests(unittest.TestCase):
    def _engine(self, tmp):
        return MlxImageEngine(AppPaths.create(home_override=Path(tmp)))

    def test_build_command_for_z_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            model_dir = Path(tmp) / "z-image-turbo"
            out = Path(tmp) / "out.png"
            with patch(_CLI, return_value="/fake/mflux-generate-z-image-turbo"):
                argv = engine.build_command(
                    model_dir,
                    {
                        "prompt": "un chien",
                        "steps": 9,
                        "width": 768,
                        "height": 768,
                        "cfg_scale": 0.0,
                    },
                    seed=42,
                    mlx={
                        "command": "mflux-generate-z-image-turbo",
                        "model_arg": "model",
                        "base_model": "z-image-turbo",
                        "quantize": 4,
                    },
                    output=out,
                )

            self.assertEqual(argv[0], "/fake/mflux-generate-z-image-turbo")
            self.assertEqual(argv[argv.index("--prompt") + 1], "un chien")
            self.assertEqual(argv[argv.index("--seed") + 1], "42")
            self.assertEqual(argv[argv.index("--steps") + 1], "9")
            self.assertEqual(argv[argv.index("--model") + 1], str(model_dir))
            self.assertEqual(argv[argv.index("--base-model") + 1], "z-image-turbo")
            self.assertEqual(argv[argv.index("--output") + 1], str(out))
            self.assertEqual(argv[argv.index("--quantize") + 1], "4")
            # Distilled guidance is omitted when cfg is 0.
            self.assertNotIn("--guidance", argv)

    def test_build_command_passes_model_and_guidance_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            with patch(_CLI, return_value="/fake/mflux-generate"):
                argv = engine.build_command(
                    Path(tmp) / "schnell",
                    {"prompt": "x", "steps": 4, "cfg_scale": 3.5},
                    seed=1,
                    mlx={"command": "mflux-generate", "model": "schnell", "quantize": 4},
                    output=Path(tmp) / "o.png",
                )
            self.assertEqual(argv[argv.index("--model") + 1], "schnell")
            self.assertEqual(argv[argv.index("--path") + 1], str(Path(tmp) / "schnell"))
            self.assertEqual(argv[argv.index("--guidance") + 1], "3.5")

    def test_build_command_raises_when_cli_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            with patch(_CLI, return_value=None), self.assertRaises(RuntimeError):
                engine.build_command(
                    Path(tmp), {"prompt": "x"}, seed=1, mlx={}, output=Path(tmp) / "o.png"
                )

    def test_generate_runs_cli_and_returns_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)

            class FakeProc:
                returncode = 0
                stdout = ""
                stderr = ""

            def fake_run(argv, **kwargs):
                # mflux writes to the --output path; emulate that.
                out = Path(argv[argv.index("--output") + 1])
                out.write_bytes(b"img")
                return FakeProc()

            with (
                patch(_CLI, return_value="/fake/mflux-generate-z-image-turbo"),
                patch(
                    "lity.services.image_generation.mlx_engine.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                path = engine.generate(
                    Path(tmp) / "z-image-turbo",
                    {"prompt": "chien", "steps": 9},
                    seed=7,
                    mlx={"command": "mflux-generate-z-image-turbo", "model_arg": "model"},
                )
            self.assertTrue(path.exists())
            self.assertTrue(path.name.endswith(".png"))

    def test_generate_surfaces_mflux_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)

            class FakeProc:
                returncode = 1
                stdout = ""
                stderr = "RuntimeError: out of memory"

            with (
                patch(_CLI, return_value="/fake/mflux-generate"),
                patch(
                    "lity.services.image_generation.mlx_engine.subprocess.run",
                    return_value=FakeProc(),
                ),
                self.assertRaises(RuntimeError) as ctx,
            ):
                engine.generate(Path(tmp), {"prompt": "x"}, seed=1, mlx={})
            self.assertIn("out of memory", str(ctx.exception))

    def test_platform_probe_is_boolean(self):
        self.assertIn(mlx_supported_platform(), (True, False))


if __name__ == "__main__":
    unittest.main()
