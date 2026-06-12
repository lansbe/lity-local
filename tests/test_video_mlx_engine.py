import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lity.infrastructure.paths import AppPaths
from lity.services.video_generation.mlx_engine import (
    MlxVideoEngine,
    mlx_video_supported_platform,
)

_CLI = "lity.services.video_generation.mlx_engine.resolve_ltx_mlx_cli"


class MlxVideoEngineTests(unittest.TestCase):
    def _engine(self, tmp):
        return MlxVideoEngine(AppPaths.create(home_override=Path(tmp)))

    def test_build_command_for_ltx_int4_low_ram(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            model_dir = Path(tmp) / "ltx2-int4-mlx"
            output = Path(tmp) / "out.mp4"
            with patch(_CLI, return_value="/fake/ltx-2-mlx"):
                argv = engine.build_command(
                    model_dir,
                    {
                        "prompt": "un chien cinematographique",
                        "num_frames": 50,
                        "width": 777,
                        "height": 511,
                    },
                    seed=42,
                    mlx={
                        "command": "ltx-2-mlx",
                        "mode": "distilled",
                        "low_ram": True,
                        "extra_args": ["--quiet"],
                    },
                    output=output,
                )

            self.assertEqual(argv[:2], ["/fake/ltx-2-mlx", "generate"])
            self.assertEqual(argv[argv.index("--prompt") + 1], "un chien cinematographique")
            self.assertEqual(argv[argv.index("--model") + 1], str(model_dir))
            self.assertEqual(argv[argv.index("--output") + 1], str(output))
            self.assertEqual(argv[argv.index("--seed") + 1], "42")
            self.assertEqual(argv[argv.index("--frames") + 1], "49")
            self.assertEqual(argv[argv.index("--width") + 1], "784")
            self.assertEqual(argv[argv.index("--height") + 1], "512")
            self.assertIn("--distilled", argv)
            self.assertIn("--low-ram", argv)
            self.assertIn("--quiet", argv)

    def test_build_command_raises_when_cli_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            with patch(_CLI, return_value=None), self.assertRaises(RuntimeError):
                engine.build_command(
                    Path(tmp), {"prompt": "x"}, seed=1, mlx={}, output=Path(tmp) / "o.mp4"
                )

    def test_generate_runs_cli_and_returns_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)

            class FakeProc:
                returncode = 0
                stdout = ""
                stderr = ""

            def fake_run(argv, **kwargs):
                out = Path(argv[argv.index("--output") + 1])
                out.write_bytes(b"mp4")
                return FakeProc()

            with (
                patch(_CLI, return_value="/fake/ltx-2-mlx"),
                patch(
                    "lity.services.video_generation.mlx_engine.subprocess.run", side_effect=fake_run
                ),
            ):
                path = engine.generate(
                    Path(tmp) / "ltx2-int4-mlx",
                    {"prompt": "chien", "num_frames": 49},
                    seed=7,
                    mlx={"command": "ltx-2-mlx", "mode": "distilled", "low_ram": True},
                )

            self.assertTrue(path.exists())
            self.assertTrue(path.name.endswith(".mp4"))

    def test_generate_surfaces_ltx_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)

            class FakeProc:
                returncode = 1
                stdout = ""
                stderr = "RuntimeError: out of memory"

            with (
                patch(_CLI, return_value="/fake/ltx-2-mlx"),
                patch(
                    "lity.services.video_generation.mlx_engine.subprocess.run",
                    return_value=FakeProc(),
                ),
                self.assertRaises(RuntimeError) as ctx,
            ):
                engine.generate(Path(tmp), {"prompt": "x"}, seed=1, mlx={})
            self.assertIn("out of memory", str(ctx.exception))

    def test_platform_probe_is_boolean(self):
        self.assertIn(mlx_video_supported_platform(), (True, False))
        self.assertGreaterEqual(sys.version_info.major, 3)


if __name__ == "__main__":
    unittest.main()
