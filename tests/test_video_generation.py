import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lity.infrastructure.paths import AppPaths
from lity.infrastructure.settings import SettingsStore
from lity.services.video_generation.checkpoints import (
    installed_video_models,
    resolve_checkpoint,
)
from lity.services.video_generation.local_engine import LocalVideoEngine
from lity.services.video_generation.manager import VideoGenerationManager
from lity.services.video_generation.model_download import resolve_video_download
from lity.services.video_generation.prompt_builder import _validated_proposal
from lity.services.video_generation.update_interpreter import (
    VideoParamUpdateInterpreter,
)


def _make_diffusers_model(video_models_dir: Path, name: str) -> Path:
    """Drop a minimal diffusers-style model folder (model_index.json marks it)."""
    model_dir = video_models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "scheduler": ["diffusers", "UniPCMultistepScheduler"],
                "text_encoder": ["transformers", "UMT5EncoderModel"],
                "tokenizer": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
            }
        )
    )
    (model_dir / "scheduler").mkdir()
    (model_dir / "scheduler" / "scheduler_config.json").write_text("{}")
    (model_dir / "text_encoder").mkdir()
    (model_dir / "text_encoder" / "config.json").write_text("{}")
    (model_dir / "text_encoder" / "model.safetensors").write_bytes(b"\x00")
    (model_dir / "tokenizer").mkdir()
    (model_dir / "tokenizer" / "tokenizer.json").write_text("{}")
    (model_dir / "transformer").mkdir()
    (model_dir / "transformer" / "config.json").write_text("{}")
    (model_dir / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00")
    (model_dir / "vae").mkdir()
    (model_dir / "vae" / "config.json").write_text("{}")
    (model_dir / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00")
    return model_dir


class VideoGenerationTests(unittest.TestCase):
    # ---------------------------------------------------------- prompt builder
    def test_prompt_proposal_bounds_untrusted_numbers(self):
        proposal = _validated_proposal(
            {
                "prompt": "a cat",
                "num_frames": 9999,
                "fps": 999,
                "steps": -4,
                "cfg_scale": 999.0,
                "width": 99999,
                "height": -10,
            },
            "un chat",
        )
        self.assertLessEqual(proposal["num_frames"], 121)
        self.assertLessEqual(proposal["fps"], 30)
        self.assertGreaterEqual(proposal["steps"], 1)
        self.assertLessEqual(proposal["cfg_scale"], 20.0)
        self.assertLessEqual(proposal["width"], 1280)
        self.assertGreaterEqual(proposal["height"], 256)

    # ----------------------------------------------------------- checkpoints
    def test_installed_models_detect_diffusers_repo_and_resolve_returns_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            _make_diffusers_model(paths.video_models_dir, "wan21-t2v-1.3b")
            found = installed_video_models(paths.video_models_dir)
            self.assertEqual([name for name, _ in found], ["wan21-t2v-1.3b"])
            resolved = resolve_checkpoint(paths.video_models_dir)
            self.assertIsNotNone(resolved)
            self.assertTrue(resolved.is_dir())
            self.assertEqual(resolved.name, "wan21-t2v-1.3b")

    def test_installed_models_reject_incomplete_diffusers_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            model_dir = paths.video_models_dir / "wan21-t2v-1.3b"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "WanPipeline",
                        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
                        "text_encoder": ["transformers", "UMT5EncoderModel"],
                        "tokenizer": ["transformers", "T5TokenizerFast"],
                        "transformer": ["diffusers", "WanTransformer3DModel"],
                        "vae": ["diffusers", "AutoencoderKLWan"],
                    }
                )
            )
            (model_dir / "scheduler").mkdir()
            (model_dir / "scheduler" / "scheduler_config.json").write_text("{}")
            (model_dir / "text_encoder").mkdir()
            (model_dir / "text_encoder" / "config.json").write_text("{}")
            (model_dir / "text_encoder" / "model-00001-of-00005.safetensors").write_bytes(b"\x00")

            self.assertEqual(installed_video_models(paths.video_models_dir), [])

    def test_download_manifest_must_have_every_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            model_dir = paths.video_models_dir / "wan21-t2v-1.3b"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text("{}")
            (model_dir / "one.safetensors").write_bytes(b"\x00")
            (model_dir / ".lity-video-model.json").write_text(
                json.dumps(
                    {"files": ["model_index.json", "one.safetensors", "missing/config.json"]}
                )
            )

            self.assertEqual(installed_video_models(paths.video_models_dir), [])

            (model_dir / "missing").mkdir()
            (model_dir / "missing" / "config.json").write_text("{}")
            self.assertEqual(
                [name for name, _ in installed_video_models(paths.video_models_dir)],
                ["wan21-t2v-1.3b"],
            )

    def test_installed_models_detect_weight_only_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            mlx_dir = paths.video_models_dir / "ltx2-int4-mlx"
            mlx_dir.mkdir(parents=True, exist_ok=True)
            (mlx_dir / "weights.safetensors").write_bytes(b"\x00")
            self.assertEqual(
                [name for name, _ in installed_video_models(paths.video_models_dir)],
                ["ltx2-int4-mlx"],
            )

    # ------------------------------------------------------------- download
    def test_resolve_video_download_keeps_subfolders(self):
        siblings = {
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "vae/diffusion_pytorch_model.safetensors"},
                {"rfilename": "transformer/config.json"},
                {"rfilename": "README.md"},
                {"rfilename": "preview.png"},
            ]
        }

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return siblings

        with patch("httpx.get", return_value=FakeResponse()):
            targets = resolve_video_download(
                {"model_url": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}
            )
        filenames = {t["filename"] for t in targets}
        self.assertIn("vae/diffusion_pytorch_model.safetensors", filenames)
        self.assertIn("transformer/config.json", filenames)
        self.assertNotIn("README.md", filenames)  # docs skipped
        self.assertNotIn("preview.png", filenames)  # images skipped

    def test_resolve_video_download_returns_empty_for_non_hf(self):
        self.assertEqual(resolve_video_download({"model_url": "https://example.com/x"}), [])

    def test_ltx_mlx_catalog_model_resolves_from_hugging_face(self):
        from lity.core.video_model_advisor import VIDEO_MODEL_CATALOG

        model = next(item for item in VIDEO_MODEL_CATALOG if item["name"] == "ltx2-int4-mlx")
        siblings = {
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "split_model.json"},
                {"rfilename": "transformer_blocks/block_000.safetensors"},
                {"rfilename": "README.md"},
            ]
        }

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return siblings

        with patch("httpx.get", return_value=FakeResponse()):
            targets = resolve_video_download(model)

        self.assertEqual(model["model_url"], "https://huggingface.co/dgrauet/ltx-2.3-mlx-q4")
        self.assertIn(
            "transformer_blocks/block_000.safetensors",
            {target["filename"] for target in targets},
        )

    def test_local_video_save_passes_explicit_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            engine = LocalVideoEngine(paths)
            captured: dict[str, object] = {}

            def fake_export(frames, *, output_video_path: str, fps: int):
                captured["frames"] = frames
                captured["fps"] = fps
                captured["path"] = output_video_path
                Path(output_video_path).write_bytes(b"mp4")
                return output_video_path

            saved = engine._save([object()], fps=12, seed=7, export_to_video=fake_export)

            self.assertEqual(captured["fps"], 12)
            self.assertEqual(saved, Path(captured["path"]))
            self.assertTrue(saved.name.endswith("_7.mp4"))
            self.assertTrue(saved.is_file())

    # ----------------------------------------------------- update interpreter
    def test_update_interpreter_confirms_common_commands_without_ollama(self):
        interp = VideoParamUpdateInterpreter(object())
        self.assertEqual(interp.interpret_correction({}, "ok")["action"], "confirm_generation")
        self.assertEqual(
            interp.interpret_correction({}, "Génère !")["action"], "confirm_generation"
        )

    def test_update_interpreter_cancels_common_commands_without_ollama(self):
        interp = VideoParamUpdateInterpreter(object())
        self.assertEqual(interp.interpret_correction({}, "annule")["action"], "cancel_generation")
        self.assertEqual(interp.interpret_correction({}, "stop")["action"], "cancel_generation")

    # --------------------------------------------------------------- session
    def test_first_prompt_generates_directly_with_model_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            _make_diffusers_model(paths.video_models_dir, "wan21-t2v-1.3b")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            captured: dict = {}

            def fake_generate(checkpoint, params, seed, backend="diffusers", config_hint=""):
                captured["params"] = dict(params)
                captured["backend"] = backend
                out = paths.output_videos_dir / "clip.mp4"
                out.write_bytes(b"mp4")
                return out

            manager.engine.generate = fake_generate
            manager.state = {"active": True, "step": "waiting_for_prompt", "current_params": None}

            result = manager.process_user_message("un chat sur une plage", object())

            self.assertEqual(result["type"], "video_generation_result")
            self.assertFalse(manager.is_active())  # session ends after a render
            self.assertTrue(result["content"]["video_path"].endswith("clip.mp4"))
            self.assertEqual(captured["params"]["checkpoint"], "wan21-t2v-1.3b")
            self.assertEqual(captured["params"]["num_frames"], 49)
            self.assertEqual(captured["params"]["fps"], 15)
            self.assertEqual(captured["backend"], "diffusers")

    def test_video_session_generates_on_plain_ok(self):
        class FakeInterpreter:
            def interpret_correction(self, current_params, user_input):
                return {"action": "confirm_generation"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            _make_diffusers_model(paths.video_models_dir, "wan21-t2v-1.3b")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.interpreter = FakeInterpreter()

            def fake_generate(checkpoint, params, seed, backend="diffusers", config_hint=""):
                out = paths.output_videos_dir / "clip.mp4"
                out.write_bytes(b"mp4")
                return out

            manager.engine.generate = fake_generate
            manager.state = {
                "active": True,
                "step": "waiting_for_confirmation",
                "current_params": {"prompt": "cat", "seed": -1},
            }
            result = manager.process_user_message("ok", object())
            self.assertEqual(result["type"], "video_generation_result")
            self.assertFalse(manager.is_active())

    def test_failed_video_generation_keeps_active_session_for_retry(self):
        class FakeInterpreter:
            def interpret_correction(self, current_params, user_input):
                return {"action": "confirm_generation"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            _make_diffusers_model(paths.video_models_dir, "wan21-t2v-1.3b")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.interpreter = FakeInterpreter()

            def boom(checkpoint, params, seed, backend="diffusers", config_hint=""):
                raise RuntimeError("render failed")

            manager.engine.generate = boom
            manager.state = {
                "active": True,
                "step": "waiting_for_confirmation",
                "current_params": {"prompt": "cat", "seed": 1},
            }
            result = manager.process_user_message("ok", object())
            self.assertEqual(result["type"], "error")
            self.assertTrue(manager.is_active())  # stays active so the user can retry

    def test_start_session_is_ready_when_engine_and_model_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            _make_diffusers_model(paths.video_models_dir, "wan21-t2v-1.3b")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            with patch(
                "lity.services.video_generation.manager.dependencies_available",
                return_value=True,
            ):
                result = manager.start_session()
            self.assertEqual(result["type"], "video_mode_ready")
            self.assertEqual(result["status"], "ready")
            self.assertTrue(manager.is_active())

    def test_start_session_reports_no_model_when_none_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            with patch(
                "lity.services.video_generation.manager.dependencies_available",
                return_value=True,
            ):
                result = manager.start_session()
            self.assertEqual(result["type"], "video_dependency")
            self.assertEqual(result["status"], "no_model")
            self.assertFalse(manager.is_active())

    def test_start_session_installs_engine_when_dependencies_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))

            def fake_install(on_progress=None, should_cancel=None):
                if on_progress:
                    on_progress(50, "Downloading diffusers…")
                return {"ok": True, "message": "Moteur vidéo installé."}

            with (
                patch(
                    "lity.services.video_generation.manager.dependencies_available",
                    return_value=False,
                ),
                patch(
                    "lity.services.video_generation.manager.install_video_engine",
                    side_effect=fake_install,
                ),
            ):
                result = manager.start_session()
                if manager._install_thread is not None:
                    manager._install_thread.join(timeout=5)

            self.assertEqual(result["type"], "video_dependency")
            self.assertEqual(result["status"], "installing")
            self.assertFalse(manager.is_active())

    def test_poll_reports_install_progress_then_activates(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            _make_diffusers_model(paths.video_models_dir, "wan21-t2v-1.3b")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))

            manager._install = {"running": True, "pct": 40, "message": "…", "ok": None}
            progress = manager.poll_launch_status()
            self.assertEqual(progress["status"], "installing")
            self.assertEqual(progress["progress"], 40)
            self.assertFalse(manager.is_active())

            manager._install = {"running": False, "pct": 100, "message": "ok", "ok": True}
            with patch(
                "lity.services.video_generation.manager.dependencies_available",
                return_value=True,
            ):
                ready = manager.poll_launch_status()
            self.assertEqual(ready["status"], "ready")
            self.assertTrue(manager.is_active())

    def test_poll_reports_install_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager._install = {"running": False, "pct": 0, "message": "boom", "ok": False}
            result = manager.poll_launch_status()
            self.assertEqual(result["status"], "error")
            self.assertIn("boom", result["message"])
            self.assertFalse(manager.is_active())

    def test_mlx_model_starts_when_runtime_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            mlx_dir = paths.video_models_dir / "ltx2-int4-mlx"
            mlx_dir.mkdir(parents=True, exist_ok=True)
            (mlx_dir / "weights.safetensors").write_bytes(b"\x00")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.select_video_model("ltx2-int4-mlx")
            with (
                patch(
                    "lity.services.video_generation.manager.mlx_video_dependencies_available",
                    return_value=True,
                ),
                patch(
                    "lity.services.video_generation.manager.mlx_video_supported_platform",
                    return_value=True,
                ),
            ):
                result = manager.start_session()
            self.assertEqual(result["status"], "ready")
            self.assertTrue(manager.is_active())

    def test_start_session_installs_mlx_runtime_when_active_command_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            mlx_dir = paths.video_models_dir / "ltx2-int4-mlx"
            mlx_dir.mkdir(parents=True, exist_ok=True)
            (mlx_dir / "weights.safetensors").write_bytes(b"\x00")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.select_video_model("ltx2-int4-mlx")

            def fake_install(paths_arg, on_progress=None, should_cancel=None):
                self.assertEqual(paths_arg, paths)
                if on_progress:
                    on_progress(50, "Installing ltx-2-mlx…")
                return {"ok": True, "message": "Runtime MLX vidéo installé."}

            with (
                patch(
                    "lity.services.video_generation.manager.mlx_video_dependencies_available",
                    return_value=False,
                ),
                patch(
                    "lity.services.video_generation.manager.mlx_video_supported_platform",
                    return_value=True,
                ),
                patch(
                    "lity.services.video_generation.manager.install_mlx_video_engine",
                    side_effect=fake_install,
                ),
            ):
                result = manager.start_session()
                if manager._install_thread is not None:
                    manager._install_thread.join(timeout=5)

            self.assertEqual(result["type"], "video_dependency")
            self.assertEqual(result["status"], "installing")
            self.assertFalse(manager.is_active())

    def test_execute_generation_routes_mlx_to_mlx_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            mlx_dir = paths.video_models_dir / "ltx2-int4-mlx"
            mlx_dir.mkdir(parents=True, exist_ok=True)
            (mlx_dir / "weights.safetensors").write_bytes(b"\x00")
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.select_video_model("ltx2-int4-mlx")
            captured: dict[str, object] = {}

            def fail_diffusers(*args, **kwargs):
                raise AssertionError("diffusers engine called for an mlx video model")

            def fake_mlx_generate(model_dir, params, seed, *, mlx):
                captured["model_dir"] = model_dir
                captured["params"] = params
                captured["mlx"] = mlx
                out = paths.output_videos_dir / "clip.mp4"
                out.write_bytes(b"mp4")
                return out

            manager.engine.generate = fail_diffusers
            manager.mlx_engine.generate = fake_mlx_generate

            result = manager.execute_generation({"prompt": "cat", "seed": 7})

            self.assertEqual(result["type"], "video_generation_result")
            self.assertEqual(captured["model_dir"], mlx_dir)
            self.assertEqual(captured["mlx"]["command"], "ltx-2-mlx")
            self.assertTrue(captured["mlx"]["low_ram"])

    def test_select_video_model_persists_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = VideoGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            result = manager.select_video_model("wan21-t2v-1.3b")
            self.assertEqual(result["selected"], "wan21-t2v-1.3b")
            self.assertEqual(manager.settings.selected_video_model, "wan21-t2v-1.3b")


if __name__ == "__main__":
    unittest.main()
