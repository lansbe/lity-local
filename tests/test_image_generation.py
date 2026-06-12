import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.infrastructure.paths import AppPaths
from lity.infrastructure.settings import SettingsStore
from lity.services.image_generation.launcher import (
    StableDiffusionLauncher,
    installation_instructions,
)
from lity.services.image_generation.manager import ImageGenerationManager
from lity.services.image_generation.prompt_builder import _validated_proposal
from lity.services.image_generation.stable_diffusion import (
    StableDiffusionService,
    normalize_generation_params,
)
from lity.services.image_generation.update_interpreter import (
    ImageParamUpdateInterpreter,
)


class ImageGenerationTests(unittest.TestCase):
    def test_launcher_support_modules_are_available(self):
        from lity.services.image_generation.launcher_paths import candidate_dirs
        from lity.services.image_generation.launcher_platform import launch_command
        from lity.services.image_generation.launcher_process import process_is_running

        self.assertTrue(callable(candidate_dirs))
        self.assertTrue(callable(launch_command))
        self.assertTrue(callable(process_is_running))

    def test_prompt_proposal_bounds_untrusted_numbers(self):
        proposal = _validated_proposal(
            {"width": 99999, "height": -1, "steps": 999, "cfg_scale": 999},
            "un chat",
        )

        self.assertEqual(proposal["width"], 1536)
        self.assertEqual(proposal["height"], 256)
        self.assertEqual(proposal["steps"], 80)
        self.assertEqual(proposal["cfg_scale"], 20.0)

    def test_generation_payload_uses_safe_defaults(self):
        params = normalize_generation_params({"prompt": "cat", "steps": "bad"}, seed=42)

        self.assertEqual(params["prompt"], "cat")
        self.assertEqual(params["steps"], 25)
        self.assertEqual(params["seed"], 42)

    def test_generate_image_normalizes_invalid_seed_before_api_failure(self):
        class FailingStableDiffusionService(StableDiffusionService):
            def _switch_checkpoint(self, checkpoint):
                return None

            def _post_json(self, endpoint, payload, timeout):
                raise RuntimeError("api down")

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            service = FailingStableDiffusionService(paths)

            result = service.generate_image({"prompt": "cat", "seed": "bad"})

            self.assertEqual(result["status"], "error")
            self.assertIsInstance(result["params"]["seed"], int)

    def test_failed_image_generation_keeps_active_session_for_retry(self):
        class FakeInterpreter:
            def interpret_correction(self, current_params, user_input):
                return {"action": "confirm_generation"}

        class FakeService:
            def generate_image(self, params):
                return {"status": "error", "message": "api down", "params": params}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.interpreter = FakeInterpreter()
            manager.service = FakeService()
            manager.state = {
                "active": True,
                "step": "waiting_for_confirmation",
                "current_params": {"prompt": "cat"},
            }

            result = manager.process_user_message("ok", object())

            self.assertEqual(result["type"], "error")
            self.assertTrue(manager.is_active())
            self.assertEqual(manager.state["step"], "waiting_for_confirmation")
            self.assertEqual(manager.state["current_params"], {"prompt": "cat"})

    def test_update_interpreter_confirms_common_commands_without_ollama(self):
        interpreter = ImageParamUpdateInterpreter(object())

        for message in ["ok", "OK", "oui", "génère", "genere", "c'est bon"]:
            with self.subTest(message=message):
                result = interpreter.interpret_correction({"prompt": "cat"}, message)

                self.assertEqual(result["action"], "confirm_generation")

    def test_update_interpreter_cancels_common_commands_without_ollama(self):
        interpreter = ImageParamUpdateInterpreter(object())

        for message in ["annule", "Annuler", "cancel", "arrête"]:
            with self.subTest(message=message):
                result = interpreter.interpret_correction({"prompt": "cat"}, message)

                self.assertEqual(result["action"], "cancel_generation")

    def test_image_session_generates_on_plain_ok(self):
        class FakeInterpreter:
            def interpret_correction(self, current_params, user_input):
                return {"action": "confirm_generation"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            model_dir = paths.image_models_dir / "sd15"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.safetensors").write_bytes(b"\x00" * 4096)

            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.interpreter = FakeInterpreter()
            calls: dict = {}

            def fake_generate(checkpoint, params, seed, **kwargs):
                calls["checkpoint"] = checkpoint
                calls["seed"] = seed
                out = paths.output_images_dir / "cat.png"
                out.write_bytes(b"img")
                return out

            manager.engine.generate = fake_generate
            manager.state = {
                "active": True,
                "step": "waiting_for_confirmation",
                "current_params": {"prompt": "cat", "seed": -1},
            }

            result = manager.process_user_message("ok", object())

            self.assertEqual(result["type"], "image_generation_result")
            self.assertEqual(calls["checkpoint"].name, "model.safetensors")
            self.assertTrue(result["content"]["image_path"].endswith("cat.png"))
            self.assertFalse(manager.is_active())

    def test_first_prompt_generates_directly_with_distilled_defaults(self):
        """Describing an image renders it straight away — no 'ok' round-trip —
        and a turbo/lightning checkpoint must override the LLM's classic
        steps/cfg defaults (25 steps at cfg 7.5 burns distilled models)."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            model_dir = paths.image_models_dir / "sd-turbo"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "sd_turbo.safetensors").write_bytes(b"\x00" * 4096)

            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            captured: dict = {}

            def fake_generate(checkpoint, params, seed, **kwargs):
                captured["params"] = dict(params)
                out = paths.output_images_dir / "cat.png"
                out.write_bytes(b"img")
                return out

            manager.engine.generate = fake_generate
            manager.state = {
                "active": True,
                "step": "waiting_for_prompt",
                "current_params": None,
            }

            # No Ollama in tests: the prompt builder falls back to its
            # classic defaults (steps 25 / cfg 7.5), which the manager must
            # then override with the model's own generation profile before
            # rendering directly.
            result = manager.process_user_message("un chat sur un toit", object())

            self.assertEqual(result["type"], "image_generation_result")
            self.assertFalse(manager.is_active())  # session ends after a render
            self.assertEqual(captured["params"]["checkpoint"], "sd-turbo")
            self.assertEqual(captured["params"]["steps"], 4)
            self.assertEqual(captured["params"]["cfg_scale"], 0.0)
            self.assertEqual(captured["params"]["sampler"], "Euler a")

    def test_start_session_is_ready_when_engine_and_model_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            model_dir = paths.image_models_dir / "sd15"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.safetensors").write_bytes(b"\x00" * 4096)
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))

            with patch(
                "lity.services.image_generation.manager.dependencies_available",
                return_value=True,
            ):
                result = manager.start_session()

            self.assertEqual(result["type"], "image_mode_ready")
            self.assertEqual(result["status"], "ready")
            self.assertTrue(manager.is_active())

    def test_start_session_reports_no_model_when_none_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))

            with patch(
                "lity.services.image_generation.manager.dependencies_available",
                return_value=True,
            ):
                result = manager.start_session()

            self.assertEqual(result["type"], "image_dependency")
            self.assertEqual(result["status"], "no_model")
            self.assertFalse(manager.is_active())

    def test_start_session_installs_engine_when_dependencies_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))

            def fake_install(on_progress=None, should_cancel=None):
                if on_progress:
                    on_progress(50, "Downloading torch…")
                return {"ok": True, "message": "Moteur image installé."}

            with (
                patch(
                    "lity.services.image_generation.manager.dependencies_available",
                    return_value=False,
                ),
                patch(
                    "lity.services.image_generation.manager.install_engine",
                    side_effect=fake_install,
                ),
            ):
                result = manager.start_session()
                if manager._install_thread is not None:
                    manager._install_thread.join(timeout=5)

            self.assertEqual(result["type"], "image_dependency")
            self.assertEqual(result["status"], "installing")
            self.assertFalse(manager.is_active())

    def test_poll_reports_install_progress_then_activates(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            model_dir = paths.image_models_dir / "sd15"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.safetensors").write_bytes(b"\x00" * 4096)
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))

            # Install still running → progress surfaced, not yet active.
            manager._install = {"running": True, "pct": 40, "message": "…", "ok": None}
            progress = manager.poll_launch_status()
            self.assertEqual(progress["status"], "installing")
            self.assertEqual(progress["progress"], 40)
            self.assertFalse(manager.is_active())

            # Install finished OK + a model present → activate.
            manager._install = {"running": False, "pct": 100, "message": "ok", "ok": True}
            with patch(
                "lity.services.image_generation.manager.dependencies_available",
                return_value=True,
            ):
                ready = manager.poll_launch_status()
            self.assertEqual(ready["status"], "ready")
            self.assertTrue(manager.is_active())

    def test_poll_reports_install_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager._install = {"running": False, "pct": 0, "message": "boom", "ok": False}

            result = manager.poll_launch_status()

            self.assertEqual(result["status"], "error")
            self.assertIn("boom", result["message"])
            self.assertFalse(manager.is_active())

    def test_launcher_starts_configured_webui_with_api_flag(self):
        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui"
            install_dir.mkdir()
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(manager.settings, popen=fake_popen)

            result = launcher.launch()

            self.assertTrue(result.launched)
            self.assertEqual(calls[0][0], ["bash", str(install_dir / "webui.sh"), "--api"])
            self.assertEqual(calls[0][1]["cwd"], str(install_dir))

    def test_launcher_does_not_auto_launch_unconfigured_automatic1111_installation(self):
        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unconfigured_install = root / "home" / "Documents" / "stable-diffusion-webui"
            unconfigured_install.mkdir(parents=True)
            (unconfigured_install / "webui.sh").write_text(
                "#!/usr/bin/env bash\n", encoding="utf-8"
            )
            paths = AppPaths.create(home_override=root / "app")
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            launcher = StableDiffusionLauncher(manager.settings, popen=fake_popen)

            with patch.object(Path, "home", return_value=root / "home"):
                result = launcher.launch()

            self.assertEqual(result.status, "missing")
            self.assertEqual(calls, [])
            self.assertIn("WebUI Forge", result.message)
            self.assertIn("~/Documents/stable-diffusion-webui-forge", result.install_command)

    def test_launcher_prepares_existing_webui_venv_for_clip_builds(self):
        popen_calls = []
        run_calls = []

        class Completed:
            def __init__(self, returncode=0, stdout=""):
                self.returncode = returncode
                self.stdout = stdout

        def fake_popen(command, **kwargs):
            popen_calls.append((command, kwargs))

        def fake_run(command, **kwargs):
            run_calls.append((command, kwargs))
            if command[1] == "-c" and "sys.version_info" in command[2]:
                return Completed(returncode=0, stdout="3.10\n")
            if command[1] == "-c" and "pkg_resources" in command[2]:
                return Completed(returncode=1, stdout="No module named 'wheel'")
            if command[1] == "-c" and "import clip" in command[2]:
                return Completed(returncode=1, stdout="No module named 'clip'")
            return Completed(returncode=0, stdout="installed wheel")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui-forge"
            venv_bin = install_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(
                manager.settings,
                popen=fake_popen,
                run=fake_run,
            )

            result = launcher.launch()

            self.assertTrue(result.launched)
            self.assertIn("sys.version_info", run_calls[0][0][2])
            self.assertEqual(run_calls[1][0][1:], ["-c", "import pkg_resources, wheel"])
            self.assertEqual(
                run_calls[2][0][1:],
                ["-m", "pip", "install", "setuptools<81", "wheel"],
            )
            self.assertEqual(run_calls[3][0][1:], ["-c", "import clip"])
            self.assertEqual(
                run_calls[4][0][1:3],
                ["-m", "pip"],
            )
            self.assertIn("--no-build-isolation", run_calls[4][0])
            self.assertIn("github.com/openai/CLIP", run_calls[4][0][5])
            self.assertEqual(popen_calls[0][1]["env"]["PIP_NO_BUILD_ISOLATION"], "1")

    def test_launcher_recreates_incompatible_python_venv_when_python310_exists(self):
        popen_calls = []
        run_calls = []

        class Completed:
            def __init__(self, returncode=0, stdout=""):
                self.returncode = returncode
                self.stdout = stdout

        def fake_popen(command, **kwargs):
            popen_calls.append((command, kwargs))

        def fake_run(command, **kwargs):
            run_calls.append((command, kwargs))
            return Completed(returncode=0, stdout="3.9\n")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui-forge"
            venv_bin = install_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(
                manager.settings,
                popen=fake_popen,
                run=fake_run,
            )

            with patch(
                "lity.services.image_generation.launcher.shutil.which",
                return_value="/opt/homebrew/bin/python3.10",
            ):
                result = launcher.launch()

            self.assertTrue(result.launched)
            self.assertFalse((install_dir / "venv").exists())
            self.assertEqual(popen_calls[0][1]["env"]["python_cmd"], "python3.10")

    def test_launcher_reports_python310_install_command_for_incompatible_venv(self):
        popen_calls = []

        class Completed:
            returncode = 0
            stdout = "3.9\n"

        def fake_popen(command, **kwargs):
            popen_calls.append((command, kwargs))

        def fake_run(command, **kwargs):
            return Completed()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui-forge"
            venv_bin = install_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(
                manager.settings,
                popen=fake_popen,
                run=fake_run,
            )

            with patch("lity.services.image_generation.launcher.shutil.which", return_value=None):
                result = launcher.launch()

            self.assertEqual(result.status, "error")
            self.assertEqual(popen_calls, [])
            self.assertIn("Python 3.10", result.message)
            self.assertIn("brew install python@3.10", result.message)

    def test_launcher_repairs_numpy_skimage_binary_mismatch_before_launch(self):
        popen_calls = []
        run_calls = []

        class Completed:
            def __init__(self, returncode=0, stdout=""):
                self.returncode = returncode
                self.stdout = stdout

        def fake_popen(command, **kwargs):
            popen_calls.append((command, kwargs))

        def fake_run(command, **kwargs):
            run_calls.append((command, kwargs))
            if command[1] == "-c" and "sys.version_info" in command[2]:
                return Completed(returncode=0, stdout="3.10\n")
            if command[1] == "-c" and "import pkg_resources" in command[2]:
                return Completed(returncode=0, stdout="")
            if command[1] == "-c" and "import clip" in command[2]:
                return Completed(returncode=0, stdout="")
            if command[1] == "-c" and "skimage" in command[2]:
                return Completed(returncode=1, stdout="numpy.dtype size changed")
            return Completed(returncode=0, stdout="installed runtime pins")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui-forge"
            venv_bin = install_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(
                manager.settings,
                popen=fake_popen,
                run=fake_run,
            )

            result = launcher.launch()

            commands = [call[0] for call in run_calls]
            self.assertTrue(result.launched)
            self.assertEqual(len(popen_calls), 1)
            self.assertTrue(
                any(
                    command[1:4] == ["-m", "pip", "install"]
                    and "numpy==1.26.2" in command
                    and "scikit-image==0.21.0" in command
                    for command in commands
                )
            )

    def test_launcher_does_not_inherit_lity_python_virtualenv(self):
        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui"
            install_dir.mkdir()
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(manager.settings, popen=fake_popen)
            lity_venv = root / "app" / ".venv"
            lity_venv_bin = lity_venv / "bin"

            with patch.dict(
                os.environ,
                {
                    "VIRTUAL_ENV": str(lity_venv),
                    "PYTHONHOME": "/bad/pythonhome",
                    "PYTHONPATH": "/bad/pythonpath",
                    "PATH": os.pathsep.join([str(lity_venv_bin), "/usr/local/bin", "/usr/bin"]),
                },
            ):
                result = launcher.launch()

            launch_env = calls[0][1]["env"]
            path_entries = launch_env["PATH"].split(os.pathsep)
            self.assertTrue(result.launched)
            self.assertNotIn("VIRTUAL_ENV", launch_env)
            self.assertNotIn("PYTHONHOME", launch_env)
            self.assertNotIn("PYTHONPATH", launch_env)
            self.assertNotIn(str(lity_venv_bin), path_entries)
            self.assertIn("/usr/local/bin", path_entries)
            self.assertIn("/usr/bin", path_entries)

    def test_launcher_disables_interactive_git_and_vscode_askpass(self):
        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui"
            install_dir.mkdir()
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(manager.settings, popen=fake_popen)

            with patch.dict(
                os.environ,
                {
                    "GIT_ASKPASS": "/Applications/Visual Studio Code.app/git-askpass.sh",
                    "SSH_ASKPASS": "/Applications/Visual Studio Code.app/git-askpass.sh",
                    "VSCODE_GIT_ASKPASS_NODE": "/Applications/Visual Studio Code.app/node",
                    "VSCODE_GIT_ASKPASS_MAIN": "/Applications/Visual Studio Code.app/askpass-main.js",
                    "VSCODE_GIT_ASKPASS_EXTRA_ARGS": "",
                    "VSCODE_GIT_IPC_HANDLE": "/tmp/vscode-git.sock",
                    "GCM_INTERACTIVE": "always",
                },
            ):
                result = launcher.launch()

            launch_env = calls[0][1]["env"]
            self.assertTrue(result.launched)
            self.assertEqual(launch_env["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(launch_env["GCM_INTERACTIVE"], "never")
            self.assertNotIn("GIT_ASKPASS", launch_env)
            self.assertNotIn("SSH_ASKPASS", launch_env)
            self.assertNotIn("VSCODE_GIT_ASKPASS_NODE", launch_env)
            self.assertNotIn("VSCODE_GIT_ASKPASS_MAIN", launch_env)
            self.assertNotIn("VSCODE_GIT_ASKPASS_EXTRA_ARGS", launch_env)
            self.assertNotIn("VSCODE_GIT_IPC_HANDLE", launch_env)

    def test_launcher_does_not_spawn_duplicate_process_while_launching(self):
        class FakeProcess:
            def poll(self):
                return None

        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui"
            install_dir.mkdir()
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(manager.settings, popen=fake_popen)

            first = launcher.launch()
            second = launcher.launch()

            self.assertEqual(first.status, "launched")
            self.assertEqual(second.status, "launching")
            self.assertEqual(len(calls), 1)

    def test_launcher_shutdown_terminates_owned_process(self):
        class FakeProcess:
            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return None if not self.terminated else 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.killed = True

        process = FakeProcess()

        def fake_popen(command, **kwargs):
            return process

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui"
            install_dir.mkdir()
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(manager.settings, popen=fake_popen)

            launcher.launch()
            result = launcher.shutdown(timeout=0.1)

            self.assertEqual(result.status, "stopped")
            self.assertTrue(process.terminated)
            self.assertFalse(process.killed)

    def test_launcher_writes_pid_file_for_launched_process(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui"
            install_dir.mkdir()
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            log_file = root / "stable-diffusion.log"
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(
                manager.settings,
                popen=lambda command, **kwargs: FakeProcess(),
                log_file=log_file,
            )

            result = launcher.launch()

            self.assertEqual(result.status, "launched")
            self.assertEqual(log_file.with_suffix(".pid").read_text(encoding="utf-8"), "12345")
            launcher._close_log_stream()

    def test_launcher_shutdown_terminates_pid_file_process(self):
        process = subprocess.Popen(["sleep", "30"], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_file = Path(tmp) / "stable-diffusion.log"
                launcher = StableDiffusionLauncher(object(), log_file=log_file)
                log_file.with_suffix(".pid").write_text(str(process.pid), encoding="utf-8")

                result = launcher.shutdown(timeout=1.0)

                self.assertEqual(result.status, "stopped")
                self.assertFalse(log_file.with_suffix(".pid").exists())
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            with suppress(ChildProcessError):
                process.wait(timeout=1.0)

    def test_launcher_status_reports_exit_code_and_log_tail_when_process_dies(self):
        class FakeProcess:
            returncode = 1

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "stable-diffusion.log"
            log_file.write_text(
                "line 1\n"
                "Couldn't clone Stable Diffusion\n"
                "https://github.com/Stability-AI/stablediffusion.git\n",
                encoding="utf-8",
            )
            launcher = StableDiffusionLauncher(object(), log_file=log_file)
            launcher._process = FakeProcess()

            result = launcher.status()

            self.assertEqual(result.status, "error")
            self.assertIn("code 1", result.message)
            self.assertIn("Stability-AI/stablediffusion.git", result.message)
            self.assertIn("stable-diffusion-webui-forge", result.message)

    def test_launcher_does_not_retry_automatic1111_after_known_upstream_clone_failure(self):
        class FakeProcess:
            returncode = 1

            def poll(self):
                return self.returncode

        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui"
            install_dir.mkdir()
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            log_file = root / "stable-diffusion.log"
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(
                manager.settings,
                popen=fake_popen,
                log_file=log_file,
            )

            with patch.object(Path, "home", return_value=root / "home"):
                first = launcher.launch()
                launcher._log_stream.write(
                    "RuntimeError: Couldn't clone Stable Diffusion.\n"
                    "https://github.com/Stability-AI/stablediffusion.git\n",
                )
                launcher._log_stream.flush()
                failure = launcher.status()
                second = launcher.launch()

            self.assertEqual(first.status, "launched")
            self.assertEqual(failure.status, "error")
            self.assertEqual(second.status, "error")
            self.assertEqual(len(calls), 1)
            self.assertIn("Automatic1111", second.message)
            self.assertIn("stable-diffusion-webui-forge", second.message)

    def test_launcher_repairs_clip_pkg_resources_failure_and_relaunches_once(self):
        class FakeProcess:
            returncode = 0

            def poll(self):
                return self.returncode

        class Completed:
            returncode = 0
            stdout = "installed wheel"

        popen_calls = []
        run_calls = []

        def fake_popen(command, **kwargs):
            popen_calls.append((command, kwargs))

        def fake_run(command, **kwargs):
            run_calls.append((command, kwargs))
            if command[1] == "-c" and "sys.version_info" in command[2]:
                return type("Completed", (), {"returncode": 0, "stdout": "3.10\n"})()
            if command[1] == "-c" and "pkg_resources" in command[2]:
                return type("Completed", (), {"returncode": 1, "stdout": "No module named wheel"})()
            return Completed()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "stable-diffusion-webui-forge"
            venv_bin = install_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (install_dir / "webui.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            log_file = root / "stable-diffusion.log"
            log_file.write_text(
                "ModuleNotFoundError: No module named 'pkg_resources'\n"
                "ERROR: Failed to build "
                "'https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip'\n",
                encoding="utf-8",
            )
            paths = AppPaths.create(home_override=root / "app")
            store = SettingsStore(paths.settings_file)
            store.set("sd_install_dir", str(install_dir))
            manager = ImageGenerationManager(object(), paths, store)
            launcher = StableDiffusionLauncher(
                manager.settings,
                popen=fake_popen,
                run=fake_run,
                log_file=log_file,
            )
            launcher._process = FakeProcess()
            launcher._active_install_dir = install_dir

            result = launcher.status()

            self.assertEqual(result.status, "launched")
            self.assertIn("CLIP", result.message)
            self.assertEqual(len(popen_calls), 1)
            commands = [call[0][1:] for call in run_calls]
            self.assertIn(["-m", "pip", "install", "setuptools<81", "wheel"], commands)
            self.assertIn(["-c", "import clip"], commands)
            launcher._close_log_stream()

    def test_launcher_recommends_python_310_when_webui_uses_bad_python_environment(self):
        class FakeProcess:
            returncode = 1

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "stable-diffusion.log"
            log_file.write_text(
                "python venv already activate or run without venv: "
                "/Users/test/lity/.venv\n"
                "INCOMPATIBLE PYTHON VERSION\n"
                "Python 3.14.5\n"
                "No module named pip\n"
                "RuntimeError: Couldn't install torch.\n",
                encoding="utf-8",
            )
            launcher = StableDiffusionLauncher(object(), log_file=log_file)
            launcher._process = FakeProcess()

            result = launcher.status()

            self.assertEqual(result.status, "error")
            self.assertIn("Python 3.10", result.message)
            self.assertIn("brew install", result.message)
            self.assertIn("venv", result.message)

    def test_installation_instructions_are_platform_specific(self):
        mac = installation_instructions("darwin")
        windows = installation_instructions("win32")

        self.assertIn("~/Documents/stable-diffusion-webui-forge", mac.install_command)
        self.assertIn("./webui.sh --api", mac.run_command)
        self.assertIn(
            "$env:USERPROFILE\\Documents\\stable-diffusion-webui-forge",
            windows.install_command,
        )
        self.assertIn(".\\webui-user.bat --api", windows.run_command)


class MlxRoutingTests(unittest.TestCase):
    """The MLX (mflux) backend coexists with diffusers without breaking it."""

    def _make_sd_model(self, paths, name="sd15"):
        folder = paths.image_models_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"\x00" * 4096)
        return folder

    def _make_mlx_model(self, paths, name="z-image-turbo"):
        from lity.services.image_generation.checkpoints import write_model_marker

        folder = paths.image_models_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"\x00" * 4096)
        write_model_marker(
            folder,
            {
                "name": name,
                "display_name": "Z-Image Turbo 6B (MLX 4-bit)",
                "backend": "mlx",
                "mlx": {
                    "command": "mflux-generate-z-image-turbo",
                    "model_arg": "model",
                    "base_model": "z-image-turbo",
                    "quantize": 4,
                },
            },
        )
        return folder

    def test_marker_round_trip(self):
        from lity.services.image_generation.checkpoints import (
            read_model_marker,
            write_model_marker,
        )

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "m"
            folder.mkdir()
            write_model_marker(folder, {"name": "m", "backend": "mlx", "mlx": {"quantize": 4}})
            marker = read_model_marker(folder)
            self.assertEqual(marker["backend"], "mlx")
            self.assertEqual(marker["mlx"]["quantize"], 4)

    def test_records_split_backends_and_diffusers_excludes_mlx(self):
        from lity.services.image_generation.checkpoints import (
            installed_image_models,
            installed_image_records,
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            self._make_sd_model(paths, "sd15")
            self._make_mlx_model(paths, "z-image-turbo")

            records = {r.name: r for r in installed_image_records(paths.image_models_dir)}
            self.assertEqual(records["sd15"].backend, "automatic1111")
            self.assertEqual(records["z-image-turbo"].backend, "mlx")
            # diffusers backend (mlx model is the *folder*, SD is the file)
            self.assertTrue(records["z-image-turbo"].path.is_dir())
            self.assertEqual(records["sd15"].path.name, "model.safetensors")

            # The diffusers view must NEVER include the mlx model — otherwise the
            # diffusers engine would try (and fail) to load mflux weights.
            diffusers_names = [name for name, _ in installed_image_models(paths.image_models_dir)]
            self.assertEqual(diffusers_names, ["sd15"])

    def test_execute_generation_routes_mlx_to_mlx_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            self._make_mlx_model(paths, "z-image-turbo")
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.settings.set_selected_image_model("z-image-turbo")

            captured: dict = {}

            def fake_mlx_generate(model_dir, params, seed, *, mlx):
                captured["model_dir"] = model_dir
                captured["mlx"] = mlx
                out = paths.output_images_dir / "dog.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"img")
                return out

            # Diffusers engine must NOT be touched for an mlx model.
            def boom(*a, **k):
                raise AssertionError("diffusers engine called for an mlx model")

            manager.mlx_engine.generate = fake_mlx_generate
            manager.engine.generate = boom

            result = manager.execute_generation({"prompt": "chien", "checkpoint": "z-image-turbo"})

            self.assertEqual(result["type"], "image_generation_result")
            self.assertEqual(result["content"]["params"]["checkpoint"], "z-image-turbo")
            self.assertTrue(result["content"]["image_path"].endswith("dog.png"))
            self.assertEqual(captured["model_dir"].name, "z-image-turbo")
            self.assertEqual(captured["mlx"]["command"], "mflux-generate-z-image-turbo")
            self.assertEqual(captured["mlx"]["model_arg"], "model")
            self.assertEqual(captured["mlx"]["base_model"], "z-image-turbo")
            self.assertEqual(captured["mlx"]["quantize"], 4)

    def test_required_backend_follows_selected_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            self._make_sd_model(paths, "sd15")
            self._make_mlx_model(paths, "z-image-turbo")
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))

            manager.settings.set_selected_image_model("z-image-turbo")
            self.assertEqual(manager._required_backend(), "mlx")
            manager.settings.set_selected_image_model("sd15")
            self.assertEqual(manager._required_backend(), "automatic1111")

    def test_start_session_installs_mlx_when_active_command_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            self._make_mlx_model(paths, "z-image-turbo")
            manager = ImageGenerationManager(object(), paths, SettingsStore(paths.settings_file))
            manager.settings.set_selected_image_model("z-image-turbo")

            def fake_install(on_progress=None, should_cancel=None):
                return {"ok": True, "message": "Moteur MLX installé."}

            with (
                patch(
                    "lity.services.image_generation.manager.mlx_dependencies_available",
                    return_value=False,
                ) as deps,
                patch(
                    "lity.services.image_generation.manager.mlx_supported_platform",
                    return_value=True,
                ),
                patch(
                    "lity.services.image_generation.manager.install_mlx_engine",
                    side_effect=fake_install,
                ),
            ):
                result = manager.start_session()
                if manager._install_thread is not None:
                    manager._install_thread.join(timeout=5)

            self.assertEqual(result["status"], "installing")
            deps.assert_called_with("mflux-generate-z-image-turbo")


if __name__ == "__main__":
    unittest.main()
