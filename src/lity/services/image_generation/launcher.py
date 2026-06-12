from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lity.services.image_generation.launcher_paths import candidate_dirs
from lity.services.image_generation.launcher_platform import (
    launch_command as _launch_command,
)
from lity.services.image_generation.launcher_process import (
    kill_pid as _kill_pid,
)
from lity.services.image_generation.launcher_process import (
    kill_process as _kill_process,
)
from lity.services.image_generation.launcher_process import (
    pid_is_running as _pid_is_running,
)
from lity.services.image_generation.launcher_process import (
    process_is_running as _process_is_running,
)
from lity.services.image_generation.launcher_process import (
    terminate_pid as _terminate_pid,
)
from lity.services.image_generation.launcher_process import (
    terminate_process as _terminate_process,
)
from lity.services.image_generation.launcher_process import (
    wait_for_pid_exit as _wait_for_pid_exit,
)
from lity.services.image_generation.settings import ImageSettingsManager

CLIP_PACKAGE_URL = (
    "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
)


@dataclass(frozen=True)
class LaunchResult:
    status: str
    message: str
    install_dir: str = ""
    install_command: str = ""
    run_command: str = ""
    help_url: str = "https://github.com/lllyasviel/stable-diffusion-webui-forge"

    @property
    def launched(self) -> bool:
        return self.status == "launched"


class StableDiffusionLauncher:
    def __init__(
        self,
        settings: ImageSettingsManager,
        popen: Callable[..., Any] = subprocess.Popen,
        run: Callable[..., Any] = subprocess.run,
        log_file: Path | None = None,
    ):
        self.settings = settings
        self._popen = popen
        self._run = run
        self._process: Any = None
        self._active_install_dir: Path | None = None
        self._blocked_installations: dict[str, str] = {}
        self._clip_repair_attempts: dict[str, int] = {}
        self._log_file = log_file
        self._pid_file = log_file.with_suffix(".pid") if log_file is not None else None
        self._log_stream: Any = None

    def launch(self) -> LaunchResult:
        if self.is_launching():
            return LaunchResult(
                status="launching",
                message=(
                    "Stable Diffusion est déjà en cours de lancement. "
                    "Le mode image s'activera automatiquement dès que l'API répond."
                ),
            )

        install_dir = self.find_installation()
        if install_dir is None:
            blocked_result = self._blocked_installation_result()
            if blocked_result is not None:
                return blocked_result

            instructions = installation_instructions()
            return LaunchResult(
                status="missing",
                message=(
                    "Stable Diffusion n'est pas installé ou son dossier n'est pas configuré. "
                    "Installe WebUI Forge avec la commande adaptée à "
                    "cet appareil, puis relance le mode image.\n\n"
                    f"Télécharger :\n{instructions.install_command}\n\n"
                    f"Lancer avec l'API :\n{instructions.run_command}"
                ),
                install_command=instructions.install_command,
                run_command=instructions.run_command,
                help_url=instructions.help_url,
            )

        command = _launch_command(install_dir)
        if command is None:
            instructions = installation_instructions()
            return LaunchResult(
                status="missing",
                install_dir=str(install_dir),
                message=(
                    "Stable Diffusion semble présent, mais aucun script de lancement "
                    "webui.sh, webui-user.sh, webui.bat ou webui-user.bat n'a été trouvé.\n\n"
                    f"Réinstaller/télécharger :\n{instructions.install_command}"
                ),
                install_command=instructions.install_command,
                run_command=instructions.run_command,
                help_url=instructions.help_url,
            )

        try:
            stdout_target: Any = subprocess.DEVNULL
            stderr_target: Any = subprocess.DEVNULL
            if self._log_file is not None:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
                self._log_stream = self._log_file.open("a", encoding="utf-8")
                self._log_stream.write("\n=== Lancement Stable Diffusion ===\n")
                self._log_stream.flush()
                stdout_target = self._log_stream
                stderr_target = subprocess.STDOUT
            build_environment_error = self._prepare_python_build_environment(install_dir)
            if build_environment_error is not None:
                self._close_log_stream()
                return build_environment_error
            popen_kwargs: dict[str, Any] = {
                "cwd": str(install_dir),
                "env": _launch_environment(install_dir),
                "stdout": stdout_target,
                "stderr": stderr_target,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            self._process = self._popen(command, **popen_kwargs)
            self._write_pid_file(self._process)
            self._active_install_dir = install_dir
        except Exception as exc:
            return LaunchResult(
                status="error",
                install_dir=str(install_dir),
                message=f"Impossible de lancer Stable Diffusion : {exc}",
            )

        return LaunchResult(
            status="launched",
            install_dir=str(install_dir),
            message=(
                "Stable Diffusion est en cours de lancement avec l'API activée. "
                "Le mode image s'activera automatiquement dès que l'API répond."
            ),
        )

    def status(self) -> LaunchResult:
        if self.is_launching():
            return LaunchResult(
                status="launching",
                message=(
                    "Stable Diffusion est en cours de lancement. "
                    "Le mode image s'active dès que l'API répond."
                ),
            )

        if self._process is not None:
            returncode = getattr(self._process, "returncode", None)
            self._close_log_stream()
            self._process = None
            self._remove_pid_file()
            tail = self._log_tail()
            install_dir = self._active_install_dir
            if install_dir is not None and _clip_pkg_resources_failure(tail):
                retry = self._relaunch_after_clip_repair(install_dir)
                if retry is not None:
                    return retry

            recommendation = _failure_recommendation(tail)
            if install_dir is not None and _known_upstream_clone_failure(tail):
                self._blocked_installations[str(install_dir)] = tail
            details = f"\n\nDernières lignes du lancement :\n{tail}" if tail else ""
            self._active_install_dir = None
            return LaunchResult(
                status="error",
                message=(
                    "Stable Diffusion s'est arrêté avant que l'API soit prête "
                    f"(code {returncode}).{details}{recommendation}"
                ),
            )

        return LaunchResult(status="stopped", message="Stable Diffusion n'est pas en cours.")

    def _prepare_python_build_environment(self, install_dir: Path) -> LaunchResult | None:
        python = _venv_python(install_dir)
        if python is None:
            return None

        version = self._python_version(python, install_dir)
        if version is not None and version < (3, 10):
            python310 = _python310_command()
            if python310 is None:
                return LaunchResult(
                    status="error",
                    install_dir=str(install_dir),
                    message=(
                        "Stable Diffusion WebUI Forge nécessite Python 3.10 ou plus récent, "
                        f"mais son venv actuel utilise Python {version[0]}.{version[1]}. "
                        "Installe Python 3.10, puis relance le mode image.\n\n"
                        "Commande macOS :\n"
                        "brew install python@3.10 cmake pkg-config cairo protobuf rust git wget\n\n"
                        "Lity recréera ensuite le venv Forge automatiquement."
                    ),
                )
            shutil.rmtree(install_dir / "venv")
            self._write_log(
                "\n=== Venv Forge recréé ===\n"
                f"Ancien venv Python {version[0]}.{version[1]} supprimé. "
                f"Relance avec {python310}.\n"
            )
            return None

        check = self._run(
            [str(python), "-c", "import pkg_resources, wheel"],
            cwd=str(install_dir),
            env=_launch_environment(install_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        if getattr(check, "returncode", 1) == 0:
            clip_error = self._ensure_clip_dependency(python, install_dir)
            if clip_error is not None:
                return clip_error
            return self._ensure_runtime_dependency_pins(python, install_dir)

        self._write_log(
            "\n=== Préparation du venv Stable Diffusion pour CLIP ===\n"
            f"{getattr(check, 'stdout', '')}\n"
        )
        install = self._run(
            [str(python), "-m", "pip", "install", "setuptools<81", "wheel"],
            cwd=str(install_dir),
            env=_launch_environment(install_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        self._write_log(f"{getattr(install, 'stdout', '')}\n")
        if getattr(install, "returncode", 1) != 0:
            return LaunchResult(
                status="error",
                install_dir=str(install_dir),
                message=(
                    "Lity n'a pas pu préparer l'environnement Python de Stable Diffusion "
                    "pour installer CLIP.\n\n"
                    f"Commande : {python} -m pip install 'setuptools<81' wheel\n\n"
                    f"Sortie :\n{getattr(install, 'stdout', '')}"
                ),
            )

        clip_error = self._ensure_clip_dependency(python, install_dir)
        if clip_error is not None:
            return clip_error
        return self._ensure_runtime_dependency_pins(python, install_dir)

    def _ensure_clip_dependency(self, python: Path, install_dir: Path) -> LaunchResult | None:
        check = self._run(
            [str(python), "-c", "import clip"],
            cwd=str(install_dir),
            env=_launch_environment(install_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        if getattr(check, "returncode", 1) == 0:
            return None

        self._write_log("\n=== Installation préventive de CLIP ===\n")
        install = self._run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                CLIP_PACKAGE_URL,
                "--prefer-binary",
            ],
            cwd=str(install_dir),
            env=_launch_environment(install_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        self._write_log(f"{getattr(install, 'stdout', '')}\n")
        if getattr(install, "returncode", 1) != 0:
            return LaunchResult(
                status="error",
                install_dir=str(install_dir),
                message=(
                    "Lity n'a pas pu installer CLIP dans l'environnement Python de "
                    "Stable Diffusion.\n\n"
                    f"Commande : {python} -m pip install --no-build-isolation {CLIP_PACKAGE_URL} "
                    "--prefer-binary\n\n"
                    f"Sortie :\n{getattr(install, 'stdout', '')}"
                ),
            )

        return None

    def _ensure_runtime_dependency_pins(
        self, python: Path, install_dir: Path
    ) -> LaunchResult | None:
        check = self._run(
            [
                str(python),
                "-c",
                (
                    "import numpy, skimage; from skimage import exposure; "
                    "raise SystemExit(0 if numpy.__version__ == '1.26.2' "
                    "and skimage.__version__ == '0.21.0' else 1)"
                ),
            ],
            cwd=str(install_dir),
            env=_launch_environment(install_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        if getattr(check, "returncode", 1) == 0:
            return None

        self._write_log(
            "\n=== Réparation numpy/scikit-image Stable Diffusion ===\n"
            f"{getattr(check, 'stdout', '')}\n"
        )
        install = self._run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--prefer-binary",
                "numpy==1.26.2",
                "scikit-image==0.21.0",
                "scipy==1.11.4",
                "Pillow==9.5.0",
            ],
            cwd=str(install_dir),
            env=_launch_environment(install_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        self._write_log(f"{getattr(install, 'stdout', '')}\n")
        if getattr(install, "returncode", 1) != 0:
            return LaunchResult(
                status="error",
                install_dir=str(install_dir),
                message=(
                    "Lity n'a pas pu réparer les dépendances numpy/scikit-image de "
                    "Stable Diffusion.\n\n"
                    f"Sortie :\n{getattr(install, 'stdout', '')}"
                ),
            )

        return None

    def _python_version(self, python: Path, install_dir: Path) -> tuple[int, int] | None:
        result = self._run(
            [
                str(python),
                "-c",
                "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
            ],
            cwd=str(install_dir),
            env=_launch_environment(install_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        if getattr(result, "returncode", 1) != 0:
            return None
        output = str(getattr(result, "stdout", "")).strip().splitlines()[-1:]
        if not output:
            return None
        try:
            major, minor = output[0].split(".", 1)
            return int(major), int(minor)
        except ValueError:
            return None

    def _relaunch_after_clip_repair(self, install_dir: Path) -> LaunchResult | None:
        key = str(install_dir)
        if self._clip_repair_attempts.get(key, 0) >= 1:
            return None

        self._clip_repair_attempts[key] = self._clip_repair_attempts.get(key, 0) + 1
        self._process = None
        self._active_install_dir = None
        relaunch = self.launch()
        if relaunch.status in {"launched", "launching"}:
            return LaunchResult(
                status=relaunch.status,
                install_dir=relaunch.install_dir,
                install_command=relaunch.install_command,
                run_command=relaunch.run_command,
                help_url=relaunch.help_url,
                message=(
                    "Lity a détecté l'erreur CLIP/pkg_resources, préparé le venv "
                    "Stable Diffusion avec setuptools/wheel, puis relancé le serveur.\n\n"
                    f"{relaunch.message}"
                ),
            )

        return LaunchResult(
            status=relaunch.status,
            install_dir=relaunch.install_dir,
            install_command=relaunch.install_command,
            run_command=relaunch.run_command,
            help_url=relaunch.help_url,
            message=(
                "Lity a détecté l'erreur CLIP/pkg_resources et tenté une réparation, "
                "mais le relancement a encore échoué.\n\n"
                f"{relaunch.message}"
            ),
        )

    def is_launching(self) -> bool:
        return self._process is not None and _process_is_running(self._process)

    def shutdown(self, timeout: float = 8.0) -> LaunchResult:
        if self._process is None:
            return self._shutdown_pid_file_process(timeout)
        if not _process_is_running(self._process):
            self._process = None
            self._active_install_dir = None
            self._remove_pid_file()
            return LaunchResult(status="stopped", message="Stable Diffusion est déjà arrêté.")

        try:
            _terminate_process(self._process)
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process(self._process)
            self._process.wait(timeout=2.0)
        except Exception as exc:
            return LaunchResult(
                status="error", message=f"Arrêt Stable Diffusion impossible : {exc}"
            )
        finally:
            self._close_log_stream()
            self._process = None
            self._active_install_dir = None
            self._remove_pid_file()

        return LaunchResult(status="stopped", message="Stable Diffusion a été arrêté proprement.")

    def _shutdown_pid_file_process(self, timeout: float) -> LaunchResult:
        pid = self._read_pid_file()
        if pid is None or not _pid_is_running(pid):
            self._remove_pid_file()
            return LaunchResult(status="stopped", message="Aucun serveur Stable Diffusion lancé.")

        try:
            _terminate_pid(pid)
            _wait_for_pid_exit(pid, timeout)
        except subprocess.TimeoutExpired:
            _kill_pid(pid)
            _wait_for_pid_exit(pid, 2.0)
        except Exception as exc:
            return LaunchResult(
                status="error", message=f"Arrêt Stable Diffusion impossible : {exc}"
            )
        finally:
            self._remove_pid_file()

        return LaunchResult(status="stopped", message="Stable Diffusion a été arrêté proprement.")

    def _close_log_stream(self) -> None:
        if self._log_stream is not None:
            try:
                self._log_stream.close()
            finally:
                self._log_stream = None

    def _write_log(self, text: str) -> None:
        if self._log_stream is None:
            return
        try:
            self._log_stream.write(text)
            self._log_stream.flush()
        except Exception:
            return

    def _write_pid_file(self, process: Any) -> None:
        if self._pid_file is None:
            return
        pid = getattr(process, "pid", None)
        if pid is None:
            return
        try:
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            self._pid_file.write_text(str(pid), encoding="utf-8")
        except Exception:
            return

    def _read_pid_file(self) -> int | None:
        if self._pid_file is None or not self._pid_file.exists():
            return None
        try:
            return int(self._pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def _remove_pid_file(self) -> None:
        if self._pid_file is None:
            return
        try:
            self._pid_file.unlink(missing_ok=True)
        except Exception:
            return

    def _log_tail(self, max_chars: int = 2400) -> str:
        if self._log_file is None or not self._log_file.exists():
            return ""
        try:
            content = self._log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        return content[-max_chars:].strip()

    def find_installation(self) -> Path | None:
        for candidate in candidate_dirs(self.settings):
            if str(candidate) in self._blocked_installations:
                continue
            if _launch_command(candidate) is not None:
                return candidate
        return None

    def _blocked_installation_result(self) -> LaunchResult | None:
        if not self._blocked_installations:
            return None
        instructions = installation_instructions()
        blocked_dirs = "\n".join(f"- {path}" for path in self._blocked_installations)
        return LaunchResult(
            status="error",
            message=(
                "L'installation Automatic1111 trouvée ne peut pas démarrer: elle dépend "
                "d'un dépôt upstream qui n'est plus clonable publiquement. "
                "Lity ne relance donc pas ce dossier en boucle.\n\n"
                f"Dossier ignoré :\n{blocked_dirs}\n\n"
                "Installe WebUI Forge dans Documents, puis relance le mode image.\n\n"
                f"Télécharger :\n{instructions.install_command}\n\n"
                f"Lancer avec l'API :\n{instructions.run_command}"
            ),
            install_command=instructions.install_command,
            run_command=instructions.run_command,
            help_url=instructions.help_url,
        )


def _launch_environment(install_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    inherited_venv = env.pop("VIRTUAL_ENV", "")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    for name in list(env):
        if name.startswith("VSCODE_GIT_ASKPASS") or name == "VSCODE_GIT_IPC_HANDLE":
            env.pop(name, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"

    if inherited_venv:
        env["PATH"] = _remove_virtualenv_bin_from_path(env.get("PATH", ""), inherited_venv)

    if install_dir is not None and _venv_python(install_dir) is not None:
        env["PIP_NO_BUILD_ISOLATION"] = "1"
    elif install_dir is not None:
        python310 = _python310_command()
        if python310 is not None:
            env["python_cmd"] = python310

    return env


def _python310_command() -> str | None:
    for command in ("python3.10", "/opt/homebrew/bin/python3.10", "/usr/local/bin/python3.10"):
        if shutil.which(command):
            return command
    return None


def _venv_python(install_dir: Path) -> Path | None:
    if sys.platform.startswith("win"):
        candidates = [install_dir / "venv" / "Scripts" / "python.exe"]
    else:
        candidates = [install_dir / "venv" / "bin" / "python"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _remove_virtualenv_bin_from_path(path_value: str, virtualenv_path: str) -> str:
    blocked_entries = {
        _normalized_path(str(Path(virtualenv_path) / "bin")),
        _normalized_path(str(Path(virtualenv_path) / "Scripts")),
    }
    path_entries = [
        entry
        for entry in path_value.split(os.pathsep)
        if entry and _normalized_path(entry) not in blocked_entries
    ]
    return os.pathsep.join(path_entries)


def _normalized_path(path_value: str) -> str:
    return os.path.normcase(os.path.normpath(path_value))


def installation_instructions(platform_name: str = sys.platform) -> LaunchResult:
    if platform_name.startswith("win"):
        return LaunchResult(
            status="instructions",
            message="Commandes Windows PowerShell pour télécharger puis lancer WebUI Forge.",
            install_command=(
                "git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git "
                '"$env:USERPROFILE\\Documents\\stable-diffusion-webui-forge"'
            ),
            run_command=(
                'cd "$env:USERPROFILE\\Documents\\stable-diffusion-webui-forge"; '
                ".\\webui-user.bat --api"
            ),
            help_url="https://github.com/lllyasviel/stable-diffusion-webui-forge",
        )

    if platform_name == "darwin":
        return LaunchResult(
            status="instructions",
            message="Commandes macOS pour télécharger puis lancer WebUI Forge.",
            install_command=(
                "git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git "
                "~/Documents/stable-diffusion-webui-forge"
            ),
            run_command="cd ~/Documents/stable-diffusion-webui-forge && ./webui.sh --api",
            help_url="https://github.com/lllyasviel/stable-diffusion-webui-forge",
        )

    return LaunchResult(
        status="instructions",
        message="Commandes Linux pour télécharger puis lancer WebUI Forge.",
        install_command=(
            "git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git "
            "~/Documents/stable-diffusion-webui-forge"
        ),
        run_command="cd ~/Documents/stable-diffusion-webui-forge && ./webui.sh --api",
    )


def _failure_recommendation(log_tail: str) -> str:
    recommendations: list[str] = []
    python_recommendation = _python_environment_recommendation(log_tail)
    if python_recommendation:
        recommendations.append(python_recommendation)

    if _known_upstream_clone_failure(log_tail):
        instructions = installation_instructions()
        recommendations.append(
            "\n\nCette erreur est connue avec les installations Automatic1111 récentes: "
            "un dépôt upstream requis n'est plus clonable publiquement. "
            "Utilise WebUI Forge, compatible avec l'API /sdapi utilisée par Lity.\n\n"
            f"Télécharger :\n{instructions.install_command}\n\n"
            f"Lancer avec l'API :\n{instructions.run_command}"
        )

    return "".join(recommendations)


def _known_upstream_clone_failure(log_tail: str) -> bool:
    return (
        "Stability-AI/stablediffusion.git" in log_tail
        or "Couldn't clone Stable Diffusion" in log_tail
    )


def _clip_pkg_resources_failure(log_tail: str) -> bool:
    return "No module named 'pkg_resources'" in log_tail and "github.com/openai/CLIP" in log_tail


def _python_environment_recommendation(log_tail: str) -> str:
    if not (
        "INCOMPATIBLE PYTHON VERSION" in log_tail
        or "No module named pip" in log_tail
        or "python venv already activate" in log_tail
    ):
        return ""

    if sys.platform == "darwin":
        return (
            "\n\nLity a détecté un environnement Python incompatible pour Stable Diffusion. "
            "Sur macOS, WebUI recommande Python 3.10 avec Homebrew. "
            "Installe les dépendances, puis recrée le venv WebUI si celui-ci a été créé "
            "avec une mauvaise version de Python.\n\n"
            "Installer les dépendances macOS :\n"
            "brew install cmake protobuf rust python@3.10 git wget\n\n"
            "Recréer le venv WebUI :\n"
            "rm -rf ~/Documents/stable-diffusion-webui/venv\n\n"
            "Relancer avec Python 3.10 :\n"
            "cd ~/Documents/stable-diffusion-webui && python_cmd=python3.10 ./webui.sh --api"
        )

    if sys.platform.startswith("win"):
        return (
            "\n\nLity a détecté un environnement Python incompatible pour Stable Diffusion. "
            "Installe Python 3.10.6 depuis python.org, coche Add Python to PATH, "
            "puis supprime le dossier venv de Stable Diffusion avant de relancer "
            "webui-user.bat --api."
        )

    instructions = installation_instructions()
    return (
        "\n\nLity a détecté un environnement Python incompatible pour Stable Diffusion. "
        "Installe Python 3.10 ou 3.11, supprime le venv WebUI existant s'il a été créé "
        "avec une mauvaise version, puis relance avec l'API.\n\n"
        f"Lancer avec l'API :\n{instructions.run_command}"
    )
