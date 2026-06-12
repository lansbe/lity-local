from __future__ import annotations

import logging
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lity.services.web.http_util import get_json

logger = logging.getLogger(__name__)

CONTAINER_NAME = "lity-searxng"
IMAGE = "searxng/searxng:latest"
DEFAULT_PORT = 8080
_PORT_RANGE = range(DEFAULT_PORT, DEFAULT_PORT + 10)

# First run pulls the image (~300 MB): give it room, but poll fast once up.
_INSTALL_TIMEOUT_S = 300
_POLL_INTERVAL_S = 2.0

# Minimal SearXNG config: inherits the defaults and ONLY enables what Lity
# needs — the JSON API (off by default, and without it the provider gets 403).
_SETTINGS_TEMPLATE = """# Généré par Lity — configuration minimale de SearXNG.
use_default_settings: true
general:
  instance_name: "Lity SearXNG"
server:
  secret_key: "{secret}"
  limiter: false
  image_proxy: false
search:
  formats:
    - html
    - json
"""

# run_fn(cmd: list[str], timeout: int) -> (exit_code, stdout). Injected so the
# installer is unit-testable without Docker.
RunFn = Callable[[list[str], int], tuple[int, str]]
# probe_fn(url: str) -> bool (the JSON search endpoint answers correctly).
ProbeFn = Callable[[str], bool]
EventFn = Callable[[str, dict], None]


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as exc:
        return 1, str(exc)


def probe_searxng(url: str, timeout: int = 3) -> bool:
    """True when the instance answers the JSON search API (what Lity uses).

    Quiet on purpose: a down SearXNG is an EXPECTED state (the user may not have
    installed it), so a failed probe must not spam the log on every health poll.
    """
    base = (url or "").rstrip("/")
    if not base:
        return False
    data = get_json(f"{base}/search?q=ping&format=json", timeout=timeout, quiet=True)
    return isinstance(data, dict) and "results" in data


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0


class SearxngInstaller:
    """One-click local SearXNG: status probe + Docker install with auto-config.

    Everything external is injected (subprocess runner, HTTP probe, clock) so
    the logic is testable without Docker or network. ``install()`` is
    synchronous — the API layer runs it in a background thread and forwards
    the progress events to the UI.
    """

    def __init__(
        self,
        config_dir: Path,
        *,
        run_fn: RunFn = _run,
        probe_fn: ProbeFn = probe_searxng,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        port_free_fn: Callable[[int], bool] = _port_free,
    ):
        self.config_dir = Path(config_dir)
        self._run = run_fn
        self._probe = probe_fn
        self._sleep = sleep_fn
        self._clock = clock
        self._port_free = port_free_fn

    # ------------------------------------------------------------- status
    def docker_ready(self) -> bool:
        """Docker installed AND its daemon answering."""
        if shutil.which("docker") is None:
            return False
        code, _output = self._run(["docker", "ps", "--format", "{{.ID}}"], 10)
        return code == 0

    def container_state(self) -> str:
        """ "running" | "exited" | … | "absent"."""
        code, output = self._run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name=^{CONTAINER_NAME}$",
                "--format",
                "{{.State}}",
            ],
            10,
        )
        state = output.strip().splitlines()[0].strip() if code == 0 and output.strip() else ""
        return state or "absent"

    def status(self, url: str) -> dict[str, Any]:
        """What the UI needs to decide: is web search REALLY ready?"""
        reachable = self._probe(url)
        docker = self.docker_ready() if not reachable else True
        return {
            "url": url,
            "reachable": reachable,
            "docker": docker,
            "container": self.container_state() if docker and not reachable else "unknown",
        }

    # ------------------------------------------------------------ install
    def install(
        self,
        on_event: EventFn,
        persist_url: Callable[[str], None],
    ) -> dict[str, Any]:
        """Install/start a local SearXNG and wait until its JSON API answers.

        Steps: write the auto-config (JSON API enabled) → start the existing
        container or `docker run` a new one on a free port → poll the search
        endpoint → persist the working URL in the settings. Returns
        ``{"ok", "url", "message"}`` and never raises.
        """

        def emit(stage: str, message: str) -> None:
            on_event("searxng_setup", {"stage": stage, "message": message, "done": False})

        if not self.docker_ready():
            return {
                "ok": False,
                "url": "",
                "message": "Docker est requis : installe Docker Desktop "
                "(https://docker.com) puis réessaie.",
            }

        emit("config", "Écriture de la configuration SearXNG (API JSON activée)…")
        config_path = self._write_config()

        state = self.container_state()
        if state == "running":
            port = self._container_port() or DEFAULT_PORT
        elif state != "absent":
            emit("start", "Démarrage du conteneur SearXNG existant…")
            code, output = self._run(["docker", "start", CONTAINER_NAME], 30)
            if code != 0:
                return {"ok": False, "url": "", "message": f"docker start a échoué : {output}"}
            port = self._container_port() or DEFAULT_PORT
        else:
            port = next((p for p in _PORT_RANGE if self._port_free(p)), DEFAULT_PORT)
            emit(
                "run",
                f"Création du conteneur SearXNG sur le port {port} (téléchargement "
                "de l'image au premier lancement, cela peut prendre quelques minutes)…",
            )
            code, output = self._run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    CONTAINER_NAME,
                    "--restart",
                    "unless-stopped",
                    "-p",
                    f"{port}:8080",
                    "-v",
                    f"{config_path.parent}:/etc/searxng",
                    IMAGE,
                ],
                _INSTALL_TIMEOUT_S,
            )
            if code != 0:
                return {"ok": False, "url": "", "message": f"docker run a échoué : {output}"}

        url = f"http://localhost:{port}"
        emit("wait", "Attente du démarrage de SearXNG…")
        deadline = self._clock() + _INSTALL_TIMEOUT_S
        while self._clock() < deadline:
            if self._probe(url):
                # SearXNG is genuinely up: persistence is a side effect that must
                # not change the verdict (and must honour "never raises"), so a
                # failing persist still reports success — the container is live.
                try:
                    persist_url(url)
                except Exception:  # pragma: no cover - persistence is best-effort
                    logger.info("persist_url failed after a successful SearXNG start")
                return {"ok": True, "url": url, "message": f"SearXNG opérationnel sur {url}."}
            self._sleep(_POLL_INTERVAL_S)
        return {
            "ok": False,
            "url": url,
            "message": "SearXNG ne répond pas encore — vérifie le conteneur "
            f"`docker logs {CONTAINER_NAME}` puis réessaie.",
        }

    def _write_config(self) -> Path:
        directory = self.config_dir / "searxng"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "settings.yml"
        if not path.exists():
            path.write_text(
                _SETTINGS_TEMPLATE.format(secret=secrets.token_hex(32)), encoding="utf-8"
            )
        return path

    def _container_port(self) -> int | None:
        code, output = self._run(["docker", "port", CONTAINER_NAME, "8080"], 10)
        if code != 0 or ":" not in output:
            return None
        try:
            return int(output.strip().splitlines()[0].rsplit(":", 1)[1])
        except (ValueError, IndexError):
            return None
