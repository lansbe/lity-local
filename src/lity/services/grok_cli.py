from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

RunFn = Callable[..., subprocess.CompletedProcess[str]]
PopenFn = Callable[..., subprocess.Popen[str]]
WhichFn = Callable[[str], str | None]

# Auth is owned by the Grok CLI: browser/device OAuth writes credentials to
# ~/.grok/auth.json, or an xAI API key is read from the environment. Lity never
# reads the auth cache and ignores secret config keys; only presence/configured
# model names are checked.
_CREDENTIAL_ENV_VARS = (
    "GROK_CODE_XAI_API_KEY",
    "XAI_API_KEY",
    "GROK_DEPLOYMENT_KEY",
)
_AUTH_FILE = Path("~/.grok/auth.json").expanduser()
_CONFIG_FILES = (
    Path("/etc/grok/managed_config.toml"),
    Path("~/.grok/managed_config.toml").expanduser(),
    Path("~/.grok/config.toml").expanduser(),
    Path("~/.grok/requirements.toml").expanduser(),
    Path("/etc/grok/requirements.toml"),
)

# Grok's actual model ids are installation/account dependent. The CLI exposes
# them through `grok models`; this fallback is intentionally minimal and uses the
# currently documented/default CLI id instead of API-only names.
_STATIC_MODELS = [
    {
        "slug": "grok-build",
        "display_name": "Grok Build",
        "description": "Modèle de codage par défaut du CLI Grok Build.",
    },
]
_DEFAULT_MODEL = "grok-build"
_REMOVED_MODEL_ALIASES = {
    "grok-build-0.1": "grok-build",
    "grok-4": "grok-build",
    "grok-3": "grok-build",
}


class GrokCliClient:
    """Small wrapper around the xAI Grok Build CLI headless flow (`grok -p`).

    Mirrors :class:`CodexCliClient` / :class:`ClaudeCliClient`: Lity never reads
    Grok's cached credentials. The Grok CLI owns its OAuth login (``~/.grok``) or
    reads an xAI API key from the environment; Lity only calls documented commands.

    NOTE: the Grok Build CLI is new and several flags below are best-effort from
    its docs (the ``--output-format json`` schema and the read-only/sandbox flag
    are not fully documented). Headless runs are kept read-only by NOT passing any
    auto-approve flag, so write/exec tools stay blocked.
    """

    def __init__(
        self,
        *,
        runner: RunFn = subprocess.run,
        popen: PopenFn = subprocess.Popen,
        which: WhichFn = shutil.which,
        command: str = "grok",
        environ: Mapping[str, str] | None = None,
        auth_file: Path | None = None,
        config_files: Iterable[Path] | None = None,
    ):
        self.runner = runner
        self.popen = popen
        self.which = which
        self.command = command
        self._environ = environ
        self._auth_file = auth_file if auth_file is not None else _AUTH_FILE
        self._config_files = tuple(config_files) if config_files is not None else _CONFIG_FILES

    def _command_path(self) -> str | None:
        if Path(self.command).exists():
            return self.command
        found = self.which(self.command)
        if found:
            return found
        installed = Path("~/.grok/bin/grok").expanduser()
        if self.command == "grok" and installed.exists():
            return str(installed)
        return None

    def _env(self) -> Mapping[str, str]:
        return self._environ if self._environ is not None else os.environ

    def _env_has_credentials(self) -> bool:
        env = self._env()
        return any((env.get(name) or "").strip() for name in _CREDENTIAL_ENV_VARS)

    def status(self) -> dict[str, Any]:
        command = self._command_path()
        if not command:
            return {
                "available": False,
                "authenticated": False,
                "message": (
                    "Grok CLI introuvable. Installe-le "
                    "(`curl -fsSL https://x.ai/cli/install.sh | bash`) puis connecte-toi."
                ),
            }
        # Auth is a browser login (creds in ~/.grok/auth.json) or an xAI API key
        # in the environment — checked without spawning a subcommand (fast, and no
        # interactive `grok inspect` that could block).
        authenticated = self._env_has_credentials()
        if not authenticated:
            try:
                authenticated = self._auth_file.exists()
            except Exception:
                authenticated = False
        message = (
            "Grok est connecté."
            if authenticated
            else (
                "Grok n'est pas connecté. Lance `grok login`, "
                "`grok login --device-auth`, ou exporte XAI_API_KEY."
            )
        )
        return {"available": True, "authenticated": authenticated, "message": message}

    def start_login(self, *, device_auth: bool = False) -> dict[str, Any]:
        command = self._command_path()
        if not command:
            return {
                "ok": False,
                "message": "Grok CLI introuvable. Installe Grok Build puis relance Lity.",
                "process": None,
            }
        try:
            args = [command, "login"]
            if device_auth:
                args.append("--device-auth")
            process = self.popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            return {"ok": False, "message": str(exc), "process": None}
        message = (
            "Connexion Grok lancée (`grok login --device-auth`). "
            "Copie le code indiqué par le terminal."
            if device_auth
            else (
                "Connexion Grok lancée (`grok login`). "
                "Termine l'authentification dans le navigateur."
            )
        )
        return {
            "ok": True,
            "message": message,
            "process": process,
        }

    def model_catalog(self) -> dict[str, Any]:
        """Return models from `grok models`, plus custom models declared in config.toml."""
        discovered_models, discovered_default = self._models_from_cli()
        source_models = discovered_models or list(_STATIC_MODELS)
        models = [self._catalog_row(item, priority) for priority, item in enumerate(source_models)]
        custom_models, configured_default = self._custom_models_from_config()
        existing = {model["slug"] for model in models}
        for offset, model in enumerate(custom_models, start=len(models)):
            if model["slug"] in existing:
                continue
            existing.add(model["slug"])
            models.append(self._catalog_row(model, offset))
        default_candidate = configured_default or discovered_default or _DEFAULT_MODEL
        default_model = default_candidate if default_candidate in existing else _DEFAULT_MODEL
        return {
            "ok": True,
            "models": models,
            "default_model": default_model,
            "message": f"{len(models)} modèle(s) Grok disponible(s).",
        }

    @staticmethod
    def _catalog_row(item: dict[str, str], priority: int) -> dict[str, Any]:
        return {
            "slug": item["slug"],
            "display_name": item["display_name"],
            "description": item["description"],
            "default_reasoning_level": "",
            "supported_reasoning_levels": [],
            "priority": priority * 10,
        }

    def run_prompt(
        self,
        prompt: str,
        *,
        model: str = "",
        workdir: Path | str | None = None,
        timeout: int = 600,
        session_id: str = "",
        resume: str = "",
        continue_session: bool = False,
        output_format: str = "json",
        plugin_dirs: Iterable[Path | str] | None = None,
        always_approve: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        command_path = self._command_path()
        if not command_path:
            return {"ok": False, "content": "", "message": "Grok CLI introuvable."}

        # `grok -p <prompt>` runs one prompt headless and prints to stdout.
        # `--output-format json` emits a single JSON object; `--no-alt-screen`
        # avoids a fullscreen TUI takeover and `--no-auto-update` keeps it
        # deterministic. No `--always-approve` is passed, so write/exec tools stay
        # blocked — Lity applies any proposed change itself, via its reviewed editor.
        output_format = (
            output_format if output_format in {"plain", "json", "streaming-json"} else "json"
        )
        command = [
            command_path,
            "-p",
            prompt,
            "--output-format",
            output_format,
            "--no-alt-screen",
            "--no-auto-update",
        ]
        normalized_model = self._normalize_model(model)
        if normalized_model:
            command += ["--model", normalized_model]
        if workdir:
            command += ["--cwd", str(Path(workdir).expanduser())]
        if session_id.strip():
            command += ["--session-id", session_id.strip()]
        if resume.strip():
            command += ["--resume", resume.strip()]
        if continue_session:
            command.append("--continue")
        for plugin_dir in plugin_dirs or ():
            command += ["--plugin-dir", str(Path(plugin_dir).expanduser())]
        if always_approve:
            command.append("--always-approve")

        if output_format == "streaming-json" and on_chunk is not None:
            return self._run_streaming(
                command, timeout=timeout, on_chunk=on_chunk, should_cancel=should_cancel
            )

        try:
            proc = self.runner(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "content": "", "message": f"Grok a expiré après {timeout}s."}
        except Exception as exc:
            return {"ok": False, "content": "", "message": str(exc)}

        if proc.returncode != 0:
            error = ((proc.stderr or "") + (proc.stdout or "")).strip()
            return {
                "ok": False,
                "content": self._extract_content(proc.stdout),
                "message": error or f"Grok a quitté avec le code {proc.returncode}.",
            }
        return {
            "ok": True,
            "content": self._extract_content(proc.stdout),
            "usage": self._extract_usage(proc.stdout),
            "message": "Réponse générée par Grok.",
        }

    def _run_streaming(
        self,
        command: list[str],
        *,
        timeout: int,
        on_chunk: Callable[[str], None],
        should_cancel: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        stdout_lines: list[str] = []
        try:
            process = self.popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            return {"ok": False, "content": "", "message": str(exc)}

        try:
            stream = process.stdout or []
            for raw_line in stream:
                if should_cancel is not None and should_cancel():
                    with contextlib.suppress(Exception):
                        process.kill()
                    return {
                        "ok": False,
                        "content": self._extract_content("\n".join(stdout_lines)),
                        "message": "Grok a été annulé.",
                        "streamed": True,
                    }
                stdout_lines.append(raw_line)
                chunk = self._extract_streaming_content(raw_line, strip=False)
                if chunk:
                    on_chunk(chunk)
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
            return {
                "ok": False,
                "content": self._extract_content("\n".join(stdout_lines)),
                "message": f"Grok a expiré après {timeout}s.",
                "streamed": True,
            }
        except Exception as exc:
            with contextlib.suppress(Exception):
                process.kill()
            return {
                "ok": False,
                "content": self._extract_content("\n".join(stdout_lines)),
                "message": str(exc),
                "streamed": True,
            }

        stdout = "".join(stdout_lines)
        stderr = ""
        with contextlib.suppress(Exception):
            stderr = (process.stderr.read() if process.stderr else "") or ""
        if returncode != 0:
            return {
                "ok": False,
                "content": self._extract_content(stdout),
                "message": (stderr + stdout).strip() or f"Grok a quitté avec le code {returncode}.",
                "streamed": True,
            }
        return {
            "ok": True,
            "content": self._extract_content(stdout),
            "usage": self._extract_usage(stdout),
            "message": "Réponse générée par Grok.",
            "streamed": True,
        }

    def _custom_models_from_config(self) -> tuple[list[dict[str, str]], str]:
        models: list[dict[str, str]] = []
        default = ""
        for path in self._config_files:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            parsed, parsed_default = self._parse_custom_model_config(text)
            if parsed_default:
                default = parsed_default
            models.extend(parsed)
        return models, default

    def _models_from_cli(self) -> tuple[list[dict[str, str]], str]:
        command = self._command_path()
        if not command:
            return [], ""
        try:
            proc = self.runner(
                [command, "models"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            return [], ""
        if proc.returncode != 0:
            return [], ""
        return self._parse_models_output((proc.stdout or "") + "\n" + (proc.stderr or ""))

    @staticmethod
    def _parse_models_output(text: str) -> tuple[list[dict[str, str]], str]:
        models: list[dict[str, str]] = []
        default = ""
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        for raw_line in text.splitlines():
            line = ansi.sub("", raw_line).strip()
            if not line:
                continue
            default_match = re.fullmatch(r"Default model:\s*(\S+)", line)
            if default_match:
                default = default_match.group(1).strip()
                continue
            model_match = re.fullmatch(r"[*-]\s+([A-Za-z0-9_.-]+)(?:\s+\(default\))?", line)
            if not model_match:
                continue
            slug = model_match.group(1).strip()
            if not slug:
                continue
            if "(default)" in line:
                default = slug
            models.append(
                {
                    "slug": slug,
                    "display_name": slug,
                    "description": "Modèle disponible via `grok models`.",
                }
            )
        return models, default

    @staticmethod
    def _normalize_model(model: str) -> str:
        cleaned = (model or "").strip()
        return _REMOVED_MODEL_ALIASES.get(cleaned, cleaned)

    @staticmethod
    def _parse_custom_model_config(text: str) -> tuple[list[dict[str, str]], str]:
        models: list[dict[str, str]] = []
        default = ""
        current_slug = ""
        current: dict[str, str] = {}

        def flush() -> None:
            if not current_slug:
                return
            display = current.get("name") or current_slug
            model_id = current.get("model") or current_slug
            base_url = current.get("base_url") or "configuration Grok"
            models.append(
                {
                    "slug": current_slug,
                    "display_name": display,
                    "description": f"Modèle Grok personnalisé ({model_id}) via {base_url}.",
                }
            )

        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            section = re.fullmatch(r"\[([^\]]+)\]", line)
            if section:
                flush()
                name = section.group(1).strip()
                current_slug = name.removeprefix("model.") if name.startswith("model.") else ""
                current = {}
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*=\s*['\"]([^'\"]+)['\"]", line)
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            if current_slug:
                current[key] = value
            elif key == "default":
                default = value
        flush()
        return models, default

    @staticmethod
    def _payload(stdout: str | None) -> Any:
        raw = (stdout or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # streaming-json / plain fallback: take the last JSON object on a line.
            for line in reversed(raw.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            return raw

    @staticmethod
    def _extract_content(stdout: str | None) -> str:
        """Pull the answer text from `--output-format json` (schema undocumented,
        so several key names are tried), tolerating plain text."""
        streamed = GrokCliClient._extract_streaming_content(stdout)
        if streamed:
            return streamed
        payload = GrokCliClient._payload(stdout)
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            for key in ("text", "result", "message", "content", "output", "response"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return (stdout or "").strip() if not isinstance(payload, dict) else ""

    @staticmethod
    def _extract_usage(stdout: str | None) -> dict[str, Any] | None:
        """Best-effort token usage from the JSON result (key names undocumented)."""
        payload = GrokCliClient._usage_payload(stdout)
        if not isinstance(payload, dict):
            return None
        usage = None
        for key in ("tokenUsage", "token_usage", "usage", "tokens"):
            block = payload.get(key)
            if isinstance(block, dict):
                usage = block
                break
        if not isinstance(usage, dict):
            return None

        def _int(*names: str) -> int:
            for name in names:
                value = usage.get(name)
                if isinstance(value, (int, float)):
                    return int(value)
            return 0

        input_tokens = _int("input", "input_tokens", "prompt_tokens")
        output_tokens = _int("output", "output_tokens", "completion_tokens")
        if not input_tokens and not output_tokens:
            return None
        cost = payload.get("cost")
        if not isinstance(cost, (int, float)):
            cost = payload.get("total_cost_usd")
        return {
            "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "by_model": {},
        }

    @staticmethod
    def _json_lines(stdout: str | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    @staticmethod
    def _extract_streaming_content(stdout: str | None, *, strip: bool = True) -> str:
        chunks: list[str] = []
        for payload in GrokCliClient._json_lines(stdout):
            for container_key in ("delta", "content"):
                container = payload.get(container_key)
                if isinstance(container, dict):
                    text = container.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            for key in ("text", "chunk"):
                value = payload.get(key)
                if isinstance(value, str):
                    chunks.append(value)
            if payload.get("type") == "text":
                value = payload.get("data")
                if isinstance(value, str):
                    chunks.append(value)
        content = "".join(chunks)
        return content.strip() if strip else content

    @staticmethod
    def _usage_payload(stdout: str | None) -> Any:
        rows = GrokCliClient._json_lines(stdout)
        for payload in reversed(rows):
            if any(
                isinstance(payload.get(key), dict)
                for key in ("tokenUsage", "token_usage", "usage", "tokens")
            ):
                return payload
        return GrokCliClient._payload(stdout)
