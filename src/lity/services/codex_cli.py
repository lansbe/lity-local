from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

RunFn = Callable[..., subprocess.CompletedProcess[str]]
PopenFn = Callable[..., subprocess.Popen[str]]
WhichFn = Callable[[str], str | None]

REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


class CodexCliClient:
    """Small wrapper around the Codex CLI ChatGPT-login flow.

    The app never reads Codex's cached credentials directly. Codex CLI owns the
    browser login, token refresh and account/workspace policy; Lity only calls
    documented commands.
    """

    def __init__(
        self,
        *,
        runner: RunFn = subprocess.run,
        popen: PopenFn = subprocess.Popen,
        which: WhichFn = shutil.which,
        command: str = "codex",
    ):
        self.runner = runner
        self.popen = popen
        self.which = which
        self.command = command

    def _command_path(self) -> str | None:
        if Path(self.command).exists():
            return self.command
        found = self.which(self.command)
        if found:
            return found
        mac_app_cli = Path("/Applications/Codex.app/Contents/Resources/codex")
        if self.command == "codex" and mac_app_cli.exists():
            return str(mac_app_cli)
        return None

    def status(self) -> dict[str, Any]:
        command = self._command_path()
        if not command:
            return {
                "available": False,
                "authenticated": False,
                "message": "Codex CLI introuvable. Installe Codex puis lance `codex login`.",
            }
        try:
            proc = self.runner(
                [command, "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            return {"available": True, "authenticated": False, "message": str(exc)}
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return {
            "available": True,
            "authenticated": proc.returncode == 0,
            "message": output
            or (
                "Codex est connecté."
                if proc.returncode == 0
                else "Codex n'est pas connecté. Lance `codex login`."
            ),
        }

    def start_login(self) -> dict[str, Any]:
        command = self._command_path()
        if not command:
            return {
                "ok": False,
                "message": "Codex CLI introuvable. Installe Codex puis relance Lity.",
                "process": None,
            }
        try:
            process = self.popen(
                [command, "login"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            return {"ok": False, "message": str(exc), "process": None}
        return {
            "ok": True,
            "message": "Connexion Codex lancée. Termine l'authentification dans la fenêtre ouverte.",
            "process": process,
        }

    def model_catalog(self, *, bundled: bool = False) -> dict[str, Any]:
        command = self._command_path()
        if not command:
            return {
                "ok": False,
                "models": [],
                "default_model": "",
                "message": "Codex CLI introuvable. Installe Codex puis relance Lity.",
            }
        try:
            command_args = [command, "debug", "models"]
            if bundled:
                command_args.append("--bundled")
            proc = self.runner(
                command_args,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            return {"ok": False, "models": [], "default_model": "", "message": str(exc)}
        if proc.returncode != 0:
            error = ((proc.stderr or "") + (proc.stdout or "")).strip()
            return {
                "ok": False,
                "models": [],
                "default_model": "",
                "message": error or f"Codex debug models a quitté avec le code {proc.returncode}.",
            }
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "models": [],
                "default_model": "",
                "message": f"Catalogue Codex illisible : {exc}.",
            }

        models = self._sanitize_models(payload.get("models") if isinstance(payload, dict) else [])
        visible_models = [model for model in models if model.pop("_visible", False)] or models
        default_model = visible_models[0]["slug"] if visible_models else ""
        return {
            "ok": True,
            "models": visible_models,
            "default_model": default_model,
            "message": f"{len(visible_models)} modèle(s) Codex disponible(s).",
        }

    def _sanitize_models(self, raw_models: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_models, list):
            return []
        models = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug") or "").strip()
            if not slug:
                continue
            levels = self._sanitize_reasoning_levels(item.get("supported_reasoning_levels"))
            default_level = str(item.get("default_reasoning_level") or "").strip()
            if default_level not in REASONING_EFFORTS:
                default_level = levels[0]["effort"] if levels else "medium"
            if not levels:
                levels = [{"effort": default_level, "description": ""}]
            priority = item.get("priority", 9999)
            models.append(
                {
                    "slug": slug,
                    "display_name": str(item.get("display_name") or slug),
                    "description": str(item.get("description") or ""),
                    "default_reasoning_level": default_level,
                    "supported_reasoning_levels": levels,
                    "priority": priority if isinstance(priority, int) else 9999,
                    "_visible": item.get("visibility") == "list",
                }
            )
        return sorted(models, key=lambda model: (model["priority"], model["slug"]))

    def _sanitize_reasoning_levels(self, raw_levels: Any) -> list[dict[str, str]]:
        if not isinstance(raw_levels, list):
            return []
        levels = []
        seen: set[str] = set()
        for item in raw_levels:
            if isinstance(item, dict):
                effort = str(item.get("effort") or "").strip()
                description = str(item.get("description") or "")
            else:
                effort = str(item or "").strip()
                description = ""
            if effort in REASONING_EFFORTS and effort not in seen:
                levels.append({"effort": effort, "description": description})
                seen.add(effort)
        return levels

    def run_prompt(
        self,
        prompt: str,
        *,
        model: str = "",
        reasoning_effort: str = "",
        workdir: Path | str | None = None,
        timeout: int = 600,
    ) -> dict[str, Any]:
        command_path = self._command_path()
        if not command_path:
            return {"ok": False, "content": "", "message": "Codex CLI introuvable."}
        effort = (reasoning_effort or "").strip()
        if effort and effort not in REASONING_EFFORTS:
            return {
                "ok": False,
                "content": "",
                "message": f"Effort de raisonnement invalide : {effort}.",
            }

        with tempfile.TemporaryDirectory(prefix="lity-codex-") as tmp:
            output_file = Path(tmp) / "last-message.txt"
            command = [
                command_path,
                "exec",
                # JSONL events on stdout — used only to read token usage; the
                # answer itself still comes from --output-last-message.
                "--json",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--output-last-message",
                str(output_file),
            ]
            if workdir:
                command += ["--cd", str(Path(workdir).expanduser())]
            if model.strip():
                command += ["--model", model.strip()]
            if effort:
                command += ["--config", f"model_reasoning_effort='{effort}'"]
            command.append("-")

            try:
                proc = self.runner(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "content": "",
                    "message": f"Codex a expiré après {timeout}s.",
                }
            except Exception as exc:
                return {"ok": False, "content": "", "message": str(exc)}

            content = ""
            if output_file.exists():
                content = output_file.read_text(encoding="utf-8").strip()
            # stdout is JSONL now (--json): reconstruct the message from it only as
            # a fallback when the output file is empty — never show raw JSONL.
            if not content:
                content = self._extract_message(proc.stdout)
            if proc.returncode != 0:
                error = ((proc.stderr or "") + (proc.stdout or "")).strip()
                return {
                    "ok": False,
                    "content": content,
                    "message": error or f"Codex a quitté avec le code {proc.returncode}.",
                }
            return {
                "ok": True,
                "content": content,
                "usage": self._extract_usage(proc.stdout),
                "message": "Réponse générée par Codex.",
            }

    @staticmethod
    def _extract_usage(stdout: str | None) -> dict[str, Any] | None:
        """Read token usage from the `token_count` JSONL event (`--json`).

        Each `codex exec` is `--ephemeral`, so the cumulative `total_token_usage`
        on the last event equals this turn's usage. Rate-limit windows are not
        read here — Codex emits them as `null` in exec output."""
        info: Any = None
        for line in (stdout or "").splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            found = CodexCliClient._find_token_usage(obj)
            if found is not None:
                info = found  # keep the latest (cumulative) snapshot
        if not isinstance(info, dict):
            return None

        def _int(value: Any) -> int:
            return int(value) if isinstance(value, (int, float)) else 0

        input_tokens = _int(info.get("input_tokens")) + _int(info.get("cached_input_tokens"))
        output_tokens = _int(info.get("output_tokens")) + _int(info.get("reasoning_output_tokens"))
        if not input_tokens and not output_tokens:
            return None
        return {
            "cost_usd": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "by_model": {},
        }

    @staticmethod
    def _find_token_usage(obj: Any) -> Any:
        if isinstance(obj, dict):
            if obj.get("type") == "token_count":
                info = obj.get("info")
                if isinstance(info, dict):
                    total = info.get("total_token_usage")
                    return total if isinstance(total, dict) else info
            for value in obj.values():
                found = CodexCliClient._find_token_usage(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = CodexCliClient._find_token_usage(value)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _extract_message(stdout: str | None) -> str:
        """Best-effort: pull the final agent message out of the JSONL stream.

        Only used when --output-last-message produced nothing, so a defensive
        heuristic is acceptable here."""
        text = ""
        for line in (stdout or "").splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            found = CodexCliClient._find_agent_message(obj)
            if found:
                text = found
        return text.strip()

    @staticmethod
    def _find_agent_message(obj: Any) -> str | None:
        if isinstance(obj, dict):
            type_ = str(obj.get("type") or "")
            if "agent_message" in type_ or type_ in ("assistant", "message"):
                for key in ("message", "text", "last_agent_message", "content"):
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
            direct = obj.get("last_agent_message")
            if isinstance(direct, str) and direct.strip():
                return direct
            for value in obj.values():
                found = CodexCliClient._find_agent_message(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = CodexCliClient._find_agent_message(value)
                if found:
                    return found
        return None
