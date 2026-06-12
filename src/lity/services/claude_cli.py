from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

RunFn = Callable[..., subprocess.CompletedProcess[str]]
PopenFn = Callable[..., subprocess.Popen[str]]
WhichFn = Callable[[str], str | None]

# Claude Code exposes an adaptive reasoning effort on the models that support it.
# Kept conservative (the trio valid on every effort-capable model); xhigh/max are
# version-specific and left out of the picker so a turn never fails on an unknown
# level.
REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

# Auth is owned by the Claude CLI (OAuth keychain / ~/.claude credentials) or read
# from the environment — Lity never stores a key. These are the variables the CLI
# itself honours, in the same precedence it documents.
_CREDENTIAL_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
)

# Claude has no `debug models` JSON catalogue (unlike Codex), so the picker is fed
# from this static list of the current model line-up. Effort support is per-model
# (authoritative, from the Claude API reference):
#   - low/medium/high + max : Fable 5, Opus 4.6/4.7/4.8, Sonnet 4.6
#   - xhigh                  : only Fable 5 and Opus 4.7/4.8 (added in 4.7)
#   - Haiku 4.5              : NO effort parameter — the CLI errors if --effort is sent
_EFFORT_DESCRIPTIONS = {
    "low": "Rapide",
    "medium": "Équilibré",
    "high": "Approfondi (défaut Claude)",
    "xhigh": "Très approfondi",
    "max": "Maximal",
}
_ALL_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
_NO_XHIGH = ["low", "medium", "high", "max"]
_STATIC_MODELS = [
    {
        "slug": "claude-fable-5",
        "display_name": "Claude Fable 5",
        "description": "Le plus puissant — un cran au-dessus d'Opus.",
        "efforts": _ALL_EFFORTS,
    },
    {
        "slug": "claude-opus-4-8",
        "display_name": "Claude Opus 4.8",
        "description": "Opus le plus capable — agentique et code complexes.",
        "efforts": _ALL_EFFORTS,
    },
    {
        "slug": "claude-opus-4-7",
        "display_name": "Claude Opus 4.7",
        "description": "Opus précédent, très autonome.",
        "efforts": _ALL_EFFORTS,
    },
    {
        "slug": "claude-opus-4-6",
        "display_name": "Claude Opus 4.6",
        "description": "Opus antérieur (sans xhigh).",
        "efforts": _NO_XHIGH,
    },
    {
        "slug": "claude-sonnet-4-6",
        "display_name": "Claude Sonnet 4.6",
        "description": "Équilibré — rapide et solide au quotidien.",
        "efforts": _NO_XHIGH,
    },
    {
        "slug": "claude-haiku-4-5",
        "display_name": "Claude Haiku 4.5",
        "description": "Le plus rapide et le plus léger (sans effort de raisonnement).",
        "efforts": [],
    },
]
_DEFAULT_MODEL = "claude-opus-4-8"
_DEFAULT_EFFORT = "high"

# Short aliases the CLI also accepts (`--model opus`), mapped to a catalogue slug
# so effort support can be looked up whichever form the caller stored.
_MODEL_ALIASES = {
    "fable": "claude-fable-5",
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


def _efforts_for_model(model: str) -> list[str] | None:
    """Supported `--effort` levels for a model, or None when the model is unknown
    (an unrecognised string is left to the CLI — we don't second-guess it)."""
    slug = _MODEL_ALIASES.get(model, model)
    for item in _STATIC_MODELS:
        if item["slug"] == slug:
            return item["efforts"]
    return None


class ClaudeCliClient:
    """Small wrapper around the Claude Code CLI headless flow (``claude -p``).

    Mirrors :class:`CodexCliClient`: Lity never reads Claude's cached credentials.
    The Claude CLI owns its OAuth login (keychain / ``~/.claude``), token refresh
    and account policy; Lity only calls documented commands and inherits whatever
    credential the CLI/environment already provides.
    """

    def __init__(
        self,
        *,
        runner: RunFn = subprocess.run,
        popen: PopenFn = subprocess.Popen,
        which: WhichFn = shutil.which,
        command: str = "claude",
        environ: Mapping[str, str] | None = None,
    ):
        self.runner = runner
        self.popen = popen
        self.which = which
        self.command = command
        self._environ = environ

    def _command_path(self) -> str | None:
        if Path(self.command).exists():
            return self.command
        return self.which(self.command)

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
                    "Claude CLI introuvable. Installe Claude Code "
                    "(`npm i -g @anthropic-ai/claude-code`) puis connecte-toi."
                ),
            }
        authenticated = False
        message = ""
        try:
            proc = self.runner(
                [command, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            authenticated = proc.returncode == 0
            message = ((proc.stdout or "") + (proc.stderr or "")).strip()
        except Exception as exc:
            message = str(exc)
        # A token/API key in the environment is a valid auth even when `auth status`
        # is unavailable (older CLI) or reports a missing subscription login.
        if not authenticated and self._env_has_credentials():
            authenticated = True
            message = "Authentifié via une variable d'environnement (clé API ou token OAuth)."
        if not message:
            message = (
                "Claude est connecté."
                if authenticated
                else "Claude n'est pas connecté. Lance `claude` puis `/login`, ou `claude setup-token`."
            )
        return {"available": True, "authenticated": authenticated, "message": message}

    def start_login(self) -> dict[str, Any]:
        command = self._command_path()
        if not command:
            return {
                "ok": False,
                "message": "Claude CLI introuvable. Installe Claude Code puis relance Lity.",
                "process": None,
            }
        try:
            process = self.popen(
                [command, "setup-token"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            return {"ok": False, "message": str(exc), "process": None}
        return {
            "ok": True,
            "message": (
                "Connexion Claude lancée (`claude setup-token`). "
                "Termine l'authentification dans le navigateur."
            ),
            "process": process,
        }

    def model_catalog(self) -> dict[str, Any]:
        """Return the static Claude model list (no `claude debug models` exists).

        Shaped like the Codex catalogue so the frontend can reuse the same picker.
        """
        models = []
        for priority, item in enumerate(_STATIC_MODELS):
            efforts = item["efforts"]
            levels = [
                {"effort": effort, "description": _EFFORT_DESCRIPTIONS.get(effort, "")}
                for effort in efforts
            ]
            if _DEFAULT_EFFORT in efforts:
                default_level = _DEFAULT_EFFORT
            else:
                default_level = efforts[0] if efforts else ""
            models.append(
                {
                    "slug": item["slug"],
                    "display_name": item["display_name"],
                    "description": item["description"],
                    "default_reasoning_level": default_level,
                    "supported_reasoning_levels": levels,
                    "priority": priority * 10,
                }
            )
        return {
            "ok": True,
            "models": models,
            "default_model": _DEFAULT_MODEL,
            "message": f"{len(models)} modèle(s) Claude disponible(s).",
        }

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
            return {"ok": False, "content": "", "message": "Claude CLI introuvable."}
        effort = (reasoning_effort or "").strip()
        if effort and effort not in REASONING_EFFORTS:
            return {
                "ok": False,
                "content": "",
                "message": f"Effort de raisonnement invalide : {effort}.",
            }
        # Drop an effort the target model can't accept — Haiku has none, and
        # Opus 4.6 / Sonnet 4.6 reject xhigh; the CLI would error otherwise. An
        # unknown model keeps the effort (we don't second-guess a custom slug).
        supported = _efforts_for_model(model.strip()) if model.strip() else None
        if effort and supported is not None and effort not in supported:
            effort = ""

        # `--permission-mode plan` is the read-only guarantee (the Claude equivalent
        # of Codex's `--sandbox read-only`): Claude can inspect the project but never
        # edits files or runs commands — Lity applies any proposed change itself,
        # through its reviewed editor. `--output-format json` yields the final answer
        # in `.result`. The prompt is the trailing positional argument.
        command = [
            command_path,
            "-p",
            "--output-format",
            "json",
            "--permission-mode",
            "plan",
        ]
        if model.strip():
            command += ["--model", model.strip()]
        if effort:
            command += ["--effort", effort]
        command.append(prompt)

        kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }
        if workdir:
            kwargs["cwd"] = str(Path(workdir).expanduser())

        try:
            proc = self.runner(command, **kwargs)
        except subprocess.TimeoutExpired:
            return {"ok": False, "content": "", "message": f"Claude a expiré après {timeout}s."}
        except Exception as exc:
            return {"ok": False, "content": "", "message": str(exc)}

        if proc.returncode != 0:
            error = ((proc.stderr or "") + (proc.stdout or "")).strip()
            return {
                "ok": False,
                "content": self._extract_content(proc.stdout),
                "message": error or f"Claude a quitté avec le code {proc.returncode}.",
            }
        return {
            "ok": True,
            "content": self._extract_content(proc.stdout),
            "usage": self._extract_usage(proc.stdout),
            "message": "Réponse générée par Claude.",
        }

    @staticmethod
    def _extract_usage(stdout: str | None) -> dict[str, Any] | None:
        """Pull cost + token usage out of `--output-format json`.

        Claude's headless JSON carries `total_cost_usd`, a `usage` object, and a
        per-model `modelUsage` map — the only usage data exposed without the
        interactive `/usage` panel (subscription quota windows are not headless).
        """
        raw = (stdout or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        def _tokens(block: Any) -> tuple[int, int]:
            if not isinstance(block, dict):
                return 0, 0
            inp = sum(
                int(block.get(key) or 0)
                for key in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
                if isinstance(block.get(key), (int, float))
            )
            out = int(block.get("output_tokens") or 0)
            return inp, out

        input_tokens, output_tokens = _tokens(payload.get("usage"))
        cost = payload.get("total_cost_usd")
        cost = float(cost) if isinstance(cost, (int, float)) else None

        by_model: dict[str, dict[str, Any]] = {}
        model_usage = payload.get("modelUsage")
        if isinstance(model_usage, dict):
            for model, block in model_usage.items():
                mi, mo = _tokens(block)
                model_cost = block.get("cost_usd") if isinstance(block, dict) else None
                by_model[str(model)] = {
                    "input_tokens": mi,
                    "output_tokens": mo,
                    "cost_usd": float(model_cost) if isinstance(model_cost, (int, float)) else None,
                }
        return {
            "cost_usd": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "by_model": by_model,
        }

    @staticmethod
    def _extract_content(stdout: str | None) -> str:
        """Pull the final text out of `--output-format json`, tolerating raw text."""
        raw = (stdout or "").strip()
        if not raw:
            return ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(payload, dict):
            for key in ("result", "content", "text"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return raw
