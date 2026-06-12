from __future__ import annotations

import json
import logging
import re
from typing import Any

from lity.services.ai.prompts import DEFAULT_MODEL_NAME, INTENT_ROUTER_PROMPT

logger = logging.getLogger(__name__)

# Constrained-decoding schema (Ollama `format=`) for the file-intent classifier:
# guarantees a valid action enum and shape, so the brittle "scan free text for a
# JSON object" path always receives well-formed JSON.
_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "set_working_dir",
                "open_file",
                "close_file",
                "reload_file",
                "load_context",
                "none",
            ],
        },
        "path_raw": {"type": ["string", "null"]},
        "targets": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action"],
}


class IntentRouter:
    def __init__(self, model: str = DEFAULT_MODEL_NAME):
        self.model = model

    def get_file_intent(self, user_input: str) -> dict[str, Any]:
        prompt = INTENT_ROUTER_PROMPT.format(user_input=user_input)
        raw_content = ""
        try:
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format=_INTENT_SCHEMA,
                think=False,  # short structured routing: no reasoning needed
                options={"temperature": 0},
            )
            raw_content = response["message"]["content"].strip()
            start = raw_content.find("{")
            end = raw_content.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found")

            data = json.loads(raw_content[start:end])
            if not isinstance(data, dict):
                raise ValueError("Intent JSON root must be an object")

            action = data.get("action")
            valid_actions = {
                "set_working_dir",
                "open_file",
                "close_file",
                "reload_file",
                "load_context",
                "none",
            }
            if action not in valid_actions:
                return {"action": "none", "path_raw": None, "targets": []}

            path_raw = data.get("path_raw")
            if path_raw is not None and not isinstance(path_raw, str):
                path_raw = None

            targets = data.get("targets", [])
            if isinstance(targets, str):
                targets = [targets]
            elif not isinstance(targets, list):
                targets = []

            data["path_raw"] = path_raw
            data["targets"] = [str(target).strip() for target in targets if target]
            return data
        except Exception as exc:
            logger.info("Intent routing fallback: %s; raw=%r", exc, raw_content)
            return {"action": "none", "path_raw": None, "targets": []}

    def process_intent(self, user_input: str, file_manager: Any) -> dict[str, Any]:
        clean_input = user_input.lower().strip()
        obvious_chat = {"?", "??", "ca va?", "ça va?", "salut", "bonjour", "allo", "ok", "d'accord"}
        if clean_input in obvious_chat:
            return {"handled": False, "action": "none", "message": "", "system_context": ""}

        # Chat-path file ops use ONLY the fast heuristic; anything more nuanced
        # is handled by the agent tool-loop — a single LLM-driven path, instead
        # of a second competing one here. get_file_intent stays as a utility.
        intent = _heuristic_file_intent(user_input)
        action = intent.get("action", "none")
        path_raw = intent.get("path_raw")
        targets = intent.get("targets", [])

        if action == "set_working_dir" and (
            not path_raw or path_raw.strip() in {"?", ".", ""} or len(path_raw.strip()) < 2
        ):
            action = "none"
            path_raw = None

        result = {"handled": False, "action": action, "message": "", "system_context": ""}
        if action == "none":
            return result

        success = False
        message = ""
        if action == "set_working_dir" and path_raw:
            success, message = file_manager.set_working_dir(path_raw)
            result["handled"] = True
        elif action == "load_context" and targets:
            loaded = []
            for target in targets:
                success, message = file_manager.load_file(target, user_input=user_input)
                if success:
                    loaded.append(target)
            if loaded:
                message = "Contexte chargé : " + ", ".join(loaded)
            else:
                message = "Aucun fichier correspondant n'a été trouvé dans le répertoire."
            result["handled"] = False
        elif action == "open_file" and path_raw:
            success, message = file_manager.load_file(path_raw, user_input=user_input)
            result["handled"] = True
        elif action == "close_file":
            target_path = None if path_raw == "__current__" else path_raw
            target_path = target_path or file_manager.current_file_path
            success, message = file_manager.close_file(target_path)
            result["handled"] = True
        elif action == "reload_file":
            target_path = None if path_raw == "__current__" else path_raw
            target_path = target_path or file_manager.current_file_path
            if target_path:
                success, message = file_manager.load_file(target_path)
                if success:
                    message += " (Rechargé)"
            else:
                message = "Aucun fichier à recharger."
            result["handled"] = True

        result["message"] = message
        result["system_context"] = f"[{'SUCCÈS' if success else 'ERREUR'}] {message}"
        return result


def _heuristic_file_intent(user_input: str) -> dict[str, Any]:
    clean_input = user_input.strip()
    lowered = clean_input.lower()

    if lowered.startswith(("/openfile ", "/workdir ", "/closefile", "/reloadfile")):
        return {"action": "none", "path_raw": None, "targets": []}

    if _starts_with_any(lowered, ("ferme", "close", "closefile")):
        return {"action": "close_file", "path_raw": _extract_path(clean_input), "targets": []}

    if _starts_with_any(lowered, ("recharge", "reload", "reloadfile")):
        return {"action": "reload_file", "path_raw": _extract_path(clean_input), "targets": []}

    if _mentions_workdir(lowered):
        path = _extract_path(clean_input)
        if path:
            return {"action": "set_working_dir", "path_raw": path, "targets": []}

    if _starts_with_any(lowered, ("ouvre", "ouvrir", "open", "charge", "charger")):
        path = _extract_path(clean_input)
        if path:
            return {"action": "open_file", "path_raw": path, "targets": []}

    if "contexte" in lowered:
        targets = _extract_file_like_targets(clean_input)
        if targets:
            return {"action": "load_context", "path_raw": None, "targets": targets}

    return {"action": "none", "path_raw": None, "targets": []}


def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text == prefix or text.startswith(prefix + " ") for prefix in prefixes)


def _mentions_workdir(text: str) -> bool:
    return any(marker in text for marker in ("workdir", "répertoire", "repertoire", "dossier"))


def _extract_path(user_input: str) -> str | None:
    lowered = user_input.lower()
    if "fichier actif" in lowered or lowered.endswith(" actif") or lowered == "actif":
        return "__current__"

    quoted = re.search(r"['\"]([^'\"]+)['\"]", user_input)
    if quoted:
        return quoted.group(1).strip()

    tokens = [token.strip(" .,:;") for token in user_input.split()]
    skip = {
        "ouvre",
        "ouvrir",
        "open",
        "charge",
        "charger",
        "ferme",
        "close",
        "closefile",
        "recharge",
        "reload",
        "reloadfile",
        "le",
        "la",
        "les",
        "fichier",
        "dossier",
        "répertoire",
        "repertoire",
        "workdir",
        "actif",
        "à",
        "sur",
        "dans",
    }
    candidates = [token for token in tokens if token and token.lower() not in skip]
    if not candidates:
        return None
    return " ".join(candidates)


def _extract_file_like_targets(user_input: str) -> list[str]:
    targets = []
    for token in user_input.split():
        clean = token.strip(" ,:;()[]'\"")
        if "." in clean or "/" in clean or "\\" in clean:
            targets.append(clean)
    return targets
