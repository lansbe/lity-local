from __future__ import annotations

import logging
from typing import Any

from lity.services.ai._engine_common import _match_installed_model
from lity.services.external import ServiceHealth

logger = logging.getLogger(__name__)


class ModelAdminMixin:
    """Ollama model administration (list / pull / delete / info / health).

    Mixed into :class:`AIEngine`, which provides ``model`` and ``_ctx_cache``.
    """

    def context_length(self, name: str | None = None) -> int:
        """Max context length the model advertises (cached), or 0 if unknown."""
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            return self.effective_num_ctx() if hasattr(self, "effective_num_ctx") else 0
        model = name or self.model
        if model in self._ctx_cache:
            return self._ctx_cache[model]
        length = 0
        try:
            import ollama

            info = ollama.show(model)
            model_info = (
                info.get("model_info")
                if isinstance(info, dict)
                else getattr(info, "model_info", None)
            )
            if isinstance(model_info, dict):
                for key, value in model_info.items():
                    if key.endswith("context_length") and isinstance(value, int):
                        length = value
                        break
        except Exception as exc:
            logger.info("context_length lookup failed: %s", exc)
        self._ctx_cache[model] = length
        return length

    def select_available_model(
        self,
        installed_models: list[str],
        preferred_model: str | None = None,
    ) -> str:
        models = [model for model in installed_models if model]
        if not models:
            return self.model

        for candidate in (preferred_model, self.model):
            matched = _match_installed_model(candidate, models)
            if matched:
                self.model = matched
                return matched

        self.model = models[0]
        return self.model

    def get_installed_models(self) -> list[str]:
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            try:
                return self._client().list_models()
            except Exception as exc:
                logger.info("Unable to list LM Studio models: %s", exc)
                return []
        try:
            import ollama

            models_info = ollama.list()
            models_list: list[Any]
            if hasattr(models_info, "models"):
                models_list = models_info.models
            elif isinstance(models_info, dict):
                models_list = models_info.get("models", [])
            else:
                models_list = models_info

            names = []
            for model in models_list:
                if hasattr(model, "model"):
                    names.append(model.model)
                elif isinstance(model, dict) and "model" in model:
                    names.append(model["model"])
                elif isinstance(model, dict) and "name" in model:
                    names.append(model["name"])
                elif hasattr(model, "name"):
                    names.append(model.name)
            return names
        except Exception as exc:
            logger.info("Unable to list Ollama models: %s", exc)
            return []

    def get_models_detailed(self) -> list[dict[str, Any]]:
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            return [{"name": name, "size": 0} for name in self.get_installed_models()]
        try:
            import ollama

            info = ollama.list()
            models = (
                info.models
                if hasattr(info, "models")
                else (info.get("models", []) if isinstance(info, dict) else info)
            )
            detailed: list[dict[str, Any]] = []
            for model in models:
                name = (
                    getattr(model, "model", None)
                    or (model.get("model") if isinstance(model, dict) else None)
                    or (
                        model.get("name")
                        if isinstance(model, dict)
                        else getattr(model, "name", None)
                    )
                )
                size = getattr(model, "size", None)
                if size is None and isinstance(model, dict):
                    size = model.get("size")
                if name:
                    detailed.append({"name": str(name), "size": int(size) if size else 0})
            return detailed
        except Exception as exc:
            logger.info("get_models_detailed failed: %s", exc)
            return []

    def pull_model(
        self,
        name: str,
        on_progress: Any | None = None,
        should_cancel: Any | None = None,
    ) -> dict[str, Any]:
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            return {
                "ok": False,
                "message": (
                    "Téléchargement non géré par Lity pour LM Studio. "
                    "Télécharge le modèle dans LM Studio, puis clique sur Réessayer."
                ),
            }
        try:
            import contextlib

            import ollama

            stream = ollama.pull(name, stream=True)
            try:
                for chunk in stream:
                    # Checked before each chunk so cancelling aborts promptly and
                    # closing the stream tears down the underlying HTTP download.
                    if should_cancel is not None and should_cancel():
                        return {
                            "ok": False,
                            "cancelled": True,
                            "message": f"Téléchargement de {name} annulé.",
                        }
                    if on_progress is None:
                        continue
                    if isinstance(chunk, dict):
                        status, completed, total = (
                            chunk.get("status", ""),
                            chunk.get("completed") or 0,
                            chunk.get("total") or 0,
                        )
                    else:
                        status = getattr(chunk, "status", "")
                        completed = getattr(chunk, "completed", 0) or 0
                        total = getattr(chunk, "total", 0) or 0
                    on_progress({"status": status, "completed": completed, "total": total})
            finally:
                closer = getattr(stream, "close", None)
                if callable(closer):
                    with contextlib.suppress(Exception):
                        closer()
            return {"ok": True, "message": f"Modèle {name} téléchargé."}
        except Exception as exc:
            logger.warning("pull_model failed: %s", exc)
            return {"ok": False, "message": str(exc)}

    def delete_model(self, name: str) -> dict[str, Any]:
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            return {
                "ok": False,
                "message": "Suppression non gérée par Lity pour LM Studio.",
            }
        try:
            import ollama

            ollama.delete(name)
            return {"ok": True, "message": f"Modèle {name} supprimé."}
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def supports_vision(self, name: str | None = None) -> bool | None:
        """Can this model read images? Reads Ollama's reported capabilities
        (authoritative, cached). Returns None when Ollama can't be reached, so
        callers fall back to a name heuristic instead of guessing wrong."""
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            return None
        model = name or self.model
        cache = getattr(self, "_vision_cache", None)
        if cache is None:
            cache = self._vision_cache = {}
        if model in cache:
            return cache[model]
        result: bool | None = None
        try:
            import ollama

            info = ollama.show(model)
            capabilities = (
                info.get("capabilities")
                if isinstance(info, dict)
                else getattr(info, "capabilities", None)
            ) or []
            result = "vision" in [str(capability).lower() for capability in capabilities]
        except Exception as exc:  # Ollama down / model gone → let the caller fall back
            logger.info("vision capability lookup failed for %s: %s", model, exc)
            result = None
        cache[model] = result
        return result

    def model_info(self, name: str) -> dict[str, Any]:
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            return {
                "parameters": "",
                "template": "",
                "details": {"backend": "LM Studio"},
                "capabilities": [],
            }
        try:
            import ollama

            info = ollama.show(name)
            if isinstance(info, dict):
                capabilities = info.get("capabilities") or []
                return {
                    "parameters": info.get("parameters", ""),
                    "template": info.get("template", ""),
                    "details": info.get("details", {}),
                    "capabilities": list(capabilities),
                }
            capabilities = getattr(info, "capabilities", None) or []
            return {
                "parameters": getattr(info, "parameters", ""),
                "template": getattr(info, "template", ""),
                "details": getattr(info, "details", {}),
                "capabilities": list(capabilities),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def check_health(self) -> ServiceHealth:
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            models = self.get_installed_models()
            if models:
                return ServiceHealth.up("LM Studio", "Modèles chargés : " + ", ".join(models))
            return ServiceHealth.down(
                "LM Studio",
                "Serveur local indisponible ou aucun modèle chargé sur http://127.0.0.1:1234.",
            )
        models = self.get_installed_models()
        if models:
            return ServiceHealth.up("Ollama", "Modèles installés : " + ", ".join(models))
        return ServiceHealth.down("Ollama", "Aucun modèle détecté ou service Ollama indisponible.")
