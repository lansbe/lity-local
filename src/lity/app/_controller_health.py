from __future__ import annotations

from typing import Any

from lity.app._modutil import _module_available

DEFAULT_SEARXNG_URL = "http://localhost:8080"


class HealthMixin:
    """Service health checks for AgentController."""

    def health(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        installed: list[str] = []
        try:
            installed = self.engine.get_installed_models()
        except Exception:
            installed = []

        if hasattr(self.engine, "check_health"):
            health = self.engine.check_health()
            results.append({"name": health.name, "ok": health.ok, "detail": health.detail})
        else:
            results.append(
                {
                    "name": "Ollama",
                    "ok": bool(installed),
                    "detail": f"{len(installed)} modèle(s)" if installed else "Indisponible",
                }
            )

        embedding = self._embedding_model()
        base = embedding.split(":", 1)[0]
        has_embed = any(base in model for model in installed)
        results.append(
            {
                "name": "Embeddings",
                "ok": has_embed,
                "detail": f"Modèle {embedding}"
                if has_embed
                else f"Absent — installe-le : ollama pull {embedding}",
            }
        )

        results.append(self._image_engine_health())
        results.append(self._video_engine_health())

        voice_ok = _module_available("faster_whisper") and _module_available("piper")
        results.append(
            {
                "name": "Voix",
                "ok": voice_ok,
                "detail": "Whisper + Piper disponibles"
                if voice_ok
                else "Dépendances audio indisponibles sur ce système",
            }
        )

        get = self.settings.get if self.settings is not None else lambda key, default=None: default
        web_enabled = bool(get("web_search_enabled", False))
        searxng_url = (get("searxng_url", DEFAULT_SEARXNG_URL) or "").strip()
        searxng_up = web_enabled and bool(searxng_url) and self._probe_searxng(searxng_url)
        has_ddg = _module_available("ddgs") or _module_available("duckduckgo_search")
        backends = []
        if searxng_up:
            backends.append("SearXNG (local)")
        if has_ddg:
            backends.append("DuckDuckGo")
        backends.append("Wikipédia")
        if not web_enabled:
            web_ok = False
            web_detail = "Désactivée — active la pastille « Web » sous le chat"
        elif searxng_up or has_ddg:
            web_ok = True
            web_detail = "Activée — moteurs : " + ", ".join(backends)
            if not searxng_up:
                web_detail += " · SearXNG non démarré (installation auto via la pastille Web)"
        else:
            web_ok = False
            web_detail = (
                "Limitée à Wikipédia — aucun moteur installé. Clique sur la pastille "
                "« Web » pour installer SearXNG (Docker) automatiquement."
            )
        if not _module_available("trafilatura"):
            web_detail += " · extraction basique (trafilatura indisponible)"
        results.append({"name": "Recherche web", "ok": web_ok, "detail": web_detail})
        return results

    def _image_engine_health(self) -> dict[str, Any]:
        """Report the local image engine (no external server): is it installed,
        is a model downloaded, is a session running."""
        from lity.services.image_generation.checkpoints import installed_image_records
        from lity.services.image_generation.local_engine import dependencies_available
        from lity.services.image_generation.mlx_engine import mlx_dependencies_available

        name = "Stable Diffusion"
        if self.is_image_session_active():
            return {"name": name, "ok": True, "detail": "Mode image actif"}
        # Either engine being present counts: diffusers (SD/SDXL) or MLX (mflux).
        if not dependencies_available() and not mlx_dependencies_available():
            return {
                "name": name,
                "ok": False,
                "detail": "Moteur local non installé — s'installe à la 1re activation du mode image",
            }
        try:
            models = installed_image_records(self.paths.image_models_dir)
        except Exception:
            models = []
        if not models:
            return {
                "name": name,
                "ok": False,
                "detail": "Aucun modèle téléchargé — Modèles → Images",
            }
        return {
            "name": name,
            "ok": True,
            "detail": f"Prêt en local — {len(models)} modèle(s) téléchargé(s)",
        }

    def _video_engine_health(self) -> dict[str, Any]:
        """Report the local video engine (no external server): is it installed,
        is a model downloaded, is a session running."""
        from lity.services.video_generation.checkpoints import installed_video_models
        from lity.services.video_generation.local_engine import dependencies_available

        name = "Génération vidéo"
        if self.is_video_session_active():
            return {"name": name, "ok": True, "detail": "Mode vidéo actif"}
        if not dependencies_available():
            return {
                "name": name,
                "ok": False,
                "detail": "Moteur local non installé — s'installe à la 1re activation du mode vidéo",
            }
        try:
            models = installed_video_models(self.paths.video_models_dir)
        except Exception:
            models = []
        if not models:
            return {
                "name": name,
                "ok": False,
                "detail": "Aucun modèle téléchargé — Modèles → Vidéos",
            }
        return {
            "name": name,
            "ok": True,
            "detail": f"Prêt en local — {len(models)} modèle(s) téléchargé(s)",
        }

    def _probe_searxng(self, url: str) -> bool:
        """Quick reachability check of the local SearXNG JSON API, cached briefly."""
        import time

        now = time.monotonic()
        cache = getattr(self, "_searxng_probe_cache", None)
        if cache is not None and cache[0] == url and now - cache[1] < 15:
            return cache[2]
        try:
            from lity.services.web.searxng_setup import probe_searxng

            result = probe_searxng(url, timeout=2)
        except Exception:
            result = False
        self._searxng_probe_cache = (url, now, result)
        return result
