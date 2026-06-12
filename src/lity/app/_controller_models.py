from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelsMixin:
    """Model suggestions, hardware-aware ranking, capability checks, generation
    stats, and the web-page fetch helper for AgentController.

    Uses ``self.engine`` and ``self._ensure_web()`` (both on the controller).
    """

    def model_suggestions(self) -> list[dict[str, str]]:
        return [
            {"name": "qwen3", "category": "chat", "note": "Chat/agent polyvalent (8B, 2026)"},
            {"name": "llama3.2", "category": "chat", "note": "Chat léger et rapide (3B)"},
            {"name": "qwen2.5-coder", "category": "chat", "note": "Spécialisé code"},
            {"name": "mistral", "category": "chat", "note": "Généraliste 7B"},
            {"name": "llama3.2-vision", "category": "vision", "note": "Multimodal (images)"},
            {"name": "bge-m3", "category": "embedding", "note": "Embeddings multilingues (FR)"},
            {"name": "nomic-embed-text", "category": "embedding", "note": "Embeddings RAG (léger)"},
        ]

    def hardware_info(self) -> dict[str, Any]:
        from lity.core.hardware import detect_hardware

        return detect_hardware()

    def fetch_page(self, url: str) -> dict[str, Any]:
        """Fetch + clean a web page (reuses the web fetcher) to add as context."""
        return self._ensure_web()["fetcher"].fetch(url)

    def generation_stats(self) -> dict[str, Any]:
        """Latest turn's speed (tokens/s) and context usage vs the model window."""
        stats = dict(getattr(self.engine, "last_stats", {}) or {})
        ctx_len = 0
        if hasattr(self.engine, "context_length"):
            try:
                ctx_len = int(self.engine.context_length())
            except Exception:
                ctx_len = 0
        used = int(stats.get("context_used", 0))
        pct = round(used / ctx_len * 100) if ctx_len and used else 0
        return {
            "tokens_per_sec": float(stats.get("tokens_per_sec", 0.0)),
            "context_used": used,
            "context_length": ctx_len,
            "usage_pct": pct,
        }

    def model_supports_tools(self, name: str = "") -> bool | None:
        """Whether a model exposes the 'tools' capability (None if unknown).

        Used to warn before enabling agent/web search on a model (e.g. a
        reasoning/distill model like deepseek-r1) that cannot tool-call.
        """
        target = (name or "").strip() or getattr(self.engine, "model", "")
        if not target or not hasattr(self.engine, "model_info"):
            return None
        info = self.engine.model_info(target)
        capabilities = info.get("capabilities") if isinstance(info, dict) else None
        if not capabilities:
            return None  # older Ollama / unknown → don't raise a false alarm
        return "tools" in [str(cap).lower() for cap in capabilities]

    def model_recommendations(self) -> dict[str, Any]:
        """Detect the device and rank Ollama models best→worst for it."""
        from lity.core.hardware import detect_hardware
        from lity.core.image_model_advisor import rank_image_models
        from lity.core.model_advisor import rank_models
        from lity.core.video_model_advisor import rank_video_models

        hardware = detect_hardware()
        installed: list[Any] = []
        if hasattr(self.engine, "get_models_detailed"):
            try:
                installed = self.engine.get_models_detailed()
            except Exception:
                installed = []
        rows = rank_models(hardware, installed)
        self._apply_live_tool_capability(rows)
        image_rows = rank_image_models(hardware, self._installed_image_checkpoints())
        self._mark_selected_image_model(image_rows)
        video_rows = rank_video_models(hardware, self._installed_video_models())
        self._mark_selected_video_model(video_rows)
        return {
            "hardware": hardware,
            "models": rows,
            "image_models": image_rows,
            "video_models": video_rows,
        }

    def _mark_selected_image_model(self, rows: list[dict[str, Any]]) -> None:
        """Flag the downloaded model the in-process engine will generate with.

        The active model is the user's saved choice, or — if none/stale — the
        first downloaded one, mirroring the engine's own fallback."""
        from lity.services.image_generation.checkpoints import installed_image_records

        paths = getattr(self, "paths", None)
        if paths is None:
            return
        installed = [record.name for record in installed_image_records(paths.image_models_dir)]
        if not installed:
            return
        settings = getattr(self, "settings", None)
        selected = (settings.get("image_selected_model", "") if settings else "") or ""
        if selected not in installed:
            selected = installed[0]
        for row in rows:
            row["selected"] = bool(row.get("installed")) and str(row.get("name")) == selected

    def _installed_image_checkpoints(self) -> list[str]:
        names: list[str] = []
        # 1) The AUTOMATIC1111 webui checkpoint folder, if one is configured.
        image_manager = getattr(self, "image_manager", None)
        settings = getattr(image_manager, "settings", None)
        checkpoints_dir = getattr(settings, "checkpoints_dir", "") if settings else ""
        if checkpoints_dir:
            path = Path(str(checkpoints_dir)).expanduser()
            if path.is_dir():
                names += [
                    item.name
                    for item in path.iterdir()
                    if item.is_file() and item.suffix.lower() in {".safetensors", ".ckpt"}
                ]
        # 2) Lity's own auto-download folder: ~/Documents/Lity/Models/Images/<name>/…
        #    A model is "installed" once its folder holds at least one weight file.
        lity_dir = getattr(getattr(self, "paths", None), "image_models_dir", None)
        if lity_dir and Path(lity_dir).is_dir():
            for sub in Path(lity_dir).iterdir():
                if sub.is_dir() and any(
                    f.is_file() and f.suffix.lower() in {".safetensors", ".ckpt", ".gguf"}
                    for f in sub.rglob("*")
                ):
                    names.append(sub.name)
        return sorted(set(names))

    def _mark_selected_video_model(self, rows: list[dict[str, Any]]) -> None:
        """Flag the downloaded video model the in-process engine will generate
        with — the user's saved choice, or the first downloaded one."""
        from lity.services.video_generation.checkpoints import installed_video_models

        paths = getattr(self, "paths", None)
        if paths is None:
            return
        installed = [name for name, _ in installed_video_models(paths.video_models_dir)]
        if not installed:
            return
        settings = getattr(self, "settings", None)
        selected = (settings.get("video_selected_model", "") if settings else "") or ""
        if selected not in installed:
            selected = installed[0]
        for row in rows:
            row["selected"] = bool(row.get("installed")) and str(row.get("name")) == selected

    def _installed_video_models(self) -> list[str]:
        """Names of downloaded video models (folders holding a real model)."""
        from lity.services.video_generation.checkpoints import installed_video_models

        paths = getattr(self, "paths", None)
        if paths is None:
            return []
        return [name for name, _ in installed_video_models(paths.video_models_dir)]

    def _apply_live_tool_capability(self, rows: list[dict[str, Any]]) -> None:
        """For INSTALLED models, Ollama's own capability report is authoritative
        and overrides the offline family prior — the prior only exists for
        models that aren't downloaded yet."""
        for row in rows:
            if not row.get("installed") or row.get("kind") == "embed":
                continue
            try:
                live = self.model_supports_tools(row["name"])
            except Exception:
                continue
            if live is not None:
                row["tool_use"] = bool(live)
