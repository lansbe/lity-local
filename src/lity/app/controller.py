from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from lity.app._controller_audio import AudioMixin
from lity.app._controller_background import BackgroundTaskMixin
from lity.app._controller_conversations import ConversationMixin, _safe_filename
from lity.app._controller_git import GitMixin
from lity.app._controller_health import DEFAULT_SEARXNG_URL, HealthMixin
from lity.app._controller_models import ModelsMixin
from lity.app._controller_retrieval import RetrievalMixin
from lity.app.results import AiResponseResult, ErrorResult, IntentHandledResult, TextResult
from lity.app.services import AppServices
from lity.infrastructure.paths import AppPaths
from lity.services.ai.openai_compatible import (
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_LM_STUDIO_MODEL,
    LM_STUDIO_RECOMMENDED_MODELS,
    normalize_base_url,
)
from lity.services.ai.prompts import DEFAULT_ASSISTANT_NAME

__all__ = [
    "AgentController",
    "DEFAULT_SEARXNG_URL",
    "_safe_filename",
    "requires_file_context",
    "should_extract_fact",
]

logger = logging.getLogger(__name__)

_CODEX_MEMORY_BLOCK_RE = re.compile(r"\[LITY_MEMORY\](.*?)\[/LITY_MEMORY\]", re.I | re.S)


class AgentController(
    RetrievalMixin,
    GitMixin,
    HealthMixin,
    ConversationMixin,
    BackgroundTaskMixin,
    AudioMixin,
    ModelsMixin,
):
    def __init__(self, paths: AppPaths | None = None, services: AppServices | None = None):
        self.paths = paths or AppPaths.create()
        self.services = services or AppServices.create(self.paths)
        self.settings = self.services.settings
        self.engine = self.services.engine
        self.memory = self.services.memory
        self.files = self.services.files
        self.router = self.services.router
        self.editor = self.services.editor
        self.image_manager = self.services.image_manager
        self.video_manager = self.services.video_manager
        self._indexer: Any = None
        self._rag_enabled = False
        self._reranker: Any = None  # local cross-encoder, built lazily
        self._reranker_tried = False
        self._memory_indexer: Any = None  # cross-session memory (built lazily)
        self._fact_store: Any = None  # durable-fact semantic recall (built lazily)
        self._mcp: Any = None  # MCP client manager (built lazily)
        self._mcp_tried = False
        self._skills: Any = None  # SkillStore (built lazily)
        self._skills_tried = False
        self._skills_router: Any = None  # SkillRouter (built lazily, reset on settings change)
        self._characters: Any = None  # CharacterStore (built lazily)
        self._character_generator: Any = None
        self._stt: Any = None
        self._tts: Any = None
        self._web: Any = None
        self._auto_num_ctx: int | None = None  # hardware probe result, cached
        # Per-process usage tally for the CLI providers (Codex/Claude), broken
        # down per model. In-memory only — the "session" view, since the CLIs
        # expose no headless subscription-quota windows.
        self._cli_usage: dict[str, Any] = {
            "codex": self._empty_usage(),
            "claude": self._empty_usage(),
            "grok": self._empty_usage(),
        }
        self._apply_settings()

    @staticmethod
    def _empty_usage() -> dict[str, Any]:
        return {
            "turns": 0,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "by_model": {},
        }

    def _record_cli_usage(self, provider: str, fallback_model: str, usage: Any) -> None:
        """Accumulate a turn's usage for a CLI provider, by model."""
        store = self._cli_usage.setdefault(provider, self._empty_usage())
        store["turns"] += 1
        if not isinstance(usage, dict):
            return
        cost = usage.get("cost_usd")
        turn_in = int(usage.get("input_tokens") or 0)
        turn_out = int(usage.get("output_tokens") or 0)
        if isinstance(cost, (int, float)):
            store["cost_usd"] += float(cost)
        store["input_tokens"] += turn_in
        store["output_tokens"] += turn_out
        by_model = usage.get("by_model") if isinstance(usage.get("by_model"), dict) else {}
        if not by_model:
            # No per-model breakdown (Codex): attribute to the configured model.
            label = (fallback_model or "").strip() or "(défaut)"
            by_model = {
                label: {
                    "input_tokens": turn_in,
                    "output_tokens": turn_out,
                    "cost_usd": cost if isinstance(cost, (int, float)) else None,
                }
            }
        for model, stats in by_model.items():
            entry = store["by_model"].setdefault(
                str(model),
                {"turns": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            entry["turns"] += 1
            entry["input_tokens"] += int(stats.get("input_tokens") or 0)
            entry["output_tokens"] += int(stats.get("output_tokens") or 0)
            model_cost = stats.get("cost_usd")
            if isinstance(model_cost, (int, float)):
                entry["cost_usd"] += float(model_cost)

    def cli_usage(self) -> dict[str, Any]:
        """A serializable snapshot of CLI-provider usage for this process."""
        result: dict[str, Any] = {}
        for provider in ("claude", "codex", "grok"):
            store = self._cli_usage.get(provider, self._empty_usage())
            by_model = [
                {
                    "model": model,
                    "turns": entry["turns"],
                    "input_tokens": entry["input_tokens"],
                    "output_tokens": entry["output_tokens"],
                    "total_tokens": entry["input_tokens"] + entry["output_tokens"],
                    "cost_usd": round(entry["cost_usd"], 4),
                }
                for model, entry in sorted(
                    store["by_model"].items(),
                    key=lambda kv: -(kv[1]["input_tokens"] + kv[1]["output_tokens"]),
                )
            ]
            result[provider] = {
                "turns": store["turns"],
                "cost_usd": round(store["cost_usd"], 4),
                "input_tokens": store["input_tokens"],
                "output_tokens": store["output_tokens"],
                "total_tokens": store["input_tokens"] + store["output_tokens"],
                "by_model": by_model,
            }
        return result

    SETTINGS_KEYS = (
        "custom_instructions",
        "embedding_model",
        "selected_model",
        "default_agent",
        "default_yolo",
        "saved_prompts",
        "web_search_enabled",
        "searxng_url",
        "cross_session_memory",
        "verify_command",
        "utility_model",
        "context_window",
        "agent_orchestration",
        "yolo_command_allowlist",
        "web_setup_resolved",
        "web_answer_gate",
        "chat_provider",
        "lmstudio_base_url",
        "lmstudio_model",
        "codex_model",
        "codex_reasoning_effort",
        "claude_model",
        "claude_effort",
        "grok_model",
        "skills_enabled",
        "skills_disabled",
        "skills_semantic",
    )

    def get_settings(self) -> dict[str, Any]:
        get = self.settings.get if self.settings is not None else lambda key, default=None: default
        prompts = get("saved_prompts", []) or []
        return {
            "custom_instructions": get("custom_instructions", "") or "",
            "embedding_model": get("embedding_model", "nomic-embed-text") or "nomic-embed-text",
            "selected_model": get("selected_model", getattr(self.engine, "model", "")) or "",
            "default_agent": bool(get("default_agent", False)),
            "default_yolo": bool(get("default_yolo", False)),
            "saved_prompts": prompts if isinstance(prompts, list) else [],
            # Default OFF: web search needs a local engine the user must opt into
            # (otherwise the toolbar pastille would read "activé" with nothing
            # behind it, and the first click — an off→on toggle — could never
            # trigger the one-click setup).
            "web_search_enabled": bool(get("web_search_enabled", False)),
            "searxng_url": get("searxng_url", DEFAULT_SEARXNG_URL),
            "cross_session_memory": bool(get("cross_session_memory", True)),
            "verify_command": get("verify_command", "") or "",
            "utility_model": get("utility_model", "") or "",
            "context_window": get("context_window", "auto") or "auto",
            "agent_orchestration": bool(get("agent_orchestration", True)),
            "yolo_command_allowlist": bool(get("yolo_command_allowlist", False)),
            # True once the user made a DELIBERATE web-setup choice (installed
            # SearXNG or chose the fallback) — so the one-click modal stops
            # auto-appearing. Dismissing the modal does NOT set this.
            "web_setup_resolved": bool(get("web_setup_resolved", False)),
            # Harness-enforced web persistence: push back a hedge/non-answer so
            # the agent reads more sources before concluding.
            "web_answer_gate": bool(get("web_answer_gate", True)),
            "chat_provider": self._chat_provider(get),
            "lmstudio_base_url": normalize_base_url(get("lmstudio_base_url", "")),
            "lmstudio_model": get("lmstudio_model", DEFAULT_LM_STUDIO_MODEL)
            or DEFAULT_LM_STUDIO_MODEL,
            "codex_model": get("codex_model", "") or "",
            "codex_reasoning_effort": self._codex_reasoning_effort(get),
            "claude_model": get("claude_model", "") or "",
            "claude_effort": self._claude_effort(get),
            "grok_model": get("grok_model", "") or "",
            # Skills ("Compétences"): a master toggle, and an opt-in semantic
            # matcher (uses the local embedding model on top of the lexical one).
            "skills_enabled": bool(get("skills_enabled", True)),
            "skills_semantic": bool(get("skills_semantic", False)),
        }

    @staticmethod
    def _chat_provider(get: Any) -> str:
        provider = get("chat_provider", "ollama") or "ollama"
        return (
            provider if provider in ("ollama", "lmstudio", "codex", "claude", "grok") else "ollama"
        )

    @staticmethod
    def _codex_reasoning_effort(get: Any) -> str:
        effort = get("codex_reasoning_effort", "medium") or "medium"
        return effort if effort in {"minimal", "low", "medium", "high", "xhigh"} else "medium"

    @staticmethod
    def _claude_effort(get: Any) -> str:
        effort = get("claude_effort", "medium") or "medium"
        return effort if effort in {"low", "medium", "high", "xhigh", "max"} else "medium"

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        if self.settings is not None and isinstance(patch, dict):
            for key, value in patch.items():
                if key in self.SETTINGS_KEYS:
                    self.settings.set(key, value)
        self._apply_settings()
        return self.get_settings()

    def _apply_settings(self) -> None:
        provider = self._chat_provider(self.settings.get) if self.settings is not None else "ollama"
        if hasattr(self.engine, "chat_backend"):
            self.engine.chat_backend = "lmstudio" if provider == "lmstudio" else "ollama"
        if self.settings is not None and hasattr(self.engine, "openai_base_url"):
            self.engine.openai_base_url = normalize_base_url(
                self.settings.get("lmstudio_base_url", DEFAULT_LM_STUDIO_BASE_URL)
            )
        if self.settings is not None and hasattr(self.engine, "openai_api_key"):
            self.engine.openai_api_key = self.settings.get("lmstudio_api_key", "") or ""
        if provider == "lmstudio" and self.settings is not None and hasattr(self.engine, "model"):
            self.engine.model = (
                self.settings.get("lmstudio_model", DEFAULT_LM_STUDIO_MODEL)
                or DEFAULT_LM_STUDIO_MODEL
            )
        elif self.settings is not None and hasattr(self.engine, "model"):
            selected = self.settings.get("selected_model", "") or ""
            if selected:
                self.engine.model = selected
        if self.settings is not None and hasattr(self.engine, "system_prompt_extra"):
            self.engine.system_prompt_extra = self.settings.get("custom_instructions", "") or ""
        if self.settings is not None and hasattr(self.engine, "utility_model"):
            self.engine.utility_model = (
                self.settings.get("utility_model", "") or ""
            ).strip() or None
        if hasattr(self.engine, "num_ctx"):
            value = (
                self.settings.get("context_window", "auto") if self.settings is not None else "auto"
            )
            self.engine.num_ctx = self._resolve_num_ctx(value)
        # Rebuild the web facade + indexers on next use so a changed SearXNG URL
        # / embedding model takes effect without a restart.
        self._web = None
        self._memory_indexer = None
        self._fact_store = None
        # The skill router caches the embedding model / semantic flag — drop it so
        # a changed setting takes effect on the next turn (the store self-reloads).
        self._skills_router = None

    def _resolve_num_ctx(self, value: Any) -> int | None:
        """Resolve the `context_window` setting: an explicit number wins; "auto"
        sizes the window from the hardware budget minus the active model's
        weights (probed once per process). None keeps the engine default."""
        if isinstance(value, (int, float)) and int(value) >= 2048:
            return int(value)
        text = str(value or "").strip().lower()
        if text.isdigit():
            return max(2048, int(text))
        if text != "auto":
            return None
        if self._auto_num_ctx is None:
            try:
                from lity.core.hardware import detect_hardware
                from lity.core.model_advisor import catalog_size_gb, recommend_num_ctx

                hardware = detect_hardware()
                model_size = catalog_size_gb(getattr(self.engine, "model", "")) or 5.0
                self._auto_num_ctx = recommend_num_ctx(
                    float(hardware.get("budget_gb") or 0), model_size
                )
            except Exception:
                self._auto_num_ctx = 0
        return self._auto_num_ctx or None

    def _apply_conversation_instructions(self) -> None:
        """Apply active conversation instructions (system prompt + temperature)."""
        base = ""
        if self.settings is not None:
            base = self.settings.get("custom_instructions", "") or ""
        instructions, temperature = "", None
        if hasattr(self.memory, "get_active_instructions"):
            data = self.memory.get_active_instructions()
            instructions = (data.get("instructions") or "").strip()
            temperature = data.get("temperature")
        character = self.get_active_character()
        character_instructions = ""
        if character is not None:
            name = str(character.get("name") or "").strip()
            body = str(character.get("instructions") or "").strip()
            if body:
                character_instructions = f"Personnage actif : {name}\n{body}" if name else body
        extra = "\n".join(
            part for part in (base, character_instructions, instructions) if part
        ).strip()
        if hasattr(self.engine, "system_prompt_extra"):
            self.engine.system_prompt_extra = extra
        if hasattr(self.engine, "temperature"):
            self.engine.temperature = (
                float(temperature) if isinstance(temperature, (int, float)) else None
            )

    def get_conversation_instructions(self) -> dict[str, Any]:
        if hasattr(self.memory, "get_active_instructions"):
            return self.memory.get_active_instructions()
        return {"instructions": "", "temperature": None}

    def set_conversation_instructions(
        self, instructions: str, temperature: float | None = None
    ) -> bool:
        if hasattr(self.memory, "set_active_instructions"):
            temp = float(temperature) if isinstance(temperature, (int, float)) else None
            return bool(self.memory.set_active_instructions(instructions or "", temp))
        return False

    # ------------------------------------------------------------- personnages
    def _ensure_characters(self) -> Any:
        if self._characters is None:
            from lity.services.characters import CharacterStore

            self._characters = CharacterStore(self.paths.characters_dir)
        return self._characters

    def _ensure_character_generator(self) -> Any:
        if (
            self._character_generator is None
            or getattr(self._character_generator, "image_manager", None) is not self.image_manager
        ):
            from lity.services.characters import CharacterImageGenerator

            self._character_generator = CharacterImageGenerator(
                self._ensure_characters(), self.image_manager
            )
        return self._character_generator

    def list_characters(self) -> dict[str, Any]:
        return {
            "characters": self._ensure_characters().list(),
            "active_character_id": self.get_active_character_id(),
            "active_character": self.get_active_character(),
        }

    def create_character(self, data: dict[str, Any]) -> dict[str, Any]:
        profile = self._ensure_characters().create(data if isinstance(data, dict) else {})
        return {"ok": True, "character": profile, **self.list_characters()}

    def update_character(self, character_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        profile = self._ensure_characters().update(
            character_id, patch if isinstance(patch, dict) else {}
        )
        if profile is None:
            return {"ok": False, "message": "Personnage introuvable.", **self.list_characters()}
        return {"ok": True, "character": profile, **self.list_characters()}

    def delete_character(self, character_id: str) -> dict[str, Any]:
        ok = self._ensure_characters().delete(character_id)
        if ok and self.get_active_character_id() == character_id:
            self.set_active_character("")
        return {"ok": ok, **self.list_characters()}

    def get_active_character_id(self) -> str:
        if hasattr(self.memory, "get_active_character_id"):
            return self.memory.get_active_character_id()
        return ""

    def get_active_character(self) -> dict[str, Any] | None:
        character_id = self.get_active_character_id()
        return self._ensure_characters().get(character_id) if character_id else None

    def set_active_character(self, character_id: str) -> dict[str, Any]:
        cleaned = str(character_id or "").strip()
        if cleaned and self._ensure_characters().get(cleaned) is None:
            return {
                "ok": False,
                "message": "Personnage introuvable.",
                "active_character": self.get_active_character(),
                **self.list_characters(),
            }
        ok = (
            bool(self.memory.set_active_character_id(cleaned))
            if hasattr(self.memory, "set_active_character_id")
            else False
        )
        return {
            "ok": ok,
            "active_character": self.get_active_character(),
            **self.list_characters(),
        }

    def generate_character_emotions(
        self, character_id: str, emotions: list[str] | None = None
    ) -> dict[str, Any]:
        result = self._ensure_character_generator().generate(character_id, emotions)
        return {**result, **self.list_characters(), "character": result.get("character")}

    def _embedding_model(self) -> str:
        if self.settings is not None:
            return self.settings.get("embedding_model", "nomic-embed-text") or "nomic-embed-text"
        return "nomic-embed-text"

    def _ensure_mcp(self) -> Any:
        """Build the MCP client manager once (None if the SDK/config is absent).

        Reads config/mcp.json (Claude-Desktop format) and connects to local MCP
        servers; their tools become agent tools. Degrades gracefully to None.
        """
        if not self._mcp_tried:
            self._mcp_tried = True
            from lity.services.mcp import build_mcp_manager

            self._mcp = build_mcp_manager(self.paths.mcp_config_file)
        return self._mcp

    # ------------------------------------------------------------- compétences
    def _ensure_skills(self) -> Any:
        """Build the skill store once (built-in + user skills). Degrades to None."""
        if not self._skills_tried:
            self._skills_tried = True
            from lity.services.skills import build_skill_store

            try:
                self._skills = build_skill_store(
                    self.paths.skills_dir, self.paths.builtin_skills_dir
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.info("Skills indisponibles : %s", exc)
                self._skills = None
        return self._skills

    def _skills_master_enabled(self) -> bool:
        return self.settings is None or bool(self.settings.get("skills_enabled", True))

    def _skills_disabled(self) -> list[str]:
        raw = self.settings.get("skills_disabled", []) if self.settings is not None else []
        return [str(item) for item in raw] if isinstance(raw, list) else []

    def _enabled_skills(self) -> list[Any]:
        store = self._ensure_skills()
        if store is None or not self._skills_master_enabled():
            return []
        disabled = set(self._skills_disabled())
        return [skill for skill in store.list() if skill.name not in disabled]

    def _skill_router(self) -> Any:
        if self._skills_router is None:
            from lity.services.skills import SkillRouter

            semantic = bool(self.settings.get("skills_semantic", False)) if self.settings else False
            embed = None
            if semantic and hasattr(self.engine, "embed"):
                model = (
                    (self.settings.get("embedding_model", "") or "nomic-embed-text")
                    if self.settings is not None
                    else "nomic-embed-text"
                )
                embed = lambda text: self.engine.embed(text, model_name=model)  # noqa: E731
            structured = None
            if hasattr(self.engine, "generate_structured"):
                # prefer_utility keeps the per-turn selection cheap (small model
                # when one is configured); it only fires when a candidate exists.
                structured = lambda prompt, schema: self.engine.generate_structured(  # noqa: E731
                    prompt, schema, think=False, prefer_utility=True
                )
            self._skills_router = SkillRouter(embed=embed, structured=structured, semantic=semantic)
        return self._skills_router

    def _last_user_text(self) -> str:
        for message in reversed(self.memory.get_context()):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def _apply_skills(self, user_input: str | None = None) -> None:
        """Route the current turn to a skill and set the engine's per-turn skills
        injection (Level-1 catalogue + an activated Level-2 body). A no-op when
        skills are off/empty; never blocks a turn on a routing failure."""
        if not hasattr(self.engine, "skills_prompt"):
            return
        skills = self._enabled_skills()
        if not skills:
            self.engine.skills_prompt = ""
            return
        text = user_input if user_input is not None else self._last_user_text()
        active = None
        if text and text.strip():
            try:
                selection = self._skill_router().select(text, skills)
                active = selection.skill if selection else None
            except Exception as exc:  # pragma: no cover - defensive
                logger.info("Sélection de compétence impossible : %s", exc)
        from lity.services.skills import build_skills_prompt

        self.engine.skills_prompt = build_skills_prompt(skills, active)

    def list_skills(self) -> dict[str, Any]:
        store = self._ensure_skills()
        disabled = set(self._skills_disabled())
        skills = store.list() if store is not None else []
        semantic = bool(self.settings.get("skills_semantic", False)) if self.settings else False
        return {
            "enabled": self._skills_master_enabled(),
            "semantic": semantic,
            "dir": str(self.paths.skills_dir),
            "skills": [
                {**skill.to_dict(), "enabled": skill.name not in disabled} for skill in skills
            ],
        }

    def toggle_skill(self, name: str, enabled: bool) -> dict[str, Any]:
        if self.settings is None:
            return {"ok": False, "message": "Réglages indisponibles."}
        from lity.services.skills import slugify

        slug = slugify(name)
        disabled = self._skills_disabled()
        if enabled:
            disabled = [item for item in disabled if item != slug]
        elif slug not in disabled:
            disabled = [*disabled, slug]
        self.settings.set("skills_disabled", disabled)
        return {"ok": True, "name": slug, "enabled": bool(enabled)}

    def create_skill(
        self,
        name: str,
        description: str,
        body: str,
        when_to_use: str = "",
        triggers: list[str] | None = None,
    ) -> dict[str, Any]:
        store = self._ensure_skills()
        if store is None:
            return {"ok": False, "message": "Compétences indisponibles."}
        ok, message, skill = store.create(
            name, description, body, when_to_use=when_to_use, triggers=triggers or []
        )
        result: dict[str, Any] = {"ok": ok, "message": message}
        if skill is not None:
            result["skill"] = {**skill.to_dict(), "enabled": True}
        return result

    def delete_skill(self, name: str) -> dict[str, Any]:
        store = self._ensure_skills()
        if store is None:
            return {"ok": False, "message": "Compétences indisponibles."}
        ok, message = store.delete(name)
        if ok and self.settings is not None:
            from lity.services.skills import slugify

            slug = slugify(name)
            self.settings.set(
                "skills_disabled", [item for item in self._skills_disabled() if item != slug]
            )
        return {"ok": ok, "message": message}

    def _ensure_web(self) -> dict[str, Any]:
        """Build the web search/fetch facade (SearXNG → DuckDuckGo → Wikipedia)."""
        if self._web is None:
            from lity.services.web import (
                DuckDuckGoProvider,
                PageFetcher,
                SearxngProvider,
                WebResearcher,
                WebSearcher,
                WikipediaProvider,
            )

            searxng_url = (
                self.settings.get("searxng_url", DEFAULT_SEARXNG_URL)
                if self.settings is not None
                else ""
            )
            providers: list[Any] = []
            if searxng_url:
                providers.append(SearxngProvider(searxng_url))
            providers.append(DuckDuckGoProvider())
            providers.append(WikipediaProvider(language="fr"))
            searcher = WebSearcher(providers)
            fetcher = PageFetcher()
            self._web = {
                "searcher": searcher,
                "fetcher": fetcher,
                "researcher": WebResearcher(searcher, fetcher),
            }
        return self._web

    def using_codex_provider(self) -> bool:
        if self.settings is None:
            return False
        return self.settings.get("chat_provider", "ollama") == "codex"

    def using_claude_provider(self) -> bool:
        if self.settings is None:
            return False
        return self.settings.get("chat_provider", "ollama") == "claude"

    def using_grok_provider(self) -> bool:
        if self.settings is None:
            return False
        return self.settings.get("chat_provider", "ollama") == "grok"

    def using_lmstudio_provider(self) -> bool:
        if self.settings is None:
            return False
        return self.settings.get("chat_provider", "ollama") == "lmstudio"

    def using_cli_provider(self) -> bool:
        """True for any CLI-backed chat provider (Codex, Claude or Grok), which all
        bypass the local Ollama engine, memory injection and the agent tool-loop."""
        return (
            self.using_codex_provider()
            or self.using_claude_provider()
            or self.using_grok_provider()
        )

    def lmstudio_models(self) -> dict[str, Any]:
        """OpenAI-compatible local model catalogue exposed by LM Studio."""
        if self.settings is not None and hasattr(self.engine, "openai_base_url"):
            self.engine.openai_base_url = normalize_base_url(
                self.settings.get("lmstudio_base_url", DEFAULT_LM_STUDIO_BASE_URL)
            )
        try:
            names = self.engine._client().list_models() if hasattr(self.engine, "_client") else []
        except Exception as exc:
            return {
                "ok": False,
                "models": [],
                "default_model": self.get_settings().get("lmstudio_model", DEFAULT_LM_STUDIO_MODEL),
                "base_url": self.get_settings().get(
                    "lmstudio_base_url", DEFAULT_LM_STUDIO_BASE_URL
                ),
                "recommended": LM_STUDIO_RECOMMENDED_MODELS,
                "message": str(exc),
            }
        model = self.get_settings().get("lmstudio_model", DEFAULT_LM_STUDIO_MODEL)
        default = model if model in names else (names[0] if names else model)
        return {
            "ok": True,
            "models": [{"slug": name, "display_name": name} for name in names],
            "default_model": default,
            "base_url": self.get_settings().get("lmstudio_base_url", DEFAULT_LM_STUDIO_BASE_URL),
            "recommended": LM_STUDIO_RECOMMENDED_MODELS,
            "message": "LM Studio connecté." if names else "Aucun modèle chargé dans LM Studio.",
        }

    def _ensure_codex_cli(self) -> Any:
        client = getattr(self, "_codex_cli", None)
        if client is None:
            from lity.services.codex_cli import CodexCliClient

            client = CodexCliClient()
            self._codex_cli = client
        return client

    def _ensure_claude_cli(self) -> Any:
        client = getattr(self, "_claude_cli", None)
        if client is None:
            from lity.services.claude_cli import ClaudeCliClient

            client = ClaudeCliClient()
            self._claude_cli = client
        return client

    def _ensure_grok_cli(self) -> Any:
        client = getattr(self, "_grok_cli", None)
        if client is None:
            from lity.services.grok_cli import GrokCliClient

            client = GrokCliClient()
            self._grok_cli = client
        return client

    def codex_status(self) -> dict[str, Any]:
        return self._ensure_codex_cli().status()

    def codex_login(self) -> dict[str, Any]:
        return self._ensure_codex_cli().start_login()

    def codex_models(self) -> dict[str, Any]:
        return self._ensure_codex_cli().model_catalog()

    def claude_status(self) -> dict[str, Any]:
        return self._ensure_claude_cli().status()

    def claude_login(self) -> dict[str, Any]:
        return self._ensure_claude_cli().start_login()

    def claude_models(self) -> dict[str, Any]:
        return self._ensure_claude_cli().model_catalog()

    def grok_status(self) -> dict[str, Any]:
        return self._ensure_grok_cli().status()

    def grok_login(self, *, device_auth: bool = False) -> dict[str, Any]:
        return self._ensure_grok_cli().start_login(device_auth=device_auth)

    def grok_models(self) -> dict[str, Any]:
        return self._ensure_grok_cli().model_catalog()

    def _grok_session_id(self) -> str:
        conversation_id = getattr(self, "active_conversation_id", "") or ""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(conversation_id)).strip("-")
        return f"lity-{safe}" if safe else ""

    def _cli_has_project_context(self) -> bool:
        return self.using_cli_provider() and bool(getattr(self.files, "working_dir", None))

    def _codex_prompt(self, user_input: str) -> str:
        parts = [
            "Tu es utilisé via Codex CLI connecté au compte ChatGPT.",
            (
                "Réponds en français, clairement. N'appelle pas Ollama, llama-server, "
                "ni aucun modèle local : quand Codex est sélectionné, Codex fait tout le "
                "raisonnement lui-même."
            ),
            (
                "L'application ne pré-injecte pas sa mémoire ni son RAG local dans ce mode. "
                "Si tu as besoin du projet, inspecte directement le dossier de travail "
                "avec tes outils Codex en lecture seule."
            ),
            (
                "Si tu as besoin du RAG ou de la mémoire locale, décide-le toi-même et lance "
                "une recherche lexicale sans modèle local avec : "
                '`uv run python -m lity.services.codex_rag --query "..."`. '
                "N'utilise cette commande que si elle est utile à la demande."
            ),
            (
                "Si tu apprends un fait durable utile à mémoriser, ajoute à la toute fin "
                "un bloc JSON [LITY_MEMORY]...[/LITY_MEMORY] avec categorie, cle et valeur. "
                "Catégories autorisées : user_profile, assistant_profile, long_term_facts. "
                "N'utilise ce bloc que pour une information stable et réellement pertinente."
            ),
        ]
        extra = self._codex_extra_instructions()
        if extra:
            parts.append("Instructions utilisateur :\n" + extra)
        parts.append("Message utilisateur :\n" + user_input)
        return "\n\n".join(parts)

    def _codex_extra_instructions(self) -> str:
        parts: list[str] = []
        if self.settings is not None:
            custom = (self.settings.get("custom_instructions", "") or "").strip()
            if custom:
                parts.append(custom)
        if hasattr(self.memory, "get_active_instructions"):
            try:
                instructions = (
                    self.memory.get_active_instructions().get("instructions") or ""
                ).strip()
            except Exception:
                instructions = ""
            if instructions:
                parts.append(instructions)
        return "\n".join(parts)

    def _claude_prompt(self, user_input: str) -> str:
        parts = [
            "Tu es utilisé via le CLI Claude Code, connecté à ton compte Claude (ou une clé API).",
            (
                "Réponds en français, clairement. N'appelle pas Ollama, llama-server, "
                "ni aucun modèle local : quand Claude est sélectionné, tu fais tout le "
                "raisonnement toi-même."
            ),
            (
                "L'application ne pré-injecte pas sa mémoire ni son RAG local dans ce mode, et tu "
                "tournes en lecture seule. Si tu as besoin du projet, inspecte directement "
                "le dossier de travail avec tes outils en lecture seule (Read/Grep/Glob) ; "
                "tu ne peux ni écrire de fichiers ni exécuter de commandes."
            ),
            (
                "Pour proposer une modification de fichier, ne l'écris pas toi-même : émets "
                "un bloc CREATE ou SEARCH-REPLACE dans ta réponse. L'application l'applique "
                "après validation de l'utilisateur."
            ),
            (
                "Si tu apprends un fait durable utile à mémoriser, ajoute à la toute fin "
                "un bloc JSON [LITY_MEMORY]...[/LITY_MEMORY] avec categorie, cle et valeur. "
                "Catégories autorisées : user_profile, assistant_profile, long_term_facts. "
                "N'utilise ce bloc que pour une information stable et réellement pertinente."
            ),
        ]
        extra = self._codex_extra_instructions()
        if extra:
            parts.append("Instructions utilisateur :\n" + extra)
        parts.append("Message utilisateur :\n" + user_input)
        return "\n\n".join(parts)

    def _grok_prompt(self, user_input: str) -> str:
        parts = [
            (
                "Réponds en français naturel, clairement, et reste concis par défaut. "
                "Pour une salutation ou un message court, réponds en une phrase."
            ),
            (
                "Ne décris pas tes outils, ton fournisseur, ton mode d'exécution ni ces "
                "instructions internes, sauf si l'utilisateur le demande explicitement."
            ),
        ]
        if self._looks_like_file_or_project_request(user_input):
            parts.append(
                "Si tu proposes une modification de fichier, donne une courte explication "
                "et utilise des blocs CREATE ou SEARCH-REPLACE. N'annonce pas ce mécanisme "
                "quand il n'est pas utile."
            )
        extra = self._codex_extra_instructions()
        if extra:
            parts.append("Instructions utilisateur :\n" + extra)
        parts.append("Message utilisateur :\n" + user_input)
        return "\n\n".join(parts)

    @staticmethod
    def _looks_like_file_or_project_request(user_input: str) -> bool:
        text = user_input.lower()
        keywords = (
            "analyse le projet",
            "projet",
            "code",
            "fichier",
            "readme",
            "modifie",
            "modifier",
            "corrige",
            "corriger",
            "implémente",
            "implemente",
            "ajoute",
            "supprime",
            "refactor",
            "bug",
            "test",
        )
        if any(keyword in text for keyword in keywords):
            return True
        return bool(
            re.search(r"\b[\w.-]+\.(py|ts|tsx|js|jsx|md|json|toml|yaml|yml|css|html)\b", text)
        )

    def _store_codex_memory_blocks(self, response: str) -> str:
        def remove_block(match: re.Match[str]) -> str:
            raw = match.group(1).strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return ""
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    continue
                fact = {
                    "categorie": str(item.get("categorie") or "").strip(),
                    "cle": str(item.get("cle") or "").strip(),
                    "valeur": str(item.get("valeur") or "").strip(),
                }
                if fact["categorie"] and fact["cle"] and fact["valeur"]:
                    try:
                        self.memory.process_extracted_fact(fact)
                    except Exception:
                        continue
            return ""

        return _CODEX_MEMORY_BLOCK_RE.sub(remove_block, response).strip()

    def _run_codex_turn(self, user_input: str, on_chunk: Any | None = None) -> Any:
        status = self.codex_status()
        if not status.get("available") or not status.get("authenticated"):
            return ErrorResult(
                message=(
                    status.get("message")
                    or "Codex n'est pas connecté. Lance `codex login` avec ton compte ChatGPT."
                )
            )
        settings = self.get_settings()
        result = self._ensure_codex_cli().run_prompt(
            self._codex_prompt(user_input),
            model=settings.get("codex_model", ""),
            reasoning_effort=settings.get("codex_reasoning_effort", "medium"),
            workdir=getattr(self.files, "working_dir", None),
        )
        if not result.get("ok"):
            return ErrorResult(message=result.get("message", "Codex n'a pas répondu."))
        self._record_cli_usage("codex", settings.get("codex_model", ""), result.get("usage"))
        response = self._store_codex_memory_blocks((result.get("content") or "").strip())
        if not response:
            return TextResult(content="Codex a terminé sans renvoyer de réponse.")
        if on_chunk is not None:
            on_chunk(response)
        self.memory.add_message("assistant", response)
        create_blocks = self.editor.parse_create_blocks(response)
        edit_blocks = self.editor.parse_search_replace_blocks(response)
        return AiResponseResult(
            content=response,
            create_blocks=create_blocks,
            edit_blocks=edit_blocks,
            system_notification=self._block_format_warning(response, create_blocks, edit_blocks),
        )

    def _run_cli_turn(
        self,
        user_input: str,
        on_chunk: Any | None = None,
        should_cancel: Any | None = None,
    ) -> Any:
        """Route a CLI-provider turn to the active backend (Codex, Claude or Grok)."""
        if self.using_claude_provider():
            return self._run_claude_turn(user_input, on_chunk=on_chunk)
        if self.using_grok_provider():
            return self._run_grok_turn(user_input, on_chunk=on_chunk, should_cancel=should_cancel)
        return self._run_codex_turn(user_input, on_chunk=on_chunk)

    def _run_claude_turn(self, user_input: str, on_chunk: Any | None = None) -> Any:
        status = self.claude_status()
        if not status.get("available") or not status.get("authenticated"):
            return ErrorResult(
                message=(
                    status.get("message")
                    or "Claude n'est pas connecté. Lance `claude` puis `/login` avec ton compte."
                )
            )
        settings = self.get_settings()
        result = self._ensure_claude_cli().run_prompt(
            self._claude_prompt(user_input),
            model=settings.get("claude_model", ""),
            reasoning_effort=settings.get("claude_effort", "medium"),
            workdir=getattr(self.files, "working_dir", None),
        )
        if not result.get("ok"):
            return ErrorResult(message=result.get("message", "Claude n'a pas répondu."))
        self._record_cli_usage("claude", settings.get("claude_model", ""), result.get("usage"))
        response = self._store_codex_memory_blocks((result.get("content") or "").strip())
        if not response:
            return TextResult(content="Claude a terminé sans renvoyer de réponse.")
        if on_chunk is not None:
            on_chunk(response)
        self.memory.add_message("assistant", response)
        create_blocks = self.editor.parse_create_blocks(response)
        edit_blocks = self.editor.parse_search_replace_blocks(response)
        return AiResponseResult(
            content=response,
            create_blocks=create_blocks,
            edit_blocks=edit_blocks,
            system_notification=self._block_format_warning(response, create_blocks, edit_blocks),
        )

    def _run_grok_turn(
        self,
        user_input: str,
        on_chunk: Any | None = None,
        should_cancel: Any | None = None,
    ) -> Any:
        status = self.grok_status()
        if not status.get("available") or not status.get("authenticated"):
            return ErrorResult(
                message=(
                    status.get("message")
                    or "Grok n'est pas connecté. Lance `grok login` ou exporte XAI_API_KEY."
                )
            )
        settings = self.get_settings()
        result = self._ensure_grok_cli().run_prompt(
            self._grok_prompt(user_input),
            model=settings.get("grok_model", ""),
            workdir=getattr(self.files, "working_dir", None),
            session_id=self._grok_session_id(),
            output_format="streaming-json",
            on_chunk=on_chunk,
            should_cancel=should_cancel,
        )
        if not result.get("ok"):
            return ErrorResult(message=result.get("message", "Grok n'a pas répondu."))
        self._record_cli_usage("grok", settings.get("grok_model", ""), result.get("usage"))
        response = self._store_codex_memory_blocks((result.get("content") or "").strip())
        if not response:
            return TextResult(content="Grok a terminé sans renvoyer de réponse.")
        if on_chunk is not None and not result.get("streamed"):
            on_chunk(response)
        self.memory.add_message("assistant", response)
        create_blocks = self.editor.parse_create_blocks(response)
        edit_blocks = self.editor.parse_search_replace_blocks(response)
        return AiResponseResult(
            content=response,
            create_blocks=create_blocks,
            edit_blocks=edit_blocks,
            system_notification=self._block_format_warning(response, create_blocks, edit_blocks),
        )

    @property
    def assistant_name(self) -> str:
        name = str(self.memory.assistant_profile.get("nom", "") or "").strip()
        if name.lower() in {"", "lity"}:
            return DEFAULT_ASSISTANT_NAME
        return name

    def start_image_session(self) -> dict[str, Any]:
        return self._ensure_image_manager().start_session()

    def poll_image_launch_status(self) -> dict[str, Any]:
        return self._ensure_image_manager().poll_launch_status()

    def is_image_session_active(self) -> bool:
        return bool(self.image_manager is not None and self.image_manager.is_active())

    def select_image_model(self, name: str) -> dict[str, Any]:
        """Persist which downloaded image model the local engine generates with."""
        return self._ensure_image_manager().select_image_model(name)

    def _handle_image_turn(self, user_input: str) -> Any:
        """Run an image-mode turn and persist a successful render into the active
        conversation, so it shows up in the sidebar and the image redisplays on
        reload (image mode otherwise never touches conversation history)."""
        result = self.image_manager.process_user_message(user_input, self.engine)
        if isinstance(result, dict) and result.get("type") == "image_generation_result":
            self.memory.add_message("user", user_input)
            # The image IS the message — store empty assistant text (no caption
            # noise) and attach the render so it survives a reload.
            self.memory.add_message("assistant", "")
            image_path = (result.get("content") or {}).get("image_path")
            data_url = _image_path_to_data_url(image_path)
            if data_url and hasattr(self.memory, "set_last_message_image"):
                self.memory.set_last_message_image(data_url)
        return result

    def start_video_session(self) -> dict[str, Any]:
        return self._ensure_video_manager().start_session()

    def poll_video_launch_status(self) -> dict[str, Any]:
        return self._ensure_video_manager().poll_launch_status()

    def is_video_session_active(self) -> bool:
        return bool(self.video_manager is not None and self.video_manager.is_active())

    def select_video_model(self, name: str) -> dict[str, Any]:
        """Persist which downloaded video model the local engine generates with."""
        return self._ensure_video_manager().select_video_model(name)

    def _handle_video_turn(self, user_input: str) -> Any:
        """Run a video-mode turn and persist a successful render into the active
        conversation, so it shows up in the sidebar and the clip redisplays on
        reload (video mode otherwise never touches conversation history)."""
        result = self.video_manager.process_user_message(user_input, self.engine)
        if isinstance(result, dict) and result.get("type") == "video_generation_result":
            self.memory.add_message("user", user_input)
            # The clip IS the message — store empty assistant text (no caption
            # noise) and attach the render so it survives a reload.
            self.memory.add_message("assistant", "")
            video_path = (result.get("content") or {}).get("video_path")
            data_url = _image_path_to_data_url(video_path)  # MIME-agnostic → video/mp4
            if data_url and hasattr(self.memory, "set_last_message_video"):
                self.memory.set_last_message_video(data_url)
        return result

    def shutdown(self) -> None:
        if self.image_manager is not None and hasattr(self.image_manager, "shutdown"):
            self.image_manager.shutdown()
        if self.video_manager is not None and hasattr(self.video_manager, "shutdown"):
            self.video_manager.shutdown()

    def sync_available_models(self, models: list[str]) -> str:
        if hasattr(self.engine, "select_available_model"):
            target_model = self.engine.select_available_model(models, self.engine.model)
        else:
            target_model = (
                models[0] if models and self.engine.model not in models else self.engine.model
            )

        self.router.model = target_model
        if self.settings is not None and models:
            self.settings.set("selected_model", target_model)
        return target_model

    def change_model(self, model_name: str) -> str:
        models = self.engine.get_installed_models()
        if models and hasattr(self.engine, "select_available_model"):
            target_model = self.engine.select_available_model(models, model_name)
        else:
            target_model = model_name
            self.engine.model = target_model
        self.router.model = target_model
        if self.settings is not None:
            self.settings.set("selected_model", target_model)
        if hasattr(self.memory, "set_conversation_model"):
            self.memory.set_conversation_model(target_model)
        return target_model

    def clear_history(self) -> None:
        self.memory.clear()

    def process_slash_command(self, msg: str) -> dict[str, Any] | None:
        clean_msg = msg.strip()
        if not clean_msg.startswith("/"):
            return None

        parts = clean_msg.split(" ")
        command = parts[0].lower()
        if command == "/model":
            if len(parts) > 1:
                model_name = parts[1].strip()
                target_model = self.change_model(model_name)
                return {
                    "handled": True,
                    "action": "change_model",
                    "message": f"Modèle commuté vers : {target_model}",
                }
            return {
                "handled": True,
                "action": "get_model",
                "message": f"Modèle actif : {self.engine.model}. Utilise /model <nom> pour changer.",
            }
        if command == "/clear":
            self.clear_history()
            return {"handled": True, "action": "clear_history", "message": "Historique effacé."}
        if command == "/openfile" and len(parts) > 1:
            success, response = self.files.load_file(parts[1])
            return {"handled": True, "action": "open_file", "success": success, "message": response}
        if command == "/workdir" and len(parts) > 1:
            success, response = self.files.set_working_dir(" ".join(parts[1:]))
            return {
                "handled": True,
                "action": "set_working_dir",
                "success": success,
                "message": response,
            }
        if command == "/listfiles":
            return {"handled": True, "action": "list_files", "message": self.files.list_files()}
        if command == "/closefile":
            target = parts[1] if len(parts) > 1 else None
            success, response = self.files.close_file(target)
            return {
                "handled": True,
                "action": "close_file",
                "success": success,
                "message": response,
            }
        if command == "/reloadfile":
            target = parts[1] if len(parts) > 1 else self.files.current_file_path
            if not target:
                return {
                    "handled": True,
                    "action": "reload_file",
                    "success": False,
                    "message": "Aucun fichier.",
                }
            success, response = self.files.load_file(target)
            return {
                "handled": True,
                "action": "reload_file",
                "success": success,
                "message": f"{response} (Rechargé)" if success else response,
            }
        if command == "/help":
            return {"handled": True, "action": "help", "message": self.help_text()}
        if command == "/quit":
            return {"handled": True, "action": "quit", "message": "Au revoir !"}
        return {"handled": True, "action": "unknown", "message": f"Commande inconnue : {command}"}

    def process_user_message_sync(self, user_input: str) -> Any:
        if self.is_video_session_active():
            return self._handle_video_turn(user_input)
        if self.is_image_session_active():
            return self._handle_image_turn(user_input)

        if self.using_cli_provider():
            self.memory.add_message("user", user_input)
            return self._run_cli_turn(user_input)

        self._trigger_background_memory_task(user_input)
        self._apply_conversation_instructions()

        intent_result = self.router.process_intent(user_input, self.files)
        if intent_result.get("handled"):
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", intent_result.get("system_context", ""))
            return IntentHandledResult(
                action=intent_result.get("action"),
                message=intent_result.get("message", ""),
                system_context=intent_result.get("system_context", ""),
            )

        has_file_context = bool(self.files.loaded_files) or self._cli_has_project_context()
        if not has_file_context and requires_file_context(user_input):
            return ErrorResult(
                message="Aucun fichier n'est chargé. Définis un répertoire puis ouvre un fichier."
            )

        self.memory.add_message("user", user_input)
        self._apply_skills(user_input)
        response = self.engine.get_response(
            self.memory.get_context(),
            user_summary=self.memory.get_user_info_summary(),
            assistant_summary=self.memory.get_assistant_info_summary(),
            assistant_name=self.assistant_name,
            files_context=self._compose_injected_context(),
            think=self._think_for_turn(),
        )

        if not response:
            response = "Je suis là, mais Ollama n'a pas renvoyé de texte."
            return TextResult(content=response)

        self.memory.add_message("assistant", response)
        self._trigger_background_summary()
        create_blocks = self.editor.parse_create_blocks(response)
        edit_blocks = self.editor.parse_search_replace_blocks(response)
        notification = (
            intent_result.get("message") if intent_result.get("action") == "load_context" else None
        )
        return AiResponseResult(
            content=response,
            create_blocks=create_blocks,
            edit_blocks=edit_blocks,
            system_notification=notification
            or self._block_format_warning(response, create_blocks, edit_blocks),
        )

    def _web_answer_grader(self) -> Any:
        """A topic-agnostic grader (question, answer) → {"answered": bool} used in
        web mode to detect a hedge/non-answer and push the agent to keep
        researching. None when the engine has no structured-generation primitive
        or the gate is disabled via the ``web_answer_gate`` setting. Routed to
        the small utility model — it's a cheap verdict, not deliberation."""
        if not hasattr(self.engine, "generate_structured"):
            return None
        if self.settings is not None and not self.settings.get("web_answer_gate", True):
            return None
        from lity.services.ai.prompts import (
            ANSWER_SUFFICIENCY_PROMPT,
            ANSWER_SUFFICIENCY_SCHEMA,
        )

        def grade(question: str, answer: str) -> dict[str, Any] | None:
            prompt = ANSWER_SUFFICIENCY_PROMPT.format(
                question=str(question)[:800], answer=str(answer)[:2000]
            )
            return self.engine.generate_structured(
                prompt, ANSWER_SUFFICIENCY_SCHEMA, think=False, prefer_utility=True
            )

        return grade

    def _think_for_turn(self) -> bool | None:
        """Think-routing for the chat path.

        On a reasoning model, skip the <think> phase for trivial chit-chat
        (returns ``False`` → faster, no wasted reasoning) and keep it otherwise
        (returns ``None`` = model default, which preserves the inline
        ``<think>`` block the UI renders as collapsible). NEVER forces
        ``think=True``, so the reasoning block is never lost. No-op on
        non-reasoning models and when the ``think_routing`` setting is off.
        """
        from lity.services.ai._engine_common import is_thinking_model, wants_thinking

        if self.settings is not None and not self.settings.get("think_routing", True):
            return None
        if not is_thinking_model(getattr(self.engine, "model", "")):
            return None
        context = self.memory.get_context()
        last_user = next(
            (msg.get("content") for msg in reversed(context) if msg.get("role") == "user"), ""
        )
        return None if wants_thinking(last_user or "") else False

    def process_user_message_stream(
        self,
        user_input: str,
        on_chunk: Any,
        should_cancel: Any | None = None,
        images: list[str] | None = None,
    ) -> Any:
        if self.is_video_session_active():
            return self._handle_video_turn(user_input)
        if self.is_image_session_active():
            return self._handle_image_turn(user_input)

        if self.using_cli_provider():
            self.memory.add_message("user", user_input)
            return self._run_cli_turn(user_input, on_chunk=on_chunk, should_cancel=should_cancel)

        self._trigger_background_memory_task(user_input)
        self._apply_conversation_instructions()

        intent_result = self.router.process_intent(user_input, self.files)
        if intent_result.get("handled"):
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", intent_result.get("system_context", ""))
            return IntentHandledResult(
                action=intent_result.get("action"),
                message=intent_result.get("message", ""),
                system_context=intent_result.get("system_context", ""),
            )

        # An attached image IS the context — never block "analyse cette image"
        # on the workspace-file gate (it only guards "analyse CE fichier" with
        # nothing loaded).
        has_file_context = bool(self.files.loaded_files) or self._cli_has_project_context()
        if not images and not has_file_context and requires_file_context(user_input):
            return ErrorResult(
                message="Aucun fichier n'est chargé. Définis un répertoire puis ouvre un fichier."
            )

        self.memory.add_message("user", user_input, images=images)
        system_notification = (
            intent_result.get("message") if intent_result.get("action") == "load_context" else None
        )
        return self._stream_current_context(
            on_chunk, should_cancel, images=images, system_notification=system_notification
        )

    def _stream_current_context(
        self,
        on_chunk: Any,
        should_cancel: Any | None = None,
        images: list[str] | None = None,
        system_notification: str | None = None,
    ) -> Any:
        """Generate an assistant turn from the current memory context (no new user message)."""
        self._apply_conversation_instructions()  # covers regenerate/edit which skip the message paths
        self._apply_skills()  # route on the last user message (covers regenerate/edit)
        files_context = self._compose_injected_context(include_rag=True)
        # When the turn carries an image — freshly attached OR stored on the last
        # user message (regenerate/edit) — make sure a vision-capable model reads
        # it. A text-only model silently drops the image, so route to an installed
        # vision model when possible, otherwise tell the user why it can't see it.
        model_override: str | None = None
        if images or self._last_user_has_images():
            model_override, image_note = self._resolve_vision_routing()
            if image_note and not system_notification:
                system_notification = image_note
        response_parts: list[str] = []
        think = self._think_for_turn()
        if hasattr(self.engine, "stream_response"):
            for chunk in self.engine.stream_response(
                self.memory.get_context(),
                user_summary=self.memory.get_user_info_summary(),
                assistant_summary=self.memory.get_assistant_info_summary(),
                assistant_name=self.assistant_name,
                files_context=files_context,
                model_name=model_override,
                images=images,
                think=think,
            ):
                if should_cancel is not None and should_cancel():
                    break
                response_parts.append(chunk)
                on_chunk(chunk)
            response = "".join(response_parts).strip()
        else:
            response = self.engine.get_response(
                self.memory.get_context(),
                user_summary=self.memory.get_user_info_summary(),
                assistant_summary=self.memory.get_assistant_info_summary(),
                assistant_name=self.assistant_name,
                files_context=files_context,
                model_name=model_override,
                images=images,
                think=think,
            )

        if not response:
            return TextResult(content="Je suis là, mais Ollama n'a pas renvoyé de texte.")

        self.memory.add_message("assistant", response)
        self._trigger_background_summary()
        create_blocks = self.editor.parse_create_blocks(response)
        edit_blocks = self.editor.parse_search_replace_blocks(response)
        return AiResponseResult(
            content=response,
            create_blocks=create_blocks,
            edit_blocks=edit_blocks,
            system_notification=system_notification
            or self._block_format_warning(response, create_blocks, edit_blocks),
        )

    # --------------------------------------------------------- vision routing
    def _last_user_has_images(self) -> bool:
        """True when the most recent user turn carries a persisted image."""
        try:
            context = self.memory.get_context()
        except Exception:
            return False
        for message in reversed(context):
            if message.get("role") == "user":
                return bool(message.get("images"))
        return False

    def _model_sees_images(self, name: str) -> bool:
        """Whether a model can read images. Ollama's reported capabilities win
        (authoritative — works for any model incl. gemma4 and custom quants);
        the name heuristic is only a fallback when Ollama can't be reached."""
        from lity.core.model_catalog import is_vision_model

        checker = getattr(self.engine, "supports_vision", None)
        if callable(checker):
            try:
                verdict = checker(name)
            except Exception:
                verdict = None
            if verdict is not None:
                return bool(verdict)
        return is_vision_model(name)

    def _installed_vision_model(self) -> str | None:
        """First installed multimodal model, or None when the user has none."""
        getter = getattr(self.engine, "get_installed_models", None)
        if not callable(getter):
            return None
        try:
            installed = getter() or []
        except Exception:
            return None
        return next((name for name in installed if self._model_sees_images(name)), None)

    def _resolve_vision_routing(self) -> tuple[str | None, str | None]:
        """Pick the model that should read an attached image (and a user note).

        Returns ``(model_override, note)``. A vision-capable active model needs
        no change. Otherwise the turn is routed to an installed vision model when
        one exists; if none is installed, no override is returned and the note
        explains why the image can't be analysed — never silently dropped."""
        current = getattr(self.engine, "model", "") or ""
        if self._model_sees_images(current):
            return None, None
        vision = self._installed_vision_model()
        if vision and vision != current:
            return vision, (
                f"🔍 J'analyse l'image avec « {vision} » — le modèle « {current} » "
                "ne gère pas les images."
            )
        return None, (
            f"⚠️ Le modèle « {current} » ne peut pas voir les images. Installe un "
            "modèle multimodal (par ex. llava, llama3.2-vision, moondream) puis "
            "sélectionne-le pour que je puisse analyser tes images."
        )

    # --------------------------------------------------------- long-term memory
    def get_memory(self) -> dict[str, Any]:
        if hasattr(self.memory, "get_memory"):
            return self.memory.get_memory()
        return {"user_profile": {}, "assistant_profile": {}, "facts": {}}

    def update_memory(self, category: str, key: str, value: str) -> dict[str, Any]:
        key = (key or "").strip()
        if key:
            if category == "user_profile" and hasattr(self.memory, "update_user_profile"):
                self.memory.update_user_profile(key, value)
            elif category == "assistant_profile" and hasattr(
                self.memory, "update_assistant_profile"
            ):
                self.memory.update_assistant_profile(key, value)
            elif category == "facts" and hasattr(self.memory, "set_fact"):
                self.memory.set_fact(key, value)
        return self.get_memory()

    def delete_memory(self, category: str, key: str) -> dict[str, Any]:
        method = {
            "user_profile": "delete_user_profile",
            "assistant_profile": "delete_assistant_profile",
            "facts": "delete_fact",
        }.get(category)
        if method and hasattr(self.memory, method):
            getattr(self.memory, method)(key)
        return self.get_memory()

    def clear_memory(self) -> dict[str, Any]:
        if hasattr(self.memory, "clear_all_memory"):
            self.memory.clear_all_memory()
        return self.get_memory()

    # --------------------------------------------------------------- git

    def regenerate_active(self, on_chunk: Any, should_cancel: Any | None = None) -> Any:
        """Drop the last assistant turn and regenerate on the existing context."""
        if hasattr(self.memory, "drop_last_assistant"):
            self.memory.drop_last_assistant()
        if self.using_cli_provider():
            last_user = next(
                (
                    msg.get("content", "")
                    for msg in reversed(self.memory.get_context())
                    if msg.get("role") == "user"
                ),
                "",
            )
            return self._run_cli_turn(
                str(last_user), on_chunk=on_chunk, should_cancel=should_cancel
            )
        return self._stream_current_context(on_chunk, should_cancel)

    def edit_and_regenerate(
        self, new_text: str, on_chunk: Any, should_cancel: Any | None = None
    ) -> Any:
        """Replace the last user message and regenerate the assistant turn."""
        if hasattr(self.memory, "drop_last_assistant"):
            self.memory.drop_last_assistant()
        if hasattr(self.memory, "set_last_user_content"):
            self.memory.set_last_user_content(new_text)
        if self.using_cli_provider():
            return self._run_cli_turn(new_text, on_chunk=on_chunk, should_cancel=should_cancel)
        return self._stream_current_context(on_chunk, should_cancel)

    def process_user_message_agent(
        self,
        user_input: str,
        on_event: Any,
        should_cancel: Any | None = None,
        allow_commands: bool = False,
        allow_write: bool = False,
        allow_web: bool = False,
        confirm: Any | None = None,
    ) -> Any:
        from lity.services.ai.agent import DEFAULT_MAX_STEPS, AgentLoop
        from lity.services.ai.prompts import (
            AGENT_TOOL_GUIDANCE,
            AGENT_WEB_GUIDANCE,
            AGENT_YOLO_GUIDANCE,
        )

        if self.is_video_session_active():
            return self._handle_video_turn(user_input)
        if self.is_image_session_active():
            return self._handle_image_turn(user_input)

        if self.using_cli_provider():
            return self.process_user_message_stream(user_input, on_event, should_cancel)

        builder = getattr(self.engine, "_build_messages", None)
        if not callable(builder) or not hasattr(self.engine, "chat_with_tools"):
            # Engine without tool support: fall back to the streaming chat path.
            return self.process_user_message_stream(user_input, on_event, should_cancel)

        # The workspace tools (list_files/read_file/search and the write/command
        # tools) are only useful when there is something to inspect or act on.
        # Without a working directory or a loaded file — and outside YOLO/command/
        # web modes — skip the tool-loop entirely and answer as a normal chat, so
        # a greeting or general question never triggers a spurious list_files call.
        workspace_ready = bool(getattr(self.files, "working_dir", None)) or bool(
            getattr(self.files, "loaded_files", None)
        )
        if not (workspace_ready or allow_write or allow_commands or allow_web):
            return self.process_user_message_stream(user_input, on_event, should_cancel)

        self._trigger_background_memory_task(user_input)
        self._apply_conversation_instructions()
        self.memory.add_message("user", user_input)
        self._apply_skills(user_input)

        messages = builder(
            self.memory.get_context(),
            user_summary=self.memory.get_user_info_summary(),
            assistant_summary=self.memory.get_assistant_info_summary(),
            assistant_name=self.assistant_name,
            files_context=self._compose_injected_context(),
        )
        if messages and messages[0].get("role") == "system":
            # The model needs today's date to interpret "latest / yesterday /
            # this year" correctly — in every agent mode, not just web search.
            from datetime import date

            messages[0]["content"] += "\n" + AGENT_TOOL_GUIDANCE
            messages[0]["content"] += f"\n\nDate du jour : {date.today().isoformat()}."
            if allow_web:
                messages[0]["content"] += "\n" + AGENT_WEB_GUIDANCE
                # Additive steer (never hides the tool): if the question looks
                # time-sensitive, nudge the model to search instead of answering
                # from stale memory. The model still decides for everything else.
                from lity.services.web.router import WEB_NUDGE, looks_time_sensitive

                if looks_time_sensitive(user_input):
                    messages[0]["content"] += "\n" + WEB_NUDGE
            if allow_write:
                messages[0]["content"] += "\n" + AGENT_YOLO_GUIDANCE

        # Explicit plan phase: only for genuinely multi-step modes (YOLO / shell),
        # and only when the model itself judges the task complex — so a one-shot
        # edit never pays the extra call. Keeps a small local model on track
        # without the latency tax on simple tasks. Off via the `agent_planning`
        # setting.
        plan: list[str] = []
        plan_on = self.settings is None or self.settings.get("agent_planning", True)
        if (
            # Planning is only worth a call when there is something to act on:
            # commands to run, or file writes WITH a workspace to write into.
            # YOLO on a workspaceless chit-chat turn has nothing to plan.
            (allow_commands or (allow_write and workspace_ready))
            and plan_on
            and hasattr(self.engine, "generate_structured")
            and messages
            and messages[0].get("role") == "system"
        ):
            from lity.services.ai.planner import build_plan, format_plan

            plan = build_plan(
                lambda prompt, schema: self.engine.generate_structured(prompt, schema, think=False),
                user_input,
            )
            if plan:
                messages[0]["content"] += "\n" + format_plan(plan)

        # Step budget by mode: editing/commands (YOLO) need the most room; web
        # research needs search → read → reformulate → answer; plain agent turns
        # stay tight so small local models don't wander past 2-3 useful steps.
        if allow_write or allow_commands:
            max_steps = 12
        elif allow_web:
            max_steps = 8
        else:
            max_steps = DEFAULT_MAX_STEPS

        get = self.settings.get if self.settings is not None else lambda key, default=None: default
        verify_command = (get("verify_command", "") or "").strip()

        def loop_factory(**overrides: Any) -> AgentLoop:
            options: dict[str, Any] = dict(
                allow_commands=allow_commands,
                allow_write=allow_write,
                # File tools only when there is a workspace, so chit-chat never
                # gets tempted into list_files on an empty project.
                allow_files=workspace_ready,
                editor=self.editor,
                confirm=confirm,
                web=self._ensure_web() if allow_web else None,
                retrieval=self._agent_retrieval(),
                mcp=self._ensure_mcp(),
                max_steps=max_steps,
                verify_command=verify_command if allow_write else None,
                restrict_commands=bool(get("yolo_command_allowlist", False)),
                # Web mode: a hedge/non-answer gets pushed back so the model
                # keeps researching instead of giving up after one source.
                answer_grader=self._web_answer_grader() if allow_web else None,
            )
            options.update(overrides)
            return AgentLoop(self.engine, self.files, **options)

        # Long plans run as SUB-TASKS with a fresh context each (a small local
        # model cannot hold a 5-step task in one 16k window without drifting);
        # short plans stay in one loop, re-anchored periodically.
        orchestrate = (
            len(plan) >= 4
            and (allow_write or allow_commands)
            and bool(get("agent_orchestration", True))
        )
        if orchestrate:
            from lity.services.ai.orchestrator import TaskOrchestrator

            orchestrator = TaskOrchestrator(loop_factory)
            response, receipts = orchestrator.run(
                goal=user_input,
                plan=plan,
                base_messages=messages,
                on_event=on_event,
                should_cancel=should_cancel,
            )
        else:
            loop = loop_factory(plan=plan)
            response = loop.run(messages, on_event=on_event, should_cancel=should_cancel)
            receipts = loop.receipts_summary()

        # Provenance + grounding verdict (anti-hallucination): surface what tools
        # actually ran and whether the answer is backed by a successful one.
        if receipts:
            on_event("receipts", receipts)
        self._reindex_after_writes(receipts)

        if not response:
            return TextResult(content="Je suis là, mais je n'ai pas pu produire de réponse.")

        self.memory.add_message("assistant", response)
        self._trigger_background_summary()
        create_blocks = self.editor.parse_create_blocks(response)
        edit_blocks = self.editor.parse_search_replace_blocks(response)
        return AiResponseResult(
            content=response,
            create_blocks=create_blocks,
            edit_blocks=edit_blocks,
            system_notification=self._block_format_warning(response, create_blocks, edit_blocks),
        )

    def _block_format_warning(
        self,
        response: str,
        create_blocks: list[dict[str, str]],
        edit_blocks: list[dict[str, str]],
    ) -> str | None:
        """Surface the previously SILENT failure where the answer clearly meant
        to propose file blocks but none parsed (format drift)."""
        if create_blocks or edit_blocks:
            return None
        detector = getattr(self.editor, "detect_malformed_blocks", None)
        if callable(detector) and detector(response):
            return (
                "La réponse semble proposer des fichiers, mais les blocs n'ont pas pu "
                "être lus (format invalide). Demande une reformulation en blocs "
                "FILE / CREATE / SEARCH-REPLACE."
            )
        return None

    def apply_create_block(self, block: dict[str, str]) -> tuple[bool, str]:
        success, message = self.editor.create_file(
            block["file_path"],
            block["content"],
            working_dir=self.files.working_dir,
        )
        if success:
            if hasattr(self.files, "refresh_files"):
                self.files.refresh_files()
            if hasattr(self.files, "refresh_loaded_file"):
                self.files.refresh_loaded_file(block["file_path"])
        return success, message

    def changes_count(self) -> int:
        history = getattr(self.editor, "history", None)
        return history.count() if history is not None else 0

    def undo_last_change(self) -> dict[str, Any]:
        history = getattr(self.editor, "history", None)
        if history is None or not history.can_undo():
            return {"ok": False, "message": "Aucun changement à annuler."}
        result = history.undo_last()
        if result.get("ok") and hasattr(self.files, "refresh_files"):
            self.files.refresh_files()
        return result

    def apply_edit_block(self, block: dict[str, str]) -> tuple[bool, str]:
        file_path = block["file_path"]
        if self.files.working_dir and not Path(file_path).is_absolute():
            file_path = str((self.files.working_dir / file_path).resolve())
        success, message = self.editor.apply_edit(
            file_path,
            block["search_content"],
            block["replace_content"],
            working_dir=self.files.working_dir,
        )
        if success and hasattr(self.files, "refresh_loaded_file"):
            self.files.refresh_loaded_file(file_path)
        return success, message

    def help_text(self) -> str:
        return """COMMANDES DISPONIBLES :
/model <nom>       : Changer le modèle Ollama
/clear             : Effacer l'historique court terme
/workdir <chemin>  : Définir le répertoire de travail
/openfile <chemin> : Charger un fichier texte dans le contexte
/listfiles         : Voir les fichiers chargés ou disponibles
/closefile [nom]   : Fermer un fichier
/reloadfile [nom]  : Recharger le fichier actif
/help              : Afficher cette aide
/quit              : Quitter l'application en mode console"""

    def _ensure_image_manager(self) -> Any:
        if self.image_manager is None:
            self.services.with_image_manager(self.paths)
            self.image_manager = self.services.image_manager
        return self.image_manager

    def _ensure_video_manager(self) -> Any:
        if self.video_manager is None:
            self.services.with_video_manager(self.paths)
            self.video_manager = self.services.video_manager
        return self.video_manager


def should_extract_fact(user_msg: str) -> bool:
    text = user_msg.strip().lower()
    if len(text) < 12:
        return False
    memory_markers = (
        "je m'appelle",
        "mon nom",
        "mon prénom",
        "j'aime",
        "je n'aime",
        "je préfère",
        "je prefere",
        "souviens",
        "rappelle",
        "ma préférence",
        "mes préférences",
        "mon projet",
        "ma langue",
    )
    return any(marker in text for marker in memory_markers)


def requires_file_context(user_msg: str) -> bool:
    text = user_msg.strip().lower()
    explicit_markers = (
        "ce fichier",
        "le fichier",
        "du fichier",
        "des fichiers",
        "mon fichier",
        "fichier chargé",
        "fichier ouvert",
        "ce code",
        "le code",
        "du code",
        "contenu du",
        "contenu de",
        "analyse ce",
        "analyse le",
        "analyse ma",
        "vois ce",
        "vois le",
    )
    return any(marker in text for marker in explicit_markers)


def _image_path_to_data_url(image_path: Any) -> str:
    """Encode a generated image file as a ``data:`` URL for persistence/display.

    Returns "" when the path is missing or unreadable, so callers can skip the
    attachment without failing the turn."""
    if not image_path:
        return ""
    import base64
    import mimetypes

    path = Path(str(image_path))
    if not path.is_file():
        return ""
    try:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""
