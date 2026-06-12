from __future__ import annotations

import logging
import time
from typing import Any

from lity.services.ai._engine_admin import ModelAdminMixin
from lity.services.ai._engine_common import (
    GENERATION_LOCK,
    _clean_title,
    _message_content,
    _normalize_fact,
    _normalize_tool_calls,
    _sampling_for,
    _stats,
    _strip_data_url,
    budget_injected_context,
)
from lity.services.ai._engine_tasks import AuxTasksMixin
from lity.services.ai.openai_compatible import (
    DEFAULT_LM_STUDIO_BASE_URL,
    OpenAICompatibleClient,
    openai_message_content,
    openai_stats,
    openai_tool_calls,
)
from lity.services.ai.prompts import (
    DEFAULT_ASSISTANT_NAME,
    DEFAULT_MODEL_NAME,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Helpers live in _engine_common now; re-exported here for backward-compatible
# imports (and tests). Listed in __all__ so the re-exports aren't flagged unused.
__all__ = ["AIEngine", "_clean_title", "_normalize_fact", "_stats"]


class AIEngine(AuxTasksMixin, ModelAdminMixin):
    """Ollama chat engine: streamed/non-streamed chat, tool-calling, embeddings.

    Auxiliary one-shot LLM tasks (summary/title/fact) come from
    :class:`AuxTasksMixin`; model administration (list/pull/delete/info/health)
    from :class:`ModelAdminMixin`. This keeps each concern in its own module
    while the public ``AIEngine`` surface stays unchanged.
    """

    def __init__(self, model: str = DEFAULT_MODEL_NAME, keep_alive: str = "10m"):
        self.model = model
        self.keep_alive = keep_alive
        self.chat_backend = "ollama"
        self.openai_base_url = DEFAULT_LM_STUDIO_BASE_URL
        self.openai_api_key = ""
        self.openai_client: Any | None = None
        self.system_prompt = SYSTEM_PROMPT
        self.system_prompt_extra = ""
        # Per-turn "Compétences" injection (Level-1 catalogue + an activated
        # skill body), set by the controller's skill router before each turn.
        # Kept separate from system_prompt_extra so custom instructions and
        # skills never clobber each other.
        self.skills_prompt = ""
        self.last_stats: dict[str, Any] = {}  # eval metrics from the latest turn
        self._ctx_cache: dict[str, int] = {}  # model → context length (cached)
        self.temperature: float | None = None  # per-conversation override
        # Hardware-aware context window override (None → DEFAULT_NUM_CTX). Set by
        # the controller from the `context_window` setting / hardware probe.
        self.num_ctx: int | None = None
        # Optional small model for auxiliary one-shot tasks (title, summary,
        # fact extraction, retrieval grading). Keeping a 1-2B model
        # resident for these avoids paying the big model's latency — and avoids
        # swapping the big model out — on every background call.
        self.utility_model: str | None = None

    def _uses_openai_compatible(self) -> bool:
        return self.chat_backend == "lmstudio"

    def _client(self) -> Any:
        if self.openai_client is not None:
            return self.openai_client
        return OpenAICompatibleClient(self.openai_base_url, self.openai_api_key)

    def _openai_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
        response_format: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client().chat(
            model=model_name or self.model,
            messages=messages,
            tools=tools,
            stream=False,
            response_format=response_format,
            options=options or self._options(),
        )

    def _options(self, *, thinking: bool = False) -> dict[str, Any]:
        # Family-aware sampling + num_ctx (fixes the silent 4096 truncation that
        # broke agent loops). The conversation temperature override, if set, wins.
        params = _sampling_for(self.model, thinking=thinking)
        if self.num_ctx:
            params["num_ctx"] = int(self.num_ctx)
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params

    def effective_num_ctx(self) -> int:
        from lity.services.ai._engine_common import DEFAULT_NUM_CTX

        return int(self.num_ctx) if self.num_ctx else DEFAULT_NUM_CTX

    def _check_context_pressure(self) -> None:
        """REAL token counts (Ollama's own prompt_eval_count) vs the window.

        The char-based budgets are estimates; this is the ground truth after the
        fact. Beyond ~90% the next turn will be silently truncated from the
        head — exactly the failure the budgets exist to prevent — so it gets
        logged loudly and exposed in last_stats for the UI health surface."""
        used = int(self.last_stats.get("context_used") or 0)
        limit = self.effective_num_ctx()
        pressure = round(used / limit, 2) if limit else 0.0
        self.last_stats["context_pressure"] = pressure
        if pressure >= 0.9:
            logger.warning(
                "Fenêtre de contexte presque pleine : %s/%s tokens (%.0f%%) — "
                "les prochains tours risquent une troncature silencieuse.",
                used,
                limit,
                pressure * 100,
            )

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        """Single non-streamed turn that may return tool calls.

        Returns ``{"content": str|None, "tool_calls": [{"name", "arguments"}]}``.
        Degrades to a plain answer (no tool_calls) for models without tool support.
        ``think=False`` skips a reasoning model's <think> phase on tool-decision
        turns (much faster per step on a small local model).
        """
        if self._uses_openai_compatible():
            try:
                started = time.monotonic()
                with GENERATION_LOCK:
                    response = self._openai_chat(
                        messages,
                        tools=tools or None,
                        model_name=model_name,
                        options=self._options(thinking=bool(think)),
                    )
                content = openai_message_content(response).strip() or None
                self.last_stats = openai_stats(response, started)
                self._check_context_pressure()
                return {"content": content, "tool_calls": openai_tool_calls(response)}
            except Exception as exc:
                logger.warning("LM Studio tool chat failed: %s", exc)
                return {"content": None, "tool_calls": [], "error": str(exc)}
        try:
            import ollama

            kwargs: dict[str, Any] = {
                "model": model_name or self.model,
                "messages": messages,
                "tools": tools or None,
                "stream": False,
                "keep_alive": keep_alive or self.keep_alive,
                "options": self._options(thinking=bool(think)),
            }
            if think is not None:
                kwargs["think"] = think
            with GENERATION_LOCK:
                response = ollama.chat(**kwargs)
            message = (
                response.get("message")
                if isinstance(response, dict)
                else getattr(response, "message", None)
            )
            content = _message_content(response).strip() or None
            self.last_stats = _stats(response)
            self._check_context_pressure()
            return {"content": content, "tool_calls": _normalize_tool_calls(message)}
        except Exception as exc:
            logger.warning("Ollama tool chat failed: %s", exc)
            return {"content": None, "tool_calls": [], "error": str(exc)}

    def embed(self, text: str, model_name: str | None = None) -> list[float] | None:
        """Return an embedding vector for ``text`` via Ollama, or None on failure."""
        if self._uses_openai_compatible():
            try:
                return self._client().embed(text, model_name or self.utility_model or self.model)
            except Exception as exc:
                logger.info("LM Studio embedding failed: %s", exc)
                return None
        try:
            import ollama

            with GENERATION_LOCK:
                response = ollama.embeddings(model=model_name or "nomic-embed-text", prompt=text)
            vector = (
                response.get("embedding")
                if isinstance(response, dict)
                else getattr(response, "embedding", None)
            )
            return list(vector) if vector else None
        except Exception as exc:
            logger.info("Embedding failed: %s", exc)
            return None

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str | None = None,
        think: bool = False,
        model_name: str | None = None,
        prefer_utility: bool = False,
    ) -> dict[str, Any] | None:
        """One-shot constrained-decoding generation.

        Returns the parsed JSON object (Ollama validates it against ``schema``
        via ``format=``) or None on any failure. Reusable primitive behind short
        structured tasks (relevance grading, query rewriting). ``think=False``
        skips a reasoning model's <think> phase — these tasks need a verdict, not
        deliberation. ``prefer_utility=True`` routes the call to the small
        utility model when one is configured (cheap verdicts don't need the big
        model). Never raises.
        """
        if self._uses_openai_compatible():
            try:
                import json

                messages: list[dict[str, Any]] = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                utility = self.utility_model if prefer_utility else None
                with GENERATION_LOCK:
                    response = self._openai_chat(
                        messages,
                        model_name=model_name or utility or self.model,
                        response_format=schema,
                        options=self._options(thinking=bool(think)),
                    )
                self.last_stats = openai_stats(response)
                content = openai_message_content(response).strip()
                if not content:
                    return None
                data = json.loads(content)
                return data if isinstance(data, dict) else None
            except Exception as exc:
                logger.info("LM Studio structured generation failed: %s", exc)
                return None
        try:
            import json

            import ollama

            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            utility = self.utility_model if prefer_utility else None
            with GENERATION_LOCK:
                response = ollama.chat(
                    model=model_name or utility or self.model,
                    messages=messages,
                    stream=False,
                    format=schema,
                    think=think,
                    keep_alive=self.keep_alive,
                    options=self._options(thinking=bool(think)),
                )
            self.last_stats = _stats(response)
            content = _message_content(response).strip()
            if not content:
                return None
            data = json.loads(content)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.info("Structured generation failed: %s", exc)
            return None

    def get_response(
        self,
        context_messages: list[dict[str, Any]],
        user_summary: str = "",
        assistant_summary: str = "",
        assistant_name: str = DEFAULT_ASSISTANT_NAME,
        files_context: str = "",
        model_name: str | None = None,
        keep_alive: str | None = None,
        images: list[str] | None = None,
        think: bool | None = None,
    ) -> str | None:
        if self._uses_openai_compatible():
            try:
                started = time.monotonic()
                messages = self._build_messages(
                    context_messages,
                    user_summary=user_summary,
                    assistant_summary=assistant_summary,
                    assistant_name=assistant_name,
                    files_context=files_context,
                    images=images,
                )
                with GENERATION_LOCK:
                    response = self._openai_chat(
                        messages,
                        model_name=model_name,
                        options=self._options(thinking=bool(think)),
                    )
                self.last_stats = openai_stats(response, started)
                self._check_context_pressure()
                text = openai_message_content(response).strip()
                return text or None
            except Exception as exc:
                logger.warning("LM Studio chat failed: %s", exc)
                return None
        try:
            import ollama

            kwargs: dict[str, Any] = {
                "model": model_name or self.model,
                "messages": self._build_messages(
                    context_messages,
                    user_summary=user_summary,
                    assistant_summary=assistant_summary,
                    assistant_name=assistant_name,
                    files_context=files_context,
                    images=images,
                ),
                "stream": False,
                "keep_alive": keep_alive or self.keep_alive,
                "options": self._options(thinking=bool(think)),
            }
            if think is not None:
                kwargs["think"] = think
            with GENERATION_LOCK:
                response = ollama.chat(**kwargs)
            self.last_stats = _stats(response)
            self._check_context_pressure()
            text = _message_content(response).strip()
            return text or None
        except Exception as exc:
            logger.warning("Ollama chat failed: %s", exc)
            return None

    def stream_response(
        self,
        context_messages: list[dict[str, Any]],
        user_summary: str = "",
        assistant_summary: str = "",
        assistant_name: str = DEFAULT_ASSISTANT_NAME,
        files_context: str = "",
        model_name: str | None = None,
        keep_alive: str | None = None,
        images: list[str] | None = None,
        think: bool | None = None,
    ):
        if self._uses_openai_compatible():
            kwargs: dict[str, Any] = {
                "model": model_name or self.model,
                "messages": self._build_messages(
                    context_messages,
                    user_summary=user_summary,
                    assistant_summary=assistant_summary,
                    assistant_name=assistant_name,
                    files_context=files_context,
                    images=images,
                ),
                "options": self._options(thinking=bool(think)),
            }
            with GENERATION_LOCK:
                for attempt in (1, 2):
                    yielded = False
                    try:
                        stream = self._client().stream_chat(**kwargs)
                        for chunk in stream:
                            self.last_stats = openai_stats(chunk) or self.last_stats
                            choices = chunk.get("choices") if isinstance(chunk, dict) else None
                            if not choices:
                                continue
                            delta = (
                                choices[0].get("delta") if isinstance(choices[0], dict) else None
                            )
                            text = delta.get("content") if isinstance(delta, dict) else ""
                            if text:
                                yielded = True
                                yield str(text)
                        return
                    except Exception as exc:
                        logger.warning(
                            "LM Studio streaming chat failed (essai %s): %s", attempt, exc
                        )
                        if yielded:
                            yield "\n[⚠️ Flux interrompu par une erreur du moteur local.]"
                            return
                return
        kwargs: dict[str, Any] = {
            "model": model_name or self.model,
            "messages": self._build_messages(
                context_messages,
                user_summary=user_summary,
                assistant_summary=assistant_summary,
                assistant_name=assistant_name,
                files_context=files_context,
                images=images,
            ),
            "stream": True,
            "keep_alive": keep_alive or self.keep_alive,
            "options": self._options(thinking=bool(think)),
        }
        if think is not None:
            kwargs["think"] = think
        # One transparent retry when the stream dies BEFORE producing anything
        # (model still loading, transient socket error). Once content has been
        # yielded a retry would duplicate text, so the failure is surfaced
        # instead of silently ending the stream mid-sentence.
        # Hold the generation lock for the whole stream so background enrichment
        # never competes with the user's answer (released when the generator is
        # closed, including on early cancellation).
        with GENERATION_LOCK:
            for attempt in (1, 2):
                yielded = False
                try:
                    import ollama

                    stream = ollama.chat(**kwargs)
                    for chunk in stream:
                        self.last_stats = (
                            _stats(chunk) or self.last_stats
                        )  # final chunk has metrics
                        text = _message_content(chunk)
                        if text:
                            yielded = True
                            yield text
                    return
                except Exception as exc:
                    logger.warning("Ollama streaming chat failed (essai %s): %s", attempt, exc)
                    if yielded:
                        yield "\n[⚠️ Flux interrompu par une erreur du moteur local.]"
                        return
            return

    def _build_messages(
        self,
        context_messages: list[dict[str, Any]],
        user_summary: str = "",
        assistant_summary: str = "",
        assistant_name: str = DEFAULT_ASSISTANT_NAME,
        files_context: str = "",
        images: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        from lity.services.ai.context import clamp_history

        # Budgets scale with the EFFECTIVE window (hardware-aware num_ctx):
        # ~half for injected context, ~45% for the conversation itself, the rest
        # for the system prompt and the answer.
        num_ctx = self.effective_num_ctx()
        injected_budget = int(num_ctx * 0.5) * 4
        history_budget = int(num_ctx * 0.45) * 4

        system_content = self.system_prompt.strip().format(name=assistant_name)
        if self.system_prompt_extra:
            system_content += "\n\nINSTRUCTIONS PERSONNALISÉES :\n" + self.system_prompt_extra
        if getattr(self, "skills_prompt", ""):
            system_content += "\n\n" + self.skills_prompt
        if assistant_summary:
            system_content += "\n" + assistant_summary
        if user_summary:
            system_content += f"\nContexte utilisateur :\n{user_summary}"
        if files_context:
            system_content += "\n" + budget_injected_context(files_context, injected_budget)

        clean_context: list[dict[str, Any]] = []
        for msg in context_messages:
            if "role" not in msg or "content" not in msg:
                continue
            entry: dict[str, Any] = {"role": msg["role"], "content": msg["content"]}
            # Carry attachments persisted on the message (Ollama wants raw base64)
            # so images stay visible to vision models across follow-up turns,
            # regenerate, and edit — not just on the turn they were attached.
            stored_images = msg.get("images")
            if stored_images:
                carried = [_strip_data_url(image) for image in stored_images if image]
                if carried:
                    entry["images"] = carried
            clean_context.append(entry)
        # Safety net under the rolling summary: keep the newest turns within the
        # window instead of letting Ollama truncate the system prompt silently.
        clean_context = clamp_history(clean_context, history_budget)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            *clean_context,
        ]
        # Live attachments passed for this turn attach to the latest user message
        # (fallback for callers that pass images without persisting them).
        if images:
            cleaned = [_strip_data_url(image) for image in images if image]
            for message in reversed(messages):
                if message.get("role") == "user":
                    message["images"] = cleaned
                    break
        return messages
