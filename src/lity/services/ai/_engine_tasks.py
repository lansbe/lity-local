from __future__ import annotations

import json
import logging
from typing import Any

from lity.services.ai._engine_common import (
    _FACT_SCHEMA,
    GENERATION_LOCK,
    _clean_title,
    _message_content,
    _normalize_fact,
)
from lity.services.ai.openai_compatible import openai_message_content
from lity.services.ai.prompts import (
    FACT_EXTRACTION_PROMPT,
    SUMMARY_PROMPT,
    TITLE_PROMPT,
)

logger = logging.getLogger(__name__)


class AuxTasksMixin:
    """Short, one-shot auxiliary LLM tasks layered onto the chat engine.

    Each is best-effort and degrades to ``None`` on failure so the caller keeps
    a cheap fallback. Mixed into :class:`AIEngine`, which provides ``model``,
    ``keep_alive`` and the optional ``utility_model`` — a small resident model
    these background tasks prefer, so they neither pay the big model's latency
    nor evict it from memory.
    """

    def _aux_model(self, model_name: str | None = None) -> str:
        return model_name or getattr(self, "utility_model", None) or self.model

    def summarize_context(
        self,
        prior_summary: str,
        messages: list[dict[str, Any]],
        model_name: str | None = None,
    ) -> str | None:
        """Update a rolling conversation summary with new (older) messages."""
        if not messages:
            return None
        convo = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)[:6000]
        user = (f"Résumé actuel :\n{prior_summary}\n\n" if prior_summary else "") + (
            f"Nouveaux échanges à intégrer :\n{convo}"
        )
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            try:
                with GENERATION_LOCK:
                    response = self._openai_chat(
                        [
                            {"role": "system", "content": SUMMARY_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        model_name=self._aux_model(model_name),
                        options={"temperature": 0.3, "num_predict": 512},
                    )
                text = openai_message_content(response).strip()
                return text or None
            except Exception as exc:
                logger.info("LM Studio context summary failed: %s", exc)
                return None
        try:
            import ollama

            with GENERATION_LOCK:
                response = ollama.chat(
                    model=self._aux_model(model_name),
                    messages=[
                        {"role": "system", "content": SUMMARY_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    stream=False,
                    keep_alive=self.keep_alive,
                    options={"temperature": 0.3, "num_predict": 512},
                )
            text = _message_content(response).strip()
            return text or None
        except Exception as exc:
            logger.info("Context summary failed: %s", exc)
            return None

    def generate_title(self, text: str, model_name: str | None = None) -> str | None:
        """Generate a short conversation title from the first user message.

        Returns a cleaned title, or ``None`` if Ollama is unavailable or the
        answer is empty (caller keeps the message-derived fallback title).
        """
        if not text or not text.strip():
            return None
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            try:
                with GENERATION_LOCK:
                    response = self._openai_chat(
                        [
                            {"role": "system", "content": TITLE_PROMPT},
                            {"role": "user", "content": text.strip()[:1000]},
                        ],
                        model_name=self._aux_model(model_name),
                        options={"temperature": 0.2, "num_predict": 24},
                    )
                return _clean_title(openai_message_content(response))
            except Exception as exc:
                logger.info("LM Studio title generation failed: %s", exc)
                return None
        try:
            import ollama

            with GENERATION_LOCK:
                response = ollama.chat(
                    model=self._aux_model(model_name),
                    messages=[
                        {"role": "system", "content": TITLE_PROMPT},
                        {"role": "user", "content": text.strip()[:1000]},
                    ],
                    stream=False,
                    keep_alive=self.keep_alive,
                    options={"temperature": 0.2, "num_predict": 24},
                )
            return _clean_title(_message_content(response))
        except Exception as exc:
            logger.info("Title generation failed: %s", exc)
            return None

    def extract_fact(self, last_user_message: str) -> dict[str, Any] | None:
        if getattr(self, "_uses_openai_compatible", lambda: False)():
            prompt = FACT_EXTRACTION_PROMPT.format(last_user_message=last_user_message)
            data = self.generate_structured(
                prompt,
                _FACT_SCHEMA,
                think=False,
                model_name=self._aux_model(),
            )
            return _normalize_fact(data)
        try:
            import ollama

            prompt = FACT_EXTRACTION_PROMPT.format(last_user_message=last_user_message)
            with GENERATION_LOCK:
                response = ollama.generate(
                    model=self._aux_model(),
                    prompt=prompt,
                    format=_FACT_SCHEMA,
                    think=False,  # short structured extraction: no reasoning needed
                    options={"temperature": 0},
                )
            content = response["response"].strip()
            if "{" not in content:
                return None
            start = content.find("{")
            end = content.rfind("}") + 1
            return _normalize_fact(json.loads(content[start:end]))
        except Exception:
            return None
