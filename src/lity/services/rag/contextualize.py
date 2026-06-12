from __future__ import annotations

from collections.abc import Callable
from typing import Any

# generate_fn(prompt, schema) -> parsed JSON object (or None). Injected so the
# rewrite logic is fully unit-testable WITHOUT a live model.
GenerateFn = Callable[[str, "dict[str, Any]"], "dict[str, Any] | None"]

# Constrained-decoding schema (Ollama ``format=``): force a single clean string,
# so the model can't drift into prose or, worse, answer the question.
_REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"standalone": {"type": "string"}},
    "required": ["standalone"],
}

_CONTEXTUALIZE_PROMPT = (
    "À partir des derniers échanges puis de la NOUVELLE DEMANDE, réécris cette "
    "demande en UNE seule requête de recherche autonome et explicite.\n"
    "- Remplace les références implicites (« ça », « celui-là », « le deuxième », "
    "« et pour X ? », pronoms, sous-entendus) par les termes réels tirés des "
    "échanges.\n"
    "- Garde-la courte et factuelle, dans la langue de la demande. N'Y RÉPONDS PAS, "
    "ne fais que la reformuler.\n"
    "- Si la demande est déjà autonome, renvoie-la quasi inchangée.\n\n"
    "Derniers échanges :\n{history}\n\n"
    "NOUVELLE DEMANDE : {query}"
)

# A reformulation is naturally short; anything past this means the model rambled
# or answered instead of rewriting — fall back to the raw query in that case.
_MAX_REWRITE_CHARS = 500


def contextualize_query(
    generate_fn: GenerateFn | None,
    history: list[dict[str, Any]],
    query: str,
    *,
    max_turns: int = 6,
) -> str:
    """Rewrite a possibly-elliptical follow-up into a standalone retrieval query.

    Conversational retrieval breaks on follow-ups: "et le deuxième ?", "corrige
    ça", "pourquoi ?" carry no standalone meaning, so searching on that text
    alone returns off-topic chunks and the answer's relevance fades. Resolving
    the latest message against the recent turns — the *history-aware retriever*
    pattern — restores recall on multi-turn conversations.

    100% local: the rewrite uses the SAME Ollama model via constrained decoding
    (``format=schema``), passed in as ``generate_fn``. Never raises and never
    returns empty — it falls back to the original ``query`` on any problem (no
    grader, blank/oversized output, exception), so retrieval is never made worse
    than the previous "last user message only" behaviour. Bounded cost: at most
    ONE small structured call, and zero when there is no prior turn to resolve
    against (the message is already standalone).
    """
    cleaned = (query or "").strip()
    if not cleaned or generate_fn is None:
        return cleaned

    turns = [
        message
        for message in (history or [])
        if message.get("role") in ("user", "assistant") and str(message.get("content", "")).strip()
    ]
    # Drop the trailing user turn when it IS the current query, so it is not
    # duplicated across "history" and "NOUVELLE DEMANDE".
    if (
        turns
        and turns[-1].get("role") == "user"
        and str(turns[-1].get("content", "")).strip() == cleaned
    ):
        turns = turns[:-1]
    if not turns:
        return cleaned  # nothing prior to disambiguate against → already standalone

    convo = "\n".join(
        f"{message['role']}: {str(message['content']).strip()}" for message in turns[-max_turns:]
    )[:2000]
    try:
        data = generate_fn(
            _CONTEXTUALIZE_PROMPT.format(history=convo, query=cleaned), _REWRITE_SCHEMA
        )
    except Exception:
        return cleaned
    if not isinstance(data, dict):
        return cleaned
    rewritten = str(data.get("standalone", "")).strip().strip('"').strip()
    if not rewritten or len(rewritten) > _MAX_REWRITE_CHARS:
        return cleaned
    return rewritten
