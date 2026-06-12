from __future__ import annotations

from typing import Any

from lity.services.ai._engine_common import (
    _INJECTED_CONTEXT_BUDGET_CHARS,
    DEFAULT_NUM_CTX,
    budget_injected_context,
)

# ~4 chars per token is the same heuristic the injected-context budget uses.
_CHARS_PER_TOKEN = 4

# When the agent's working conversation exceeds this share of the context
# window, Ollama starts truncating SILENTLY from the head — dropping the system
# prompt and the task itself. Compact BEFORE that happens, visibly.
_AGENT_WORK_SHARE = 0.75
DEFAULT_AGENT_WORK_BUDGET_CHARS = int(DEFAULT_NUM_CTX * _AGENT_WORK_SHARE) * _CHARS_PER_TOKEN

# How many of the most recent messages survive compaction verbatim: the model
# needs its latest observations intact to act on them.
_KEEP_RECENT = 8

# Older tool observations shrink to a digest of this size — enough to remember
# WHAT happened without re-paying for the full output every remaining step.
_DIGEST_CHARS = 240

_COMPACTION_NOTE = (
    "[HISTORIQUE COMPACTÉ] Les anciennes étapes ci-dessous ont été résumées pour "
    "tenir dans la fenêtre de contexte. Les informations complètes peuvent être "
    "relues avec les outils si nécessaire."
)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Local engines expose no tokenizer
    fast enough to call per-message; this errs on the safe side."""
    return (len(text or "") + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def estimate_messages_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def _digest(message: dict[str, Any]) -> dict[str, Any]:
    """Shrink one mid-conversation message while keeping the dialogue valid."""
    content = str(message.get("content", ""))
    if len(content) <= _DIGEST_CHARS:
        return message
    role = message.get("role")
    if role == "tool":
        name = message.get("name", "outil")
        clipped = content[:_DIGEST_CHARS]
        return {
            "role": "tool",
            "name": name,
            "content": f"[résultat de {name}, compacté]\n{clipped}…",
        }
    clipped = content[:_DIGEST_CHARS]
    compact = {key: value for key, value in message.items() if key != "content"}
    compact["content"] = clipped + "…"
    return compact


def compact_agent_messages(
    messages: list[dict[str, Any]],
    max_chars: int = DEFAULT_AGENT_WORK_BUDGET_CHARS,
    keep_recent: int = _KEEP_RECENT,
) -> list[dict[str, Any]]:
    """Compact an agent working conversation that outgrew its budget.

    Deterministic and LLM-free (no extra latency): keeps the system prompt and
    the FIRST user message (the task) verbatim, keeps the most recent
    ``keep_recent`` messages verbatim, and digests everything in between (old
    tool outputs shrink to short summaries). If that still busts the budget, the
    middle is dropped entirely behind an explicit marker. Never silent: the
    model is told the history was compacted. Returns the input list unchanged
    when it already fits.
    """
    if estimate_messages_chars(messages) <= max_chars:
        return messages

    head: list[dict[str, Any]] = []
    rest = list(messages)
    if rest and rest[0].get("role") == "system":
        head.append(rest.pop(0))
    first_user = next((m for m in rest if m.get("role") == "user"), None)
    if first_user is not None:
        head.append(first_user)
        rest.remove(first_user)

    if len(rest) <= keep_recent:
        return messages  # nothing compactable beyond the protected zone

    recent = rest[-keep_recent:]
    middle = [_digest(message) for message in rest[:-keep_recent]]
    note = {"role": "user", "content": _COMPACTION_NOTE}
    compacted = head + [note] + middle + recent

    if estimate_messages_chars(compacted) > max_chars:
        # Digests were not enough: summarize the middle into one line per step.
        lines = []
        for message in middle:
            role = message.get("role")
            label = message.get("name") if role == "tool" else role
            lines.append(f"- {label}: {str(message.get('content', ''))[:80]}")
        summary = {
            "role": "user",
            "content": _COMPACTION_NOTE + "\nRésumé des étapes :\n" + "\n".join(lines[-40:]),
        }
        compacted = head + [summary] + recent

    return compacted


def compose_injected_context(
    sections: list[tuple[str, float]],
    max_chars: int = _INJECTED_CONTEXT_BUDGET_CHARS,
) -> str:
    """Per-type budgets with rollover so one section can't evict the others.

    Before this, facts + memory + RAG + loaded files were concatenated then
    clipped as ONE blob — a big loaded file silently evicted everything after
    it, and vice versa. Here each (text, share) gets its proportional slice of
    the budget; unused space rolls over to the next section. Give the
    code/files section the dominant share: for a coding task the loaded code
    matters more than recalled conversation snippets."""
    present = [(text, share) for text, share in sections if text]
    if not present:
        return ""
    total_share = sum(share for _text, share in present) or 1.0
    parts: list[str] = []
    rollover = 0
    for text, share in present:
        budget = int(max_chars * share / total_share) + rollover
        if len(text) <= budget:
            parts.append(text)
            rollover = budget - len(text)
        else:
            parts.append(budget_injected_context(text, max_chars=max(budget, 200)))
            rollover = 0
    return "".join(parts)


def clamp_history(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Keep the NEWEST chat turns within a char budget, dropping the oldest.

    Safety net under the rolling summary: a handful of giant pasted messages
    can blow the window even within the 20-turn memory window. The cut is
    explicit (a short system note replaces what was dropped)."""
    if estimate_messages_chars(messages) <= max_chars:
        return messages
    kept: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        size = len(str(message.get("content", "")))
        if kept and used + size > max_chars:
            break
        kept.append(message)
        used += size
    kept.reverse()
    if len(kept) < len(messages):
        kept.insert(
            0,
            {
                "role": "system",
                "content": "[Des messages plus anciens ont été retirés pour tenir "
                "dans la fenêtre de contexte.]",
            },
        )
    return kept
