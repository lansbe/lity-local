from __future__ import annotations

import json
import re
import threading
from typing import Any

# Process-wide serialization of local-model generations. A single Ollama server
# allocates a separate KV cache per *concurrent* request, so the main agent turn
# running at the same time as the background enrichment threads (title, fact
# extraction, summary) multiplies memory use and thrashes a laptop GPU/RAM —
# the cause of whole-machine lag on a trivial turn. Holding this lock around
# every generation funnels them one-at-a-time: the background work simply queues
# behind the user's turn instead of competing with it. RLock so a path that
# re-enters on the same thread never self-deadlocks.
GENERATION_LOCK = threading.RLock()

# Context window for chat/agent turns. The Ollama default is 4096, which SILENTLY
# truncates an agent loop (it drops the tool list + system guidance mid-run). A
# larger window is the single most impactful agent-reliability fix; Ollama clamps
# it to the model's trained maximum, so over-asking is harmless.
DEFAULT_NUM_CTX = 16384


def _sampling_for(model: str, *, thinking: bool = False) -> dict[str, Any]:
    """Per-family sampling params + num_ctx. NEVER greedy (Qwen3 forbids it:
    greedy → repetition/degradation). Thinking models want a slightly different
    profile (higher top_p, min_p) than their non-thinking turns."""
    name = (model or "").lower()
    if "qwen" in name:
        params = (
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
            if thinking
            else {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
        )
    elif "deepseek" in name:
        params = {"temperature": 0.6, "top_p": 0.95}
    elif "mistral" in name or "mixtral" in name:
        params = {"temperature": 0.7, "top_p": 0.9}
    else:  # llama / gemma / phi / unknown
        params = {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
    params["num_ctx"] = DEFAULT_NUM_CTX
    return params


# Budget for the discretionary injected context (RAG + cross-session memory +
# loaded files). The loaded-files reader alone can hand back up to 120k chars —
# roughly twice this whole window — so without a cap a single large file pushes
# the system prompt and the recent turns past num_ctx, where Ollama truncates
# SILENTLY (typically from the start, dropping the instructions). That is exactly
# how a long session "loses the thread". Keep injected context to about half the
# window (~4 chars/token) so instructions + history + the answer still fit, and
# trim VISIBLY rather than letting the truncation happen out of sight.
_CHARS_PER_TOKEN = 4
_INJECTED_CONTEXT_BUDGET_CHARS = int(DEFAULT_NUM_CTX * 0.5) * _CHARS_PER_TOKEN
_TRUNCATION_MARKER = "\n[…contexte tronqué pour tenir dans la fenêtre de contexte…]\n"


def budget_injected_context(text: str, max_chars: int = _INJECTED_CONTEXT_BUDGET_CHARS) -> str:
    """Clip injected context to a char budget, keeping the HEAD and flagging the
    cut. Head-keep is deliberate: the highest-signal items (retrieved snippets,
    then the start of a loaded file) come first, so trimming the tail sheds the
    least relevant material while the system prompt and the conversation that
    follow stay intact. Never silently drops content — the marker tells the model
    (and the reader) that something was cut."""
    if not text or len(text) <= max_chars:
        return text
    keep = max_chars - len(_TRUNCATION_MARKER)
    if keep <= 0:
        return _TRUNCATION_MARKER
    clipped = text[:keep]
    newline = clipped.rfind("\n")
    if newline > keep * 0.6:  # snap to a line boundary when it costs little
        clipped = clipped[:newline]
    return clipped + _TRUNCATION_MARKER


# Constrained-decoding schemas (Ollama `format=`). Forcing the shape of short,
# deterministic outputs removes the brittle "find the JSON in free text" parsing
# and guarantees a valid label/object. NOT used for chat or agent reasoning,
# where a forced grammar would degrade the answer.
_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "categorie": {
            "type": "string",
            "enum": ["assistant_profile", "user_profile", "long_term_facts"],
        },
        "cle": {"type": "string"},
        "valeur": {"type": "string"},
    },
    "required": ["found"],
}


# --- Think-routing -----------------------------------------------------------
# Reasoning ("thinking") models burn seconds on a <think> phase before answering.
# That pays off on hard problems and is pure waste on "merci" / "ok". These two
# helpers gate it: only for thinking-capable families, and only when the message
# looks like it needs reasoning. CONSERVATIVE — think by default, skip ONLY clearly
# trivial short messages, so answer quality is never traded away.
_THINKING_FAMILIES = ("qwen3", "deepseek-r1", "qwq", "marco-o1", "-r1", "/r1")
_THINK_CUES = (
    "pourquoi",
    "explique",
    "expliqu",
    "compare",
    "démontre",
    "demontre",
    "calcule",
    "résous",
    "resous",
    "résoudre",
    "resoudre",
    "analyse",
    "optimise",
    "optimis",
    "debug",
    "débogue",
    "corrige",
    "implémente",
    "implemente",
    "algorithme",
    "étape",
    "etape",
    "raisonne",
    "prouve",
    "déduis",
    "deduis",
    "code",
    "fonction",
    "refactor",
    "traduis",
    "résume",
    "resume",
    "planifie",
    "conçois",
    "concois",
    "stratégie",
    "strategie",
)


def is_thinking_model(model: str) -> bool:
    """True for reasoning models that emit a <think> phase (Qwen3, DeepSeek-R1…)."""
    name = (model or "").lower()
    return any(tag in name for tag in _THINKING_FAMILIES)


def wants_thinking(text: str) -> bool:
    """Should this user turn use the model's <think> phase?

    Returns True (reason) for substantial / reasoning / arithmetic / question
    turns, False (skip) only for short, cue-less, social messages. Errs toward
    True: a false positive merely costs a little latency, a false negative would
    cost answer quality on a hard question.
    """
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return False
    words = cleaned.split()
    if len(words) > 8:
        return True
    if any(cue in cleaned for cue in _THINK_CUES):
        return True
    if any(ch.isdigit() for ch in cleaned) and any(op in cleaned for op in "+-*/=%^"):
        return True
    return "?" in cleaned and len(words) > 4


def _stats(response: Any) -> dict[str, Any]:
    """Extract eval metrics (tokens, durations) from an Ollama response/chunk."""

    def field(key: str) -> int:
        value = response.get(key) if isinstance(response, dict) else getattr(response, key, None)
        return int(value) if isinstance(value, int) else 0

    eval_count = field("eval_count")
    eval_duration = field("eval_duration")  # nanoseconds
    prompt_count = field("prompt_eval_count")
    tps = round(eval_count / (eval_duration / 1e9), 1) if eval_duration else 0.0
    return {
        "eval_count": eval_count,
        "prompt_eval_count": prompt_count,
        "context_used": prompt_count + eval_count,
        "tokens_per_sec": tps,
    }


def _normalize_fact(data: Any) -> dict[str, Any] | None:
    """Validate a fact-extraction object into ``{categorie, cle, valeur}`` or None.

    Returns None when the model reports no durable fact (``found: false``) or
    when the key/value pair is incomplete, so the caller never stores a partial
    or fabricated fact.
    """
    if not isinstance(data, dict) or data.get("found") is False:
        return None
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("cat"):
            normalized["categorie"] = value
        elif key == "cle":
            normalized["cle"] = value
        elif key in {"valeur", "val"}:
            normalized["valeur"] = value
    if not normalized.get("cle") or not normalized.get("valeur"):
        return None
    return normalized


def _clean_title(raw: str) -> str | None:
    """Normalize a model-produced title: first line, no quotes/prefix/punctuation."""
    if not raw or not raw.strip():
        return None
    title = raw.strip().splitlines()[0]
    title = re.sub(r"(?i)^\s*(titre|title)\s*[:\-–]\s*", "", title)
    # Strip surrounding quotes and stray edge punctuation from both ends.
    title = title.strip(" \t\"'«»`*.!?:;,")
    if not title:
        return None
    words = title.split()
    if len(words) > 8:
        title = " ".join(words[:8])
    return title[:60] or None


def _strip_data_url(data: str) -> str:
    """Ollama expects raw base64; drop any ``data:image/...;base64,`` prefix."""
    if isinstance(data, str) and data.startswith("data:") and "," in data:
        return data.split(",", 1)[1]
    return data


def _match_installed_model(candidate: str | None, installed_models: list[str]) -> str | None:
    if not candidate:
        return None
    if candidate in installed_models:
        return candidate

    candidate_base = candidate.split(":", 1)[0]
    for model in installed_models:
        if model.split(":", 1)[0] == candidate_base:
            return model
    return None


def _normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
    if message is None:
        return []
    raw_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not raw_calls:
        return []

    normalized: list[dict[str, Any]] = []
    for call in raw_calls:
        function = (
            call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        )
        if function is None:
            continue
        name = function.get("name") if isinstance(function, dict) else getattr(function, "name", "")
        arguments = (
            function.get("arguments")
            if isinstance(function, dict)
            else getattr(function, "arguments", {})
        )
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        if name:
            normalized.append({"name": str(name), "arguments": arguments})
    return normalized


def _message_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", ""))
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))
