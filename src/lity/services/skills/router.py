from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lity.services.skills.models import Skill, tokenize

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], "list[float] | None"]
StructuredFn = Callable[[str, dict], "dict[str, Any] | None"]

# Tuning. The router is deliberately conservative: chit-chat must NOT activate a
# skill (it would waste the small model's context), so a turn needs real token
# overlap before a skill is even a candidate.
_FLOOR = 0.08  # below this a skill is not even a candidate for the model pick
_THRESHOLD = 0.22  # at/above this the top lexical match auto-activates
_TOP_K = 4
_LEXICAL_WEIGHT = 0.55  # blend weight when embeddings are available

_NONE = "aucune"

_PICK_PROMPT = (
    "Tu choisis la compétence la plus pertinente pour traiter la demande de "
    "l'utilisateur, ou « {none} » si aucune ne s'applique vraiment.\n\n"
    "DEMANDE :\n{request}\n\n"
    "COMPÉTENCES CANDIDATES :\n{candidates}\n\n"
    "Réponds par le nom EXACT d'une seule compétence ci-dessus, ou « {none} » si "
    "la demande n'a pas besoin de l'une d'elles. Ne choisis une compétence que si "
    "elle correspond clairement."
)


@dataclass(frozen=True)
class SkillSelection:
    skill: Skill
    score: float
    method: str  # "lexical" | "model"


def _strong_tokens(skill: Skill) -> set[str]:
    tokens: set[str] = set(tokenize(skill.name.replace("-", " ")))
    for trigger in skill.triggers:
        tokens |= set(tokenize(trigger))
    return tokens


def _weak_tokens(skill: Skill) -> set[str]:
    return set(tokenize(skill.when_to_use)) | set(tokenize(skill.description))


_MIN_PREFIX = 4  # shared-prefix length that counts as a match (French inflections)


def _prefix_match(a: str, b: str) -> bool:
    """True when two tokens are the same word up to inflection — equal, or
    sharing a prefix of ≥4 chars (traduire / traduction / traduis all match)."""
    if a == b:
        return True
    common = 0
    for char_a, char_b in zip(a, b, strict=False):
        if char_a != char_b:
            break
        common += 1
    return common >= _MIN_PREFIX


def _matches_any(token: str, terms: set[str]) -> bool:
    return any(_prefix_match(token, term) for term in terms)


def lexical_score(request_tokens: set[str], skill: Skill) -> float:
    """Fraction of the request covered by the skill's terms, weighting a
    name/trigger hit twice a description hit. 1.0 = every request token is a
    strong trigger; 0.0 = no overlap. Matching is prefix-based so French
    inflections (relire / relis, traduire / traduction) still count."""
    if not request_tokens:
        return 0.0
    strong = _strong_tokens(skill)
    weak = _weak_tokens(skill)
    matched_strong = 0
    matched_weak = 0
    for token in request_tokens:
        if _matches_any(token, strong):
            matched_strong += 1
        elif _matches_any(token, weak):
            matched_weak += 1
    raw = 2 * matched_strong + matched_weak
    return raw / (2 * len(request_tokens))


def _cosine(a: list[float], b: list[float]) -> float:
    from lity.services.rag.vector_index import cosine_similarity

    return cosine_similarity(a, b)


class SkillRouter:
    """Two-stage skill selector tuned for small local models.

    Stage 1 (lexical, always) ranks enabled skills by token overlap with the
    request — fully offline and deterministic. Stage 2 is optional and only runs
    when stage 1 surfaces a candidate, so a greeting never pays for it:
      - with an embedding model: blend cosine similarity into the ranking;
      - with a constrained-decoding callable: let the model pick exactly one of
        the top-k candidates (or "aucune") over a short list — the reliable way
        to get a weak model to choose (vs. dumping a flat catalogue at it).
    Always offers a "no skill" outcome so an irrelevant turn stays unaffected.
    """

    def __init__(
        self,
        *,
        embed: EmbedFn | None = None,
        structured: StructuredFn | None = None,
        semantic: bool = False,
        threshold: float = _THRESHOLD,
        floor: float = _FLOOR,
        top_k: int = _TOP_K,
    ):
        self.embed = embed
        self.structured = structured
        self.semantic = semantic and embed is not None
        self.threshold = threshold
        self.floor = floor
        self.top_k = top_k
        self._embed_cache: dict[str, list[float] | None] = {}

    def rank(self, text: str, skills: list[Skill]) -> list[tuple[Skill, float]]:
        request = set(tokenize(text))
        scored = [(skill, lexical_score(request, skill)) for skill in skills]
        if self.semantic:
            scored = self._blend_semantic(text, scored)
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _blend_semantic(
        self, text: str, scored: list[tuple[Skill, float]]
    ) -> list[tuple[Skill, float]]:
        # Only embed the request once, and only when something is already a weak
        # candidate — embedding on pure chit-chat is wasted latency.
        if not any(score >= self.floor for _skill, score in scored):
            return scored
        query_vec = self._embed(text)
        if query_vec is None:
            return scored
        blended: list[tuple[Skill, float]] = []
        for skill, lexical in scored:
            skill_vec = self._embed(
                f"{skill.description} {skill.when_to_use}".strip(), key=skill.name
            )
            if skill_vec is None:
                blended.append((skill, lexical))
                continue
            cosine = max(0.0, _cosine(query_vec, skill_vec))
            blended.append((skill, _LEXICAL_WEIGHT * lexical + (1 - _LEXICAL_WEIGHT) * cosine))
        return blended

    def _embed(self, text: str, key: str | None = None) -> list[float] | None:
        if self.embed is None:
            return None
        cache_key = key or text
        if cache_key not in self._embed_cache:
            try:
                self._embed_cache[cache_key] = self.embed(text)
            except Exception:  # pragma: no cover - defensive
                self._embed_cache[cache_key] = None
        return self._embed_cache[cache_key]

    def select(self, text: str, skills: list[Skill]) -> SkillSelection | None:
        if not skills or not text.strip():
            return None
        ranked = self.rank(text, skills)
        if not ranked:
            return None
        candidates = [(skill, score) for skill, score in ranked if score >= self.floor][
            : self.top_k
        ]
        if not candidates:
            return None

        # Stage 2: let the model pick over the short list when we can constrain it.
        if self.structured is not None:
            chosen = self._model_pick(text, [skill for skill, _ in candidates])
            if chosen is not None:
                score = next((s for skill, s in candidates if skill.name == chosen.name), 0.0)
                return SkillSelection(skill=chosen, score=score, method="model")
            # An explicit "aucune" from the model is respected only when the top
            # lexical signal is weak; a very strong keyword match still wins.
            top_skill, top_score = candidates[0]
            if top_score >= max(self.threshold, 0.5):
                return SkillSelection(skill=top_skill, score=top_score, method="lexical")
            return None

        # No model available → activate the top match only when clearly relevant.
        top_skill, top_score = candidates[0]
        if top_score >= self.threshold:
            return SkillSelection(skill=top_skill, score=top_score, method="lexical")
        return None

    def _model_pick(self, text: str, candidates: list[Skill]) -> Skill | None:
        names = [skill.name for skill in candidates]
        listing = "\n".join(
            f"- {skill.name} : {skill.description}"
            + (f" (quand : {skill.when_to_use})" if skill.when_to_use else "")
            for skill in candidates
        )
        schema = {
            "type": "object",
            "properties": {"competence": {"type": "string", "enum": [*names, _NONE]}},
            "required": ["competence"],
        }
        prompt = _PICK_PROMPT.format(none=_NONE, request=text.strip(), candidates=listing)
        try:
            verdict = self.structured(prompt, schema)
        except Exception:  # pragma: no cover - defensive
            return None
        if not isinstance(verdict, dict):
            return None
        choice = str(verdict.get("competence", "")).strip()
        if choice == _NONE or choice not in names:
            return None
        return next((skill for skill in candidates if skill.name == choice), None)
