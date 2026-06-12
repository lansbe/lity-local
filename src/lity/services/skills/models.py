from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Per the Agent Skills spec (agentskills.io): name is 1-64 chars, lowercase
# a-z/0-9 and hyphens; description is the load-bearing trigger field, capped at
# 1024 chars by the upstream API. We sanitise rather than reject (this is a
# local user feature), but keep the same shape so skills stay portable.
_NAME_MAX = 64
_DESCRIPTION_MAX = 1024
_BODY_CHAR_BUDGET = 20_000  # ~5k tokens — the Level-2 ceiling the spec recommends

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)

# French + English stop words: stripped from the lexical match surface so a skill
# matches on its meaningful terms, not on "the/le/de/and".
_STOP_WORDS_TEXT = """
    le la les un une des du de d au aux et ou à a en dans par pour sur avec sans
    ce cet cette ces se sa son ses leur leurs ne pas que qui quoi dont où est
    sont être avoir fais fait faire ton ta tes tu te toi je me mon ma mes nous
    vous il elle on ils elles y comme plus moins très peu si car donc or ni mais
    the a an of to in on for and or with without this that these those is are be
    you your it its as at by from
"""
_STOP_WORDS = frozenset(_STOP_WORDS_TEXT.split())


def _fold_accents(text: str) -> str:
    """Drop combining marks so French names slug cleanly (compétence ->
    competence, synthèse -> synthese) instead of fragmenting on the accent."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def slugify(value: str) -> str:
    """Lowercase-hyphen slug constrained to the skill-name charset."""
    text = _SLUG_RE.sub("-", _fold_accents(str(value)).strip().lower()).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:_NAME_MAX] or "competence"


def tokenize(text: str) -> list[str]:
    """Meaningful lowercase tokens (stop words and 1-char noise removed)."""
    return [
        token
        for raw in _TOKEN_RE.findall(str(text).lower())
        if len(token := raw) > 1 and token not in _STOP_WORDS
    ]


@dataclass(frozen=True)
class Skill:
    """A parsed SKILL.md — the unit the router selects and the engine injects.

    ``body`` is the Level-2 instructions (loaded only when the skill is
    activated). ``description`` + ``when_to_use`` + ``triggers`` form the
    Level-1 matching surface. ``source`` distinguishes a bundled built-in from
    a user-authored skill; user skills win a name collision.
    """

    name: str
    description: str
    body: str
    when_to_use: str = ""
    triggers: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    source: str = "user"  # "builtin" | "user"
    path: str = ""

    @property
    def builtin(self) -> bool:
        return self.source == "builtin"

    def match_tokens(self) -> set[str]:
        """The token set the lexical router ranks a request against.

        Name and explicit triggers are weighted by repetition (they are the
        strongest signal), then the description and when_to_use prose.
        """
        tokens: list[str] = []
        tokens += tokenize(self.name.replace("-", " ")) * 3
        for trigger in self.triggers:
            tokens += tokenize(trigger) * 3
        tokens += tokenize(self.when_to_use) * 2
        tokens += tokenize(self.description)
        return set(tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "triggers": list(self.triggers),
            "allowed_tools": list(self.allowed_tools),
            "source": self.source,
            "builtin": self.builtin,
            "path": self.path,
        }


def normalize_name(raw: str, fallback: str) -> str:
    candidate = slugify(raw) if raw else ""
    return candidate or slugify(fallback)


def clamp_description(raw: str) -> str:
    text = " ".join(str(raw).split())
    return text[:_DESCRIPTION_MAX]


def clamp_body(raw: str) -> str:
    text = str(raw).strip()
    if len(text) <= _BODY_CHAR_BUDGET:
        return text
    return text[:_BODY_CHAR_BUDGET].rstrip() + "\n\n[… compétence tronquée …]"
