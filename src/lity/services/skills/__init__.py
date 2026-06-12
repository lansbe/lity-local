"""Agent Skills ("Compétences") — model-applicable know-how packaged as folders.

A skill is a ``SKILL.md`` (YAML frontmatter ``name`` + ``description``, then a
markdown body) optionally bundling reference files and scripts, following the
open Agent Skills format used by Claude Code and Codex. Progressive disclosure:
every enabled skill's name+description stays in the system prompt (cheap), and a
skill's full body is injected only when the turn matches it.

Selection is tuned for a small local model — see :class:`SkillRouter`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lity.services.skills.catalog import build_skills_prompt, format_active, format_catalog
from lity.services.skills.models import Skill, slugify, tokenize
from lity.services.skills.router import SkillRouter, SkillSelection
from lity.services.skills.store import SkillStore

logger = logging.getLogger(__name__)

__all__ = [
    "Skill",
    "SkillStore",
    "SkillRouter",
    "SkillSelection",
    "build_skill_store",
    "build_skills_prompt",
    "format_active",
    "format_catalog",
    "slugify",
    "tokenize",
]


def build_skill_store(user_dir: Path | str, builtin_dir: Path | str | None = None) -> SkillStore:
    """Construct a :class:`SkillStore`, ensuring the writable user directory
    exists so the UI can create skills into it on first use."""
    user_path = Path(user_dir)
    try:
        user_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - defensive
        logger.info("Skills user dir unavailable (%s): %s", user_path, exc)
    return SkillStore(user_path, builtin_dir)
