from __future__ import annotations

from lity.services.skills.models import Skill

# Level 1 (catalogue) budget: names+descriptions for every enabled skill stay
# resident so the model knows what it can do. Bounded so a big library can't
# crowd out the conversation (the same overflow guard Claude Code / Codex apply).
_CATALOG_MAX_SKILLS = 15
_CATALOG_DESC_CHARS = 180

_CATALOG_HEADER = (
    "COMPÉTENCES DISPONIBLES (savoir-faire que tu peux appliquer quand la demande s'y prête) :"
)

_ACTIVE_HEADER = "COMPÉTENCE ACTIVE pour cette demande — « {name} » :"
_ACTIVE_FOOTER = (
    "Applique cette méthode pour répondre à la demande. Si, à la lecture, elle ne "
    "correspond pas vraiment à ce qui est demandé, ignore-la et réponds normalement."
)


def substitute_paths(body: str, skill: Skill) -> str:
    """Resolve ${SKILL_DIR}/${CLAUDE_SKILL_DIR} to the skill's folder so a body
    that references bundled scripts/assets works regardless of the cwd (the
    scripts themselves only run through the gated run_command tool)."""
    if not skill.path:
        return body
    for token in ("${SKILL_DIR}", "${CLAUDE_SKILL_DIR}", "$CLAUDE_SKILL_DIR", "$SKILL_DIR"):
        body = body.replace(token, skill.path)
    return body


def format_catalog(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = [_CATALOG_HEADER]
    for skill in skills[:_CATALOG_MAX_SKILLS]:
        description = skill.description.strip()
        if len(description) > _CATALOG_DESC_CHARS:
            description = description[:_CATALOG_DESC_CHARS].rstrip() + "…"
        lines.append(f"- {skill.name} : {description}")
    extra = len(skills) - _CATALOG_MAX_SKILLS
    if extra > 0:
        lines.append(f"- (+{extra} autre(s) compétence(s))")
    return "\n".join(lines)


def format_active(skill: Skill) -> str:
    body = substitute_paths(skill.body, skill).strip()
    return "\n".join([_ACTIVE_HEADER.format(name=skill.name), body, _ACTIVE_FOOTER])


def build_skills_prompt(catalog: list[Skill], active: Skill | None) -> str:
    """The full per-turn skills injection: the Level-1 catalogue plus, when a
    skill was selected for this turn, its Level-2 body."""
    blocks: list[str] = []
    catalog_text = format_catalog(catalog)
    if catalog_text:
        blocks.append(catalog_text)
    if active is not None:
        blocks.append(format_active(active))
    return "\n\n".join(blocks)
