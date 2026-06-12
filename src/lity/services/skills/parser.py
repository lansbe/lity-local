from __future__ import annotations

import logging
from typing import Any

from lity.services.skills.models import (
    Skill,
    clamp_body,
    clamp_description,
    normalize_name,
)

logger = logging.getLogger(__name__)

_FENCE = "---"


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split a SKILL.md into (frontmatter_yaml, body).

    A skill with no ``---`` fenced header is all body (name comes from the
    folder); this mirrors how Claude Code / Codex treat a bare command file.
    """
    stripped = text.lstrip("﻿")  # tolerate a UTF-8 BOM
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        return "", stripped.strip()
    for index in range(1, len(lines)):
        if lines[index].strip() == _FENCE:
            front = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :]).strip()
            return front, body
    # An unclosed header → treat the whole thing as body rather than losing it.
    return "", stripped.strip()


def parse_frontmatter(front: str) -> dict[str, Any]:
    """Parse the YAML header. Uses PyYAML when present (it is, transitively),
    falling back to a minimal key/value parser so a skill never fails to load
    just because YAML is unavailable."""
    if not front.strip():
        return {}
    try:
        import yaml

        data = yaml.safe_load(front)
        return data if isinstance(data, dict) else {}
    except Exception:  # pragma: no cover - exercised only without/with broken yaml
        return _minimal_yaml(front)


def _minimal_yaml(front: str) -> dict[str, Any]:
    """Tiny fallback: top-level ``key: value`` and inline ``[a, b]`` lists.

    Deliberately small — it covers the documented skill frontmatter fields
    (name, description, when_to_use, license, allowed-tools, triggers) and
    ignores nested structures it cannot represent."""
    out: dict[str, Any] = {}
    for line in front.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("'\"")
        if value.startswith("[") and value.endswith("]"):
            out[key] = [
                item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()
            ]
        elif value:
            out[key] = value
    return out


def _as_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        # Accept "Bash(python:*) Read" or "a, b" or "a b".
        parts = [piece.strip() for piece in value.replace(",", " ").split()]
        return tuple(piece for piece in parts if piece)
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def build_skill(text: str, *, folder_name: str, source: str, path: str) -> Skill | None:
    """Turn raw SKILL.md text into a :class:`Skill`, or None when there is no
    usable description (a skill the router could never trigger)."""
    front, body = split_frontmatter(text)
    meta = parse_frontmatter(front)

    name = normalize_name(_as_str(meta.get("name")), folder_name)
    description = clamp_description(_as_str(meta.get("description")))
    when_to_use = clamp_description(_as_str(meta.get("when_to_use") or meta.get("when-to-use")))
    if not description and not when_to_use and not body:
        return None
    if not description:
        # Fall back to the first body line so the skill is still discoverable.
        description = clamp_description(body.splitlines()[0] if body else name)

    raw_meta = meta.get("metadata")
    metadata = {str(k): str(v) for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}

    return Skill(
        name=name,
        description=description,
        body=clamp_body(body),
        when_to_use=when_to_use,
        triggers=_as_list(meta.get("triggers") or meta.get("trigger")),
        allowed_tools=_as_list(meta.get("allowed-tools") or meta.get("allowed_tools")),
        metadata=metadata,
        source=source if source in ("builtin", "user") else "user",
        path=path,
    )
