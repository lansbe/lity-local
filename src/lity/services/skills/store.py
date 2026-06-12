from __future__ import annotations

import logging
import shutil
from pathlib import Path

from lity.services.skills.models import Skill, slugify
from lity.services.skills.parser import build_skill

logger = logging.getLogger(__name__)

_SKILL_FILES = ("SKILL.md", "skill.md")


class SkillStore:
    """Filesystem-backed catalogue of skills.

    Scans a read-only ``builtin_dir`` (skills packaged with the app) and a
    writable ``user_dir`` (``~/Documents/Lity/skills``). A user skill shadows a
    built-in of the same name, mirroring the personal-over-bundled precedence
    Claude Code / Codex use. The catalogue is re-read whenever the directories
    change on disk (live reload), so dropping a folder in works without a
    restart. No model, no network — pure filesystem.
    """

    def __init__(self, user_dir: Path | str, builtin_dir: Path | str | None = None):
        self.user_dir = Path(user_dir)
        self.builtin_dir = Path(builtin_dir) if builtin_dir else None
        self._cache: dict[str, Skill] = {}
        self._fingerprint: tuple | None = None

    # ------------------------------------------------------------------ scan
    def _roots(self) -> list[tuple[Path, str]]:
        roots: list[tuple[Path, str]] = []
        if self.builtin_dir and self.builtin_dir.exists():
            roots.append((self.builtin_dir, "builtin"))
        roots.append((self.user_dir, "user"))
        return roots

    @staticmethod
    def _skill_file(entry: Path) -> Path | None:
        if entry.is_dir():
            for candidate in _SKILL_FILES:
                target = entry / candidate
                if target.is_file():
                    return target
            return None
        # A flat top-level *.md is treated as a single-file skill (convenience),
        # excluding obvious non-skill docs.
        if entry.suffix.lower() == ".md" and entry.stem.lower() not in ("readme", "license"):
            return entry
        return None

    def _entries(self) -> list[tuple[Path, str, str]]:
        """(skill_file, folder_name, source) for everything on disk."""
        found: list[tuple[Path, str, str]] = []
        for root, source in self._roots():
            try:
                children = sorted(root.iterdir())
            except OSError:
                continue
            for entry in children:
                skill_file = self._skill_file(entry)
                if skill_file is None:
                    continue
                folder = entry.name if entry.is_dir() else entry.stem
                found.append((skill_file, folder, source))
        return found

    def _compute_fingerprint(self, entries: list[tuple[Path, str, str]]) -> tuple:
        signature = []
        for skill_file, _folder, source in entries:
            try:
                stat = skill_file.stat()
                signature.append((str(skill_file), source, int(stat.st_mtime), stat.st_size))
            except OSError:
                continue
        return tuple(signature)

    def reload(self) -> None:
        entries = self._entries()
        cache: dict[str, Skill] = {}
        for skill_file, folder, source in entries:
            try:
                text = skill_file.read_text(encoding="utf-8")
            except OSError as exc:
                logger.info("Skill illisible %s : %s", skill_file, exc)
                continue
            skill = build_skill(
                text, folder_name=folder, source=source, path=str(skill_file.parent)
            )
            if skill is None:
                continue
            # User skills are scanned last, so they overwrite a same-named
            # built-in here (personal-over-bundled precedence).
            existing = cache.get(skill.name)
            if existing is not None and existing.source == "user" and source == "builtin":
                continue
            cache[skill.name] = skill
        self._cache = cache
        self._fingerprint = self._compute_fingerprint(entries)

    def _ensure_fresh(self) -> None:
        entries = self._entries()
        fingerprint = self._compute_fingerprint(entries)
        if fingerprint != self._fingerprint:
            self.reload()

    # --------------------------------------------------------------- queries
    def list(self) -> list[Skill]:
        self._ensure_fresh()
        return sorted(self._cache.values(), key=lambda skill: skill.name)

    def get(self, name: str) -> Skill | None:
        self._ensure_fresh()
        return self._cache.get(slugify(name))

    # ----------------------------------------------------------------- crud
    def create(
        self,
        name: str,
        description: str,
        body: str,
        *,
        when_to_use: str = "",
        triggers: list[str] | None = None,
    ) -> tuple[bool, str, Skill | None]:
        slug = slugify(name)
        if not slug:
            return False, "Nom de compétence invalide.", None
        target_dir = self.user_dir / slug
        # Reject either layout for the same slug — a folder OR a flat <slug>.md —
        # so a folder skill can't shadow a same-named flat file (which would then
        # orphan the folder on delete).
        if (target_dir / "SKILL.md").exists() or self._flat_file_for(slug) is not None:
            return False, f"La compétence « {slug} » existe déjà.", None
        if not (description.strip() or body.strip()):
            return False, "Donne au moins une description ou un contenu.", None
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "SKILL.md").write_text(
                _render_skill_md(slug, description, body, when_to_use, triggers or []),
                encoding="utf-8",
            )
        except OSError as exc:
            return False, f"Écriture impossible : {exc}", None
        self.reload()
        return True, f"Compétence « {slug} » créée.", self.get(slug)

    def delete(self, name: str) -> tuple[bool, str]:
        skill = self.get(name)
        if skill is None:
            return False, "Compétence introuvable."
        if skill.builtin:
            return False, "Les compétences intégrées ne peuvent pas être supprimées."
        folder = Path(skill.path)
        try:
            if folder.is_dir() and folder != self.user_dir and folder.parent == self.user_dir:
                # Proper one-folder-per-skill layout under the user dir.
                shutil.rmtree(folder)
            else:
                # Flat single-file skill dropped directly in the user dir: remove
                # exactly the file that produced this skill (never the dir itself).
                target = self._flat_file_for(skill.name)
                if target is None:
                    return False, "Fichier de la compétence introuvable."
                target.unlink(missing_ok=True)
        except OSError as exc:
            return False, f"Suppression impossible : {exc}"
        self.reload()
        return True, f"Compétence « {skill.name} » supprimée."

    def _flat_file_for(self, slug: str) -> Path | None:
        """Find the top-level ``<user_dir>/*.md`` whose parsed skill name matches
        ``slug`` (filename casing/accents may differ from the slug)."""
        try:
            children = sorted(self.user_dir.iterdir())
        except OSError:
            return None
        for entry in children:
            if entry.is_file() and entry.suffix.lower() == ".md" and slugify(entry.stem) == slug:
                return entry
        return None


def _render_skill_md(
    name: str, description: str, body: str, when_to_use: str, triggers: list[str]
) -> str:
    lines = ["---", f"name: {name}", f"description: {_yaml_scalar(description)}"]
    if when_to_use.strip():
        lines.append(f"when_to_use: {_yaml_scalar(when_to_use)}")
    clean_triggers = [str(item).strip() for item in triggers if str(item).strip()]
    if clean_triggers:
        # Escape each trigger as a YAML scalar (collapses newlines, quotes
        # YAML-significant chars) so a trigger value can't inject frontmatter.
        rendered = ", ".join(_yaml_scalar(item) for item in clean_triggers)
        lines.append(f"triggers: [{rendered}]")
    lines.append("---")
    lines.append("")
    lines.append(body.strip() or f"# {name}")
    lines.append("")
    return "\n".join(lines)


def _yaml_scalar(value: str) -> str:
    """Single-line YAML scalar: quote when the text carries YAML-significant
    characters so the frontmatter round-trips cleanly."""
    text = " ".join(str(value).split())
    if any(ch in text for ch in ":#\"'") or text != value.strip():
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text
