from __future__ import annotations

import re
import shutil
from pathlib import Path

from lity.services.editing.history import WorkspaceHistory
from lity.services.editing.policy import EditPolicy

# Tolerant block markers: small models drift on the exact marker shape (bold
# FILE lines, 4 chevrons instead of 7, lowercase keywords). The structure stays
# strict — only the decoration is forgiven.
_FILE_RE = re.compile(r"^[*#\->\s`]*(?:FILE|FICHIER)\s*:\s*(.+?)\s*$", re.I)
_CREATE_OPEN_RE = re.compile(r"^<{4,}\s*CREATE\b", re.I)
_CREATE_CLOSE_RE = re.compile(r"^>{4,}\s*CREATE\b", re.I)
_SEARCH_OPEN_RE = re.compile(r"^<{4,}\s*SEARCH\b", re.I)
_SEPARATOR_RE = re.compile(r"^={5,}\s*$")
_REPLACE_CLOSE_RE = re.compile(r"^>{4,}\s*REPLACE\b", re.I)


def _file_path_from(line: str) -> str | None:
    match = _FILE_RE.match(line)
    if not match:
        return None
    return match.group(1).strip().strip("`\"'* ") or None


class CodeEditor:
    def __init__(self, policy: EditPolicy | None = None, history: WorkspaceHistory | None = None):
        self.policy = policy or EditPolicy()
        self.history = history

    def parse_create_blocks(self, text: str) -> list[dict[str, str]]:
        blocks = []
        if not text:
            return blocks

        current_file = None
        in_block = False
        block_content = []
        for line in text.splitlines():
            stripped = line.strip()
            if not in_block:
                file_path = _file_path_from(stripped)
                if file_path:
                    current_file = file_path
                    continue
            if _CREATE_OPEN_RE.match(stripped):
                if current_file:
                    in_block = True
                    block_content = []
                continue
            if _CREATE_CLOSE_RE.match(stripped):
                if in_block and current_file:
                    blocks.append({"file_path": current_file, "content": "\n".join(block_content)})
                    in_block = False
                    current_file = None
                continue
            if in_block:
                block_content.append(line)
        return blocks

    def parse_search_replace_blocks(self, text: str) -> list[dict[str, str]]:
        blocks = []
        if not text:
            return blocks

        current_file = None
        state = "seeking"
        search_lines = []
        replace_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if state == "seeking":
                file_path = _file_path_from(stripped)
                if file_path:
                    current_file = file_path
                    continue
            if _SEARCH_OPEN_RE.match(stripped):
                if current_file:
                    state = "search"
                    search_lines = []
                    replace_lines = []
                continue
            if state == "search" and _SEPARATOR_RE.match(stripped):
                state = "replace"
                continue
            if state == "replace" and _REPLACE_CLOSE_RE.match(stripped):
                if current_file:
                    blocks.append(
                        {
                            "file_path": current_file,
                            "search_content": "\n".join(search_lines),
                            "replace_content": "\n".join(replace_lines),
                        }
                    )
                state = "seeking"
                current_file = None
                continue
            if state == "search":
                search_lines.append(line)
            elif state == "replace":
                replace_lines.append(line)
        return blocks

    def detect_malformed_blocks(self, text: str) -> bool:
        """True when the answer LOOKS like it proposes file blocks but none
        parse — the silent-failure case where the user previously saw nothing."""
        if not text:
            return False
        has_marker = any(
            _FILE_RE.match(line.strip())
            or _CREATE_OPEN_RE.match(line.strip())
            or _SEARCH_OPEN_RE.match(line.strip())
            for line in text.splitlines()
        )
        if not has_marker:
            return False
        return not (self.parse_create_blocks(text) or self.parse_search_replace_blocks(text))

    def apply_edit(
        self,
        file_path: str | Path,
        search_text: str,
        replace_text: str,
        working_dir: str | Path | None = None,
    ) -> tuple[bool, str]:
        try:
            path = self._resolve_write_path(file_path, working_dir)
            if not path.exists():
                return False, f"Le fichier '{file_path}' n'existe pas."

            original_content = path.read_text(encoding="utf-8")
            newline_char = "\r\n" if "\r\n" in original_content else "\n"
            search_normalized = search_text.replace("\r\n", "\n").replace("\n", newline_char)
            replace_normalized = replace_text.replace("\r\n", "\n").replace("\n", newline_char)

            occurrences = original_content.count(search_normalized)
            if occurrences > 1:
                return False, "Le bloc SEARCH apparaît plusieurs fois. Modification annulée."
            if occurrences == 1:
                new_content = original_content.replace(search_normalized, replace_normalized, 1)
            else:
                # Tolerant fallback: small local models routinely miss a trailing
                # space or an indent level. Recover the intended block when it is
                # unambiguous; otherwise fail WITH the closest real fragment so
                # the next attempt copies real text instead of guessing again.
                ok, result = self._tolerant_edit(
                    original_content, search_normalized, replace_normalized, newline_char
                )
                if not ok:
                    return False, result
                new_content = result

            policy_result = self.policy.validate_write(path, new_content)
            if not policy_result.allowed:
                return False, policy_result.message

            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
            try:
                path.write_text(new_content, encoding="utf-8")
                backup_path.unlink(missing_ok=True)
                if self.history is not None:
                    self.history.record(str(path), original_content)
                return True, "La modification a été appliquée avec succès."
            except Exception as write_error:
                shutil.move(str(backup_path), str(path))
                return (
                    False,
                    f"Erreur d'écriture : {write_error}. Le fichier original a été restauré.",
                )
        except Exception as exc:
            return False, f"Une erreur inattendue est survenue : {exc}"

    def _tolerant_edit(
        self,
        original_content: str,
        search_normalized: str,
        replace_normalized: str,
        newline_char: str,
    ) -> tuple[bool, str]:
        """Line-based tolerant matching. Returns (True, new_content) on a unique
        match, (False, actionable_message) otherwise."""
        from lity.services.editing.fuzzy_match import (
            AmbiguousMatch,
            closest_fragment,
            find_block,
            reindent,
        )

        lines = original_content.split(newline_char)
        search_lines = search_normalized.split(newline_char)
        replace_lines = replace_normalized.split(newline_char)
        try:
            match = find_block(lines, search_lines)
        except AmbiguousMatch:
            return False, "Le bloc SEARCH apparaît plusieurs fois. Modification annulée."
        if match is None:
            message = "Le bloc SEARCH n'a pas été trouvé exactement."
            fragment = closest_fragment(lines, search_lines)
            if fragment:
                message += (
                    " Le passage le plus proche dans le fichier est :\n---\n"
                    + fragment
                    + "\n---\nRecopie ce passage EXACTEMENT dans SEARCH puis réessaie."
                )
            return False, message
        new_lines = lines[: match.start] + reindent(replace_lines, match) + lines[match.end :]
        return True, newline_char.join(new_lines)

    def create_file(
        self,
        file_path: str | Path,
        content: str,
        working_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> tuple[bool, str]:
        try:
            path = self._resolve_write_path(file_path, working_dir)
            policy_result = self.policy.validate_write(path, content)
            if not policy_result.allowed:
                return False, policy_result.message
            existed = path.exists()
            if existed and not overwrite:
                return False, f"Le fichier '{path.name}' existe déjà."
            prior = path.read_text(encoding="utf-8") if existed else None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if self.history is not None:
                self.history.record(str(path), prior)
            return True, f"Le fichier '{path.name}' a été créé avec succès."
        except Exception as exc:
            return False, f"Erreur lors de la création du fichier : {exc}"

    def _resolve_write_path(
        self,
        file_path: str | Path,
        working_dir: str | Path | None = None,
    ) -> Path:
        path = Path(file_path).expanduser()
        if not path.is_absolute() and working_dir:
            resolved_path = (Path(working_dir).expanduser() / path).resolve()
        else:
            resolved_path = path.resolve()

        if working_dir:
            root = Path(working_dir).expanduser().resolve()
            try:
                resolved_path.relative_to(root)
            except ValueError:
                raise ValueError("Accès refusé : écriture hors du répertoire de travail.") from None
        return resolved_path
