from __future__ import annotations

from pathlib import Path

from lity.core.models import LoadedFile

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


class FileManager:
    def __init__(self, max_mb: float = 2.0, max_context_chars: int = 120_000):
        self.loaded_files: dict[str, LoadedFile] = {}
        self.max_mb = max_mb
        self.max_context_chars = max_context_chars
        self.working_dir: Path | None = None
        self.current_file_path: str | None = None
        self.current_file_content: str | None = None
        self.current_file_numbered_content: str | None = None
        self.last_file_load_success = False
        self.last_file_error: str | None = None
        self._recursive_files_cache: list[str] | None = None

    def set_working_dir(self, path: str | Path) -> tuple[bool, str]:
        try:
            abs_path = Path(path).expanduser().resolve()
            if not abs_path.exists() or not abs_path.is_dir():
                self.working_dir = None
                self._recursive_files_cache = None
                return False, f"Le répertoire '{path}' n'existe pas ou n'est pas un dossier."
            self.working_dir = abs_path
            self._recursive_files_cache = None
            return True, f"Répertoire de travail défini sur : {self.working_dir}"
        except Exception as exc:
            return False, f"Erreur : {exc}"

    def _resolve_path(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser()
        if not path.is_absolute() and self.working_dir:
            resolved_path = (self.working_dir / path).resolve()
        else:
            resolved_path = path.resolve()

        if self.working_dir and not _is_relative_to(resolved_path, self.working_dir):
            raise ValueError("Accès refusé : hors du répertoire de travail.")
        return resolved_path

    def load_file(self, path_str: str | Path, user_input: str | None = None) -> tuple[bool, str]:
        target = str(path_str)
        if user_input:
            success, resolution = self.resolve_file_reference(target, user_input)
            if success:
                target = str(resolution)
            else:
                if isinstance(resolution, list):
                    files_str = ", ".join(resolution)
                    message = f"Je n'ai pas trouvé '{target}'. Fichiers disponibles : ({files_str})"
                else:
                    message = str(resolution)
                self._reset_current_file(message)
                return False, message

        try:
            abs_path = self._resolve_path(target)
            if not abs_path.exists():
                self._reset_current_file(f"Le fichier '{target}' n'existe pas.")
                return False, self.last_file_error or ""
            if not abs_path.is_file():
                self._reset_current_file(f"'{target}' n'est pas un fichier.")
                return False, self.last_file_error or ""

            max_bytes = int(self.max_mb * 1024 * 1024)
            if abs_path.stat().st_size > max_bytes:
                self._reset_current_file(f"Le fichier dépasse la limite de {self.max_mb:.1f} Mo.")
                return False, self.last_file_error or ""

            content = abs_path.read_text(encoding="utf-8")
            numbered_content = self._add_line_numbers(content)
            loaded = LoadedFile(path=abs_path, content=content, numbered_content=numbered_content)
            self.loaded_files[str(abs_path)] = loaded
            self.current_file_path = str(abs_path)
            self.current_file_content = content
            self.current_file_numbered_content = numbered_content
            self.last_file_load_success = True
            self.last_file_error = None
            return (
                True,
                f"Fichier chargé : {abs_path.name}\nNombre de lignes : {len(content.splitlines())}",
            )
        except UnicodeDecodeError:
            self._reset_current_file("Le fichier n'est pas un texte UTF-8 lisible.")
            return False, self.last_file_error or ""
        except Exception as exc:
            self._reset_current_file(str(exc))
            return False, str(exc)

    def _add_line_numbers(self, content: str) -> str:
        return "\n".join(f"{index + 1}: {line}" for index, line in enumerate(content.splitlines()))

    def _reset_current_file(self, error_msg: str | None) -> None:
        self.current_file_path = None
        self.current_file_content = None
        self.current_file_numbered_content = None
        self.last_file_load_success = False
        self.last_file_error = error_msg

    def get_status_summary(self) -> str:
        status_lines = []
        if self.working_dir:
            status_lines.append(f"Répertoire actif: {self.working_dir}")
            files = self.get_available_files()
            status_lines.append(f"Fichiers disponibles: {', '.join(files) if files else 'Aucun'}")
        if self.loaded_files:
            names = [loaded.path.name for loaded in self.loaded_files.values()]
            status_lines.append(f"Fichiers chargés: {', '.join(names)}")
        if self.current_file_path:
            status_lines.append(f"Fichier actif: {self.current_file_path}")
        if self.last_file_error and (self.working_dir or self.current_file_path):
            status_lines.append(f"Erreur récente: {self.last_file_error}")
        if not status_lines:
            return ""
        return "\n[ÉTAT SYSTÈME]\n" + "\n".join(status_lines) + "\n[/ÉTAT SYSTÈME]\n"

    def get_context_for_ai(self) -> str:
        if not self.working_dir and not self.loaded_files and not self.last_file_load_success:
            return ""
        context = self.get_status_summary()
        for loaded in self.loaded_files.values():
            context += f"\n--- CONTENU DE {loaded.path.name} ---\n"
            context += loaded.numbered_content + "\n"
            if len(context) > self.max_context_chars:
                return context[: self.max_context_chars] + "\n[CONTENU TRONQUÉ]\n"
        return context

    def close_file(self, path_str: str | None = None) -> tuple[bool, str]:
        if not self.loaded_files:
            return False, "Aucun fichier n'est actuellement ouvert."

        target = path_str or self.current_file_path
        if not target:
            return False, "Aucun fichier à fermer."
        try:
            abs_path = str(self._resolve_path(target))
        except Exception:
            abs_path = str(target)

        removed = self.loaded_files.pop(abs_path, None)
        if not removed:
            return False, "Ce fichier n'est pas actuellement ouvert."

        if self.current_file_path == abs_path:
            self._reset_current_file(None)
            if self.loaded_files:
                last_loaded = next(reversed(self.loaded_files.values()))
                self.current_file_path = str(last_loaded.path)
                self.current_file_content = last_loaded.content
                self.current_file_numbered_content = last_loaded.numbered_content
                self.last_file_load_success = True
        return True, f"Fichier '{Path(abs_path).name}' fermé."

    def list_files(self) -> str:
        if self.loaded_files:
            names = ", ".join(loaded.path.name for loaded in self.loaded_files.values())
            return f"Ouverts : {names}"
        files = self.get_available_files()
        return (
            f"Fichiers disponibles : {', '.join(files)}"
            if files
            else "Aucun fichier dans le répertoire."
        )

    def refresh_files(self) -> None:
        """Invalidate the recursive file cache (e.g. after creating a file)."""
        self._recursive_files_cache = None

    def refresh_loaded_file(self, path_str: str | Path) -> bool:
        """Reload a file that is already part of the injected context.

        File writes happen through the editor, outside ``load_file``. Without a
        targeted refresh the next prompt can still inject the old cached content.
        """
        try:
            abs_path = self._resolve_path(path_str)
        except Exception:
            return False
        key = str(abs_path)
        if key not in self.loaded_files or not abs_path.exists() or not abs_path.is_file():
            return False
        try:
            content = abs_path.read_text(encoding="utf-8")
        except Exception:
            return False
        numbered_content = self._add_line_numbers(content)
        loaded = LoadedFile(path=abs_path, content=content, numbered_content=numbered_content)
        self.loaded_files[key] = loaded
        if self.current_file_path == key:
            self.current_file_content = content
            self.current_file_numbered_content = numbered_content
            self.last_file_load_success = True
            self.last_file_error = None
        return True

    def refresh_loaded_files(self, paths: list[str] | None = None) -> int:
        """Reload loaded-context cache entries; returns how many refreshed."""
        targets = paths or list(self.loaded_files.keys())
        return sum(1 for target in targets if self.refresh_loaded_file(target))

    def reset(self) -> None:
        """Clear working dir, loaded files and caches (e.g. when switching project)."""
        self.loaded_files = {}
        self.working_dir = None
        self._recursive_files_cache = None
        self._reset_current_file(None)

    def read_file_safe(self, path_str: str | Path, max_chars: int = 20_000) -> tuple[bool, str]:
        """Read a workspace file without mutating loaded context (for agent tools)."""
        try:
            abs_path = self._resolve_path(path_str)
        except Exception as exc:
            return False, str(exc)
        if not abs_path.exists() or not abs_path.is_file():
            return False, f"'{path_str}' introuvable dans le répertoire de travail."
        try:
            content = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return False, f"'{path_str}' n'est pas un texte UTF-8 lisible."
        except Exception as exc:
            return False, str(exc)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...contenu tronqué...]"
        return True, content

    def get_available_files(self, recursive: bool = False) -> list[str]:
        if not self.working_dir or not self.working_dir.exists():
            return []
        try:
            if recursive:
                if self._recursive_files_cache is not None:
                    return list(self._recursive_files_cache)
                files = sorted(
                    str(item.relative_to(self.working_dir))
                    for item in self.working_dir.rglob("*")
                    if item.is_file() and not _is_ignored(item, self.working_dir)
                )
                self._recursive_files_cache = files
                return list(files)
            return sorted(
                item.name
                for item in self.working_dir.iterdir()
                if item.is_file() and not item.name.startswith(".")
            )
        except Exception:
            return []

    def resolve_file_reference(
        self, raw_target: str, user_input: str
    ) -> tuple[bool, str | list[str]]:
        available_files = self.get_available_files(recursive=True)
        if not available_files:
            return False, "Aucun fichier disponible dans le répertoire actuel."

        if not raw_target or raw_target.strip().lower() in {"null", "none", ""}:
            return False, available_files

        user_input_lower = user_input.lower()
        clean_target = raw_target.strip("\"'")
        resolution = self._match_available_file(clean_target, user_input_lower, available_files)
        if resolution[0]:
            return resolution

        self._recursive_files_cache = None
        refreshed_files = self.get_available_files(recursive=True)
        if refreshed_files != available_files:
            resolution = self._match_available_file(clean_target, user_input_lower, refreshed_files)
            if resolution[0]:
                return resolution
            available_files = refreshed_files

        return False, available_files

    def _match_available_file(
        self,
        clean_target: str,
        user_input_lower: str,
        available_files: list[str],
    ) -> tuple[bool, str | list[str]]:
        if clean_target in available_files:
            return True, clean_target

        basename_matches = [name for name in available_files if Path(name).name == clean_target]
        if len(basename_matches) == 1:
            return True, basename_matches[0]
        if len(basename_matches) > 1:
            return False, basename_matches

        found_files = [
            name
            for name in sorted(available_files, key=len, reverse=True)
            if name.lower() in user_input_lower or Path(name).name.lower() in user_input_lower
        ]
        if len(found_files) == 1:
            return True, found_files[0]
        if len(found_files) > 1:
            return False, found_files
        return False, available_files


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_ignored(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in IGNORED_DIRS or part.startswith(".") for part in relative_parts)
