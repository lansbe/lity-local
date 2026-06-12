from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedFile:
    path: Path
    content: str
    numbered_content: str = ""

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class FileEdit:
    file_path: str
    search_content: str
    replace_content: str


@dataclass(frozen=True)
class FileCreate:
    file_path: str
    content: str
