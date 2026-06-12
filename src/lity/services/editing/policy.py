from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(openai|anthropic|github|hf|huggingface|aws|stripe)[_\- ]?(api[_\- ]?)?key\s*="
    ),
    re.compile(r"(?i)\b(database|postgres|mysql|mongo(db)?|redis)[_\- ]?url\s*="),
    re.compile(r"(?i)\b(jwt|session|webhook|signing)[_\- ]?secret\s*="),
    re.compile(r"(?i)\b(private|access|refresh|auth|bearer)[_\- ]?token\s*="),
    re.compile(r"(?i)\b[A-Z0-9_]*(secret|password|private[_\- ]?key)[A-Z0-9_]*\s*="),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
]

PROTECTED_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
PROTECTED_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    message: str = ""


class EditPolicy:
    def validate_write(self, file_path: Path, content: str) -> PolicyResult:
        if _is_protected_path(file_path):
            return PolicyResult(False, "Écriture refusée : le fichier cible ressemble à un secret.")
        if _contains_secret(content):
            return PolicyResult(False, "Écriture refusée : le contenu ressemble à un secret.")
        return PolicyResult(True, "")


def _contains_secret(content: str) -> bool:
    return any(pattern.search(content) for pattern in SECRET_PATTERNS)


def _is_protected_path(file_path: Path) -> bool:
    name = file_path.name.lower()
    if name in PROTECTED_NAMES:
        return True
    return file_path.suffix.lower() in PROTECTED_SUFFIXES
