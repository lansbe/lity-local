from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lity.services.commands.policy import is_auto_allowed, is_dangerous


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    output: str


class CommandRunner:
    """Run workspace commands without invoking a shell."""

    def __init__(self, workdir: Path | None, *, autonomous: bool = False, timeout: int = 30):
        self.workdir = Path(workdir) if workdir else None
        self.autonomous = autonomous
        self.timeout = timeout

    def run(self, command: str, timeout: int | None = None) -> CommandResult:
        command = (command or "").strip()
        if not command:
            return CommandResult(False, "Argument 'command' requis.")
        if self.workdir is None:
            return CommandResult(False, "Aucun répertoire de travail défini.")
        danger = is_dangerous(command)
        if danger:
            return CommandResult(False, f"Commande bloquée pour raison de sécurité : {danger}.")
        if self.autonomous and not is_auto_allowed(command):
            return CommandResult(False, "Commande hors de la liste blanche du mode autonome.")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return CommandResult(False, f"Commande invalide : {exc}")
        if not argv:
            return CommandResult(False, "Commande vide.")
        limit = timeout or self.timeout
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=limit,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(False, f"Commande expirée après {limit}s.")
        except Exception as exc:
            return CommandResult(False, str(exc))
        output = ((proc.stdout or "") + (proc.stderr or "")).strip() or "(aucune sortie)"
        return CommandResult(proc.returncode == 0, f"[exit {proc.returncode}]\n{output}")
