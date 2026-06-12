from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class GitService:
    """Thin wrapper around the `git` CLI scoped to a working directory."""

    def __init__(self, workdir: str | Path | None):
        self.workdir = Path(workdir) if workdir else None

    def _run(self, args: list[str], timeout: int = 15) -> tuple[int, str, str]:
        if self.workdir is None:
            return 1, "", "Aucun répertoire de travail."
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except FileNotFoundError:
            return 1, "", "git introuvable."
        except Exception as exc:  # pragma: no cover - defensive
            return 1, "", str(exc)

    def is_repo(self) -> bool:
        code, out, _ = self._run(["rev-parse", "--is-inside-work-tree"])
        return code == 0 and out.strip() == "true"

    def current_branch(self) -> str:
        code, out, _ = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return out.strip() if code == 0 else ""

    def branches(self) -> list[str]:
        code, out, _ = self._run(["branch", "--format=%(refname:short)"])
        return [line.strip() for line in out.splitlines() if line.strip()] if code == 0 else []

    def status(self) -> dict[str, Any]:
        if not self.is_repo():
            return {"is_repo": False, "branch": "", "files": []}
        _code, out, _err = self._run(["status", "--porcelain"])
        files = [
            {"status": line[:2].strip(), "path": line[3:]}
            for line in out.splitlines()
            if len(line) >= 4
        ]
        return {"is_repo": True, "branch": self.current_branch(), "files": files}

    def diff(self, path: str | None = None) -> str:
        args = ["diff"]
        if path:
            args.append(path)
        _code, out, _err = self._run(args)
        return out

    def commit(self, message: str) -> dict[str, Any]:
        if not message.strip():
            return {"ok": False, "message": "Message de commit requis."}
        if not self.is_repo():
            return {"ok": False, "message": "Pas un dépôt git."}
        add_code, _out, add_err = self._run(["add", "-A"])
        if add_code != 0:
            return {"ok": False, "message": add_err.strip() or "git add a échoué."}
        code, out, err = self._run(["commit", "-m", message])
        if code == 0:
            return {"ok": True, "message": out.strip() or "Commit créé."}
        return {"ok": False, "message": (err or out).strip() or "Rien à committer."}
