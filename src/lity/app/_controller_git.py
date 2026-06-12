from __future__ import annotations

from typing import Any


class GitMixin:
    """Git integration for AgentController (delegates to GitService over the workdir)."""

    def _git(self) -> Any:
        from lity.services.git.service import GitService

        return GitService(getattr(self.files, "working_dir", None))

    def git_status(self) -> dict[str, Any]:
        return self._git().status()

    def git_diff(self, path: str | None = None) -> dict[str, Any]:
        return {"diff": self._git().diff(path)}

    def git_branches(self) -> dict[str, Any]:
        git = self._git()
        return {"branches": git.branches(), "current": git.current_branch()}

    def git_commit(self, message: str) -> dict[str, Any]:
        result = self._git().commit(message)
        result["status"] = self._git().status()
        return result
