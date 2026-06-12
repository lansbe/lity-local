from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class AppResult:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass(frozen=True)
class IntentHandledResult(AppResult):
    action: str | None = None
    message: str = ""
    system_context: str = ""
    type: str = "intent_handled"


@dataclass(frozen=True)
class ErrorResult(AppResult):
    message: str
    type: str = "error"


@dataclass(frozen=True)
class TextResult(AppResult):
    content: str
    system_notification: str | None = None
    type: str = "text"


@dataclass(frozen=True)
class AiResponseResult(AppResult):
    content: str
    create_blocks: list[dict[str, str]] = field(default_factory=list)
    edit_blocks: list[dict[str, str]] = field(default_factory=list)
    system_notification: str | None = None
    type: str = "ai_response"


def result_to_dict(result: AppResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, AppResult):
        return result.to_dict()
    return result
