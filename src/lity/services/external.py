from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceHealth:
    name: str
    ok: bool
    detail: str = ""

    @classmethod
    def up(cls, name: str, detail: str = "") -> ServiceHealth:
        return cls(name=name, ok=True, detail=detail)

    @classmethod
    def down(cls, name: str, detail: str = "") -> ServiceHealth:
        return cls(name=name, ok=False, detail=detail)
