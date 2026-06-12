from __future__ import annotations

from typing import Any


def summarize_receipts(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a provenance summary for tool calls in the latest agent run."""
    if not items:
        return None
    copied = [dict(item) for item in items]
    return {
        "items": copied,
        "grounded": any(bool(item.get("ok")) for item in copied),
        "tools_used": sorted(
            {str(item.get("name", "")) for item in copied if str(item.get("name", ""))}
        ),
    }
