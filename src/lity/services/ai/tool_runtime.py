from __future__ import annotations

from typing import Any


def missing_required_arg(spec: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Return the first required tool arg that is absent or blank."""
    for required in spec.get("parameters", {}).get("required", []):
        value = args.get(required)
        if value is None or (isinstance(value, str) and not value.strip()):
            return required
    return None
