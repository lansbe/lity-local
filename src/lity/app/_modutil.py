from __future__ import annotations

import importlib.util


def _module_available(name: str) -> bool:
    """True if an optional module can be imported (guards graceful degradation)."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
