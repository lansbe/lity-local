from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; Lity/1.0; +local-agent)"
DEFAULT_TIMEOUT = 8


def get_text(url: str, timeout: int = DEFAULT_TIMEOUT, *, quiet: bool = False) -> str | None:
    """Fetch a URL as text. Prefers httpx (if installed), falls back to urllib.

    Returns ``None`` on any failure (network down, blocked, bad status) so
    callers degrade gracefully instead of raising. ``quiet=True`` suppresses the
    failure logs — used for EXPECTED-down checks (e.g. probing a local service
    that may simply not be running) so they don't spam the log on every poll.
    """
    try:
        import httpx
    except Exception:
        httpx = None

    if httpx is not None:
        try:
            response = httpx.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            if not quiet:
                logger.info("httpx GET failed (%s): %s", url, exc)
            return None

    # Minimal fallback for environments that truly do not have httpx installed.
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        if not quiet:
            logger.info("urllib GET failed (%s): %s", url, exc)
        return None


def get_json(url: str, timeout: int = DEFAULT_TIMEOUT, *, quiet: bool = False) -> Any | None:
    """Fetch and parse a JSON endpoint, or ``None`` on any failure."""
    raw = get_text(url, timeout=timeout, quiet=quiet)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        if not quiet:
            logger.info("JSON parse failed (%s): %s", url, exc)
        return None
