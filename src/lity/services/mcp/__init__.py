from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_servers(config_text: str) -> list[dict[str, Any]]:
    """Parse a Claude-Desktop-style ``{"mcpServers": {name: {command, args, env}}}``
    config into a clean server list. Pure + testable; ignores malformed entries."""
    try:
        data = json.loads(config_text)
    except Exception:
        return []
    raw = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return []
    servers: list[dict[str, Any]] = []
    for name, cfg in raw.items():
        if not isinstance(cfg, dict) or not cfg.get("command"):
            continue
        if cfg.get("disabled"):
            continue
        servers.append(
            {
                "name": str(name),
                "command": str(cfg["command"]),
                "args": list(cfg.get("args", []) or []),
                "env": cfg.get("env") or None,
            }
        )
    return servers


def build_mcp_manager(config_path: Path | str):
    """Build an MCPManager from a config file, or return None (graceful).

    None when: the optional ``mcp`` SDK isn't installed, the config is missing/
    empty, or no server exposes any tool. The agent then runs without MCP.
    """
    if importlib.util.find_spec("mcp") is None:
        logger.info("MCP disabled: the 'mcp' SDK is not installed.")
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    servers = _load_servers(path.read_text(encoding="utf-8"))
    if not servers:
        return None
    try:
        from lity.services.mcp.manager import MCPManager

        manager = MCPManager(servers)
    except Exception as exc:  # pragma: no cover - depends on live servers
        logger.info("MCP manager failed to start: %s", exc)
        return None
    # No tools discovered (all servers failed) → behave as if MCP is off.
    return manager if manager.tool_specs() else None
