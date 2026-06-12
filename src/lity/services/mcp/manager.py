from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _safe_name(server: str, tool: str) -> str:
    """Namespaced, function-call-safe tool name (alphanumerics + underscore)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", f"mcp_{server}_{tool}")


class MCPManager:
    """Connects to local MCP servers (stdio) and exposes their tools to the
    SYNCHRONOUS ``AgentLoop``.

    The MCP SDK is async, so we run ONE background event loop and ONE long-lived
    task per server: that task opens the session and is the ONLY thing that
    touches it (calls are funnelled to it through an ``asyncio.Queue``), which
    avoids the classic anyio "cancel scope in a different task" error you get if
    you call the session from another coroutine. Fully guarded — a server that
    fails to start is skipped and the agent keeps working without it. 100% local
    (servers are local subprocesses), no API key.
    """

    def __init__(
        self,
        servers: list[dict[str, Any]],
        connect_timeout: float = 20.0,
        call_timeout: float = 60.0,
    ):
        self._servers = servers
        self._call_timeout = call_timeout
        self._specs: list[dict[str, Any]] = []
        self._route: dict[str, tuple[str, str]] = {}  # public name → (server, raw tool)
        self._queues: dict[str, asyncio.Queue] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop)
        try:
            future.result(timeout=connect_timeout)
        except Exception as exc:  # pragma: no cover - depends on live servers
            logger.info("MCP connect timed out / failed: %s", exc)

    # ----------------------------------------------------- sync public API
    def tool_specs(self) -> list[dict[str, Any]]:
        return list(self._specs)

    def call(self, public_name: str, args: dict[str, Any]) -> tuple[bool, str]:
        route = self._route.get(public_name)
        if route is None:
            return False, f"Outil MCP inconnu : {public_name}"
        _server, tool = route
        queue = self._queues.get(_server)
        if queue is None:
            return False, f"Serveur MCP « {_server} » indisponible."

        async def _enqueue() -> Any:
            done = self._loop.create_future()
            await queue.put((tool, args, done))
            return await done

        try:
            result = asyncio.run_coroutine_threadsafe(_enqueue(), self._loop).result(
                timeout=self._call_timeout
            )
        except Exception as exc:  # pragma: no cover - depends on live servers
            return False, f"Échec de l'outil MCP {public_name} : {exc}"
        return _format_result(result)

    def close(self) -> None:  # pragma: no cover - lifecycle
        for queue in self._queues.values():
            asyncio.run_coroutine_threadsafe(queue.put(None), self._loop)
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------- async internals
    def _run_loop(self) -> None:  # pragma: no cover - background thread
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all(self) -> None:  # pragma: no cover - depends on SDK/servers
        readies = []
        for server in self._servers:
            name = str(server.get("name") or server.get("command") or "server")
            queue: asyncio.Queue = asyncio.Queue()
            self._queues[name] = queue
            ready = self._loop.create_future()
            asyncio.ensure_future(self._serve(name, server, queue, ready))
            readies.append(ready)
        await asyncio.gather(*readies, return_exceptions=True)

    async def _serve(self, name, server, queue, ready) -> None:  # pragma: no cover - SDK/servers
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=server["command"],
                args=list(server.get("args", []) or []),
                env=server.get("env") or None,
            )
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                listed = await session.list_tools()
                self._register(name, listed.tools)
                if not ready.done():
                    ready.set_result(True)
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    tool, args, done = item
                    try:
                        done.set_result(await session.call_tool(tool, args or {}))
                    except Exception as exc:
                        if not done.done():
                            done.set_exception(exc)
        except Exception as exc:
            logger.info("MCP server %s unavailable: %s", name, exc)
            if not ready.done():
                ready.set_exception(exc)

    def _register(self, name, tools) -> None:  # pragma: no cover - depends on SDK
        for tool in tools or []:
            raw = getattr(tool, "name", "")
            if not raw:
                continue
            public = _safe_name(name, raw)
            self._route[public] = (name, raw)
            schema = getattr(tool, "inputSchema", None)
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            self._specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": public,
                        "description": (getattr(tool, "description", "") or f"Outil MCP {raw}")[
                            :400
                        ],
                        "parameters": schema,
                    },
                }
            )


def _format_result(result: Any) -> tuple[bool, str]:
    """Flatten an MCP CallToolResult into (ok, text)."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
        elif isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    text = "\n".join(parts) or "(réponse MCP vide)"
    ok = not bool(getattr(result, "isError", False))
    return ok, text
