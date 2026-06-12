from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LM_STUDIO_MODEL = "qwen2.5-coder-14b-instruct-mlx-4bit"

LM_STUDIO_RECOMMENDED_MODELS = [
    {
        "slug": "qwen2.5-coder-14b-instruct-mlx-4bit",
        "display_name": "Qwen2.5 Coder 14B MLX 4-bit",
        "note": "Driver quotidien recommandé sur Mac 16 Go.",
    },
    {
        "slug": "qwen3-8b-4bit-dwq",
        "display_name": "Qwen3 8B MLX DWQ",
        "note": "Très confortable pour agent local always-on.",
    },
    {
        "slug": "devstral-small-2507-mlx-4bit",
        "display_name": "Devstral Small 2507 MLX 4-bit",
        "note": "Qualité agentique supérieure, mémoire serrée sur 16 Go.",
    },
]

RequestFn = Callable[[str, str, dict[str, str], bytes | None, bool], Any]


class OpenAICompatibleClient:
    """Tiny OpenAI-compatible HTTP client for local runtimes like LM Studio.

    It intentionally uses the stdlib instead of the OpenAI SDK: Lity only needs
    a few local endpoints, and tests can inject ``request`` without touching the
    network.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
        api_key: str = "",
        *,
        timeout: float = 120.0,
        request: RequestFn | None = None,
    ):
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self._request = request or self._urllib_request

    def list_models(self) -> list[str]:
        data = self._json("GET", "/models")
        raw = data.get("data") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        names: list[str] = []
        for item in raw:
            if isinstance(item, dict) and item.get("id"):
                names.append(str(item["id"]))
        return names

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._chat_payload(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            response_format=response_format,
            options=options,
        )
        return self._json("POST", "/chat/completions", payload)

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        payload = self._chat_payload(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            response_format=None,
            options=options,
        )
        response = self._request(
            "POST", self._url("/chat/completions"), self._headers(), _dump(payload), True
        )
        yield from _iter_sse_json(response)

    def embed(self, text: str, model: str) -> list[float] | None:
        data = self._json("POST", "/embeddings", {"model": model, "input": text})
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            return None
        vector = rows[0].get("embedding") if isinstance(rows[0], dict) else None
        return [float(item) for item in vector] if isinstance(vector, list) else None

    def _chat_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
        response_format: dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_to_openai_message(message) for message in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = _json_schema_response_format(response_format)
        for key, value in (options or {}).items():
            mapped = "max_tokens" if key == "num_predict" else key
            if mapped == "num_ctx" or value is None:
                continue
            if mapped in {"temperature", "top_p", "top_k", "min_p", "max_tokens"}:
                payload[mapped] = value
        return payload

    def _json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = _dump(payload) if payload is not None else None
        raw = self._request(method, self._url(path), self._headers(), body, False)
        if isinstance(raw, dict):
            return raw
        data = raw.read() if hasattr(raw, "read") else raw
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data or "{}")

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _urllib_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        stream: bool,
    ) -> Any:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)  # noqa: S310 - local URL
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{exc.code} {exc.reason}: {detail}") from exc


def normalize_base_url(value: str | None) -> str:
    text = (value or DEFAULT_LM_STUDIO_BASE_URL).strip().rstrip("/")
    if text.endswith("/v1"):
        return text
    return text + "/v1"


def openai_message_content(response: Any) -> str:
    message = _first_message(response)
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def openai_tool_calls(response: Any) -> list[dict[str, Any]]:
    message = _first_message(response)
    if not isinstance(message, dict):
        return []
    return normalize_openai_tool_calls(message.get("tool_calls"))


def normalize_openai_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for call in raw_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        args = function.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if name:
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return calls


def openai_stats(response: Any, started_at: float | None = None) -> dict[str, Any]:
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    elapsed = max(time.monotonic() - started_at, 0.001) if started_at else 0.0
    return {
        "eval_count": completion,
        "prompt_eval_count": prompt,
        "context_used": prompt + completion,
        "tokens_per_sec": round(completion / elapsed, 1) if completion and elapsed else 0.0,
    }


def _first_message(response: Any) -> dict[str, Any] | None:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    return first.get("message") if isinstance(first, dict) else None


def _iter_sse_json(response: Any) -> Iterable[dict[str, Any]]:
    for raw_line in response:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else str(raw_line)
        )
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON OpenAI stream line: %s", line)
            continue
        if isinstance(data, dict):
            yield data


def _to_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role") or "user")
    content = message.get("content", "")
    images = message.get("images") or []
    if images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": str(content)}]
        for image in images:
            url = str(image)
            if not url.startswith("data:"):
                url = f"data:image/png;base64,{url}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": role, "content": parts}
    return {"role": role, "content": str(content)}


def _json_schema_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "lity_structured_response", "schema": schema, "strict": False},
    }


def _dump(payload: dict[str, Any] | None) -> bytes:
    return json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
