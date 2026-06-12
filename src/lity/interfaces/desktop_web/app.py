from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

from lity.infrastructure.paths import AppPaths

DEV_SERVER_URL = "http://localhost:5173"


def run_desktop_web(paths: AppPaths, dev: bool = False) -> int:
    try:
        import webview

        from lity.app.controller import AgentController
        from lity.interfaces.desktop_web.api import DesktopApi
    except ImportError as exc:
        print(_missing_pywebview_message(str(exc)))
        return 1

    controller = AgentController(paths=paths)
    window_holder: dict[str, object] = {}

    def emit(event: str, payload: dict) -> None:
        window = window_holder.get("window")
        if window is None:
            return
        message = json.dumps({"event": event, "payload": payload}, ensure_ascii=True)
        with contextlib.suppress(Exception):
            window.evaluate_js(
                f"window.__lityBus && window.__lityBus(JSON.parse({json.dumps(message)}))"
            )

    def folder_picker() -> str | None:
        window = window_holder.get("window")
        if window is None:
            return None
        result = window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else str(result)

    api = DesktopApi(controller, emit=emit, folder_picker=folder_picker)

    try:
        url = DEV_SERVER_URL if dev else _bundled_index().as_uri()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    window = webview.create_window(
        "Lity",
        url=url,
        js_api=api,
        width=1200,
        height=820,
        min_size=(900, 600),
        background_color="#0b0d10",
    )
    window_holder["window"] = window
    window.events.closed += controller.shutdown
    webview.start(debug=dev)
    return 0


def _bundled_index() -> Path:
    base = _web_dist_dir() / "index.html"
    if not base.exists():
        raise FileNotFoundError(
            f"Frontend introuvable : {base}. Construis-le avec "
            "`cd frontend && npm install && npm run build`, ou lance "
            "`lity --ui web --dev` avec le serveur Vite."
        )
    return base


def _web_dist_dir() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "lity" / "interfaces" / "desktop_web" / "web_dist"
    return Path(__file__).resolve().parent / "web_dist"


def _missing_pywebview_message(detail: str) -> str:
    return (
        "pywebview n'est pas installé. Lance `uv sync --extra web` "
        "puis `uv run lity --ui web`.\n"
        f"Détail : {detail}"
    )
