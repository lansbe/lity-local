from __future__ import annotations

from lity.infrastructure.paths import AppPaths


def run_desktop(paths: AppPaths) -> int:
    try:
        from PySide6.QtWidgets import QApplication

        from lity.app.controller import AgentController
        from lity.interfaces.desktop_qt.main_window import MainWindow
    except ImportError as exc:
        print(_missing_pyside_message(str(exc)))
        return 1

    app = QApplication([])
    app.setApplicationName("Lity")
    window = MainWindow(AgentController(paths=paths))
    app.aboutToQuit.connect(window.shutdown)
    window.resize(1180, 780)
    window.show()
    return app.exec()


def _missing_pyside_message(detail: str) -> str:
    return (
        "PySide6 n'est pas installé. Lance `uv sync --extra desktop` "
        "puis `uv run lity`.\n"
        f"Détail : {detail}"
    )
