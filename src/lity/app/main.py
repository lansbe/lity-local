from __future__ import annotations

import argparse
import sys

from lity.infrastructure.logging import configure_logging
from lity.infrastructure.paths import AppPaths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lity")
    parser.add_argument(
        "--ui",
        choices=["web", "qt", "console"],
        help="Choose the interface: web (pywebview), qt (PySide6) or console.",
    )
    parser.add_argument(
        "--console", action="store_true", help="Shortcut for --ui console (CLI mode)."
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Web UI dev mode: load the Vite dev server instead of the bundled build.",
    )
    parser.add_argument("--home", help="Override the app data/config root directory.")
    args = parser.parse_args(argv)

    paths = AppPaths.create(home_override=args.home)
    configure_logging(paths.log_dir)

    ui = args.ui or ("console" if args.console else "qt")

    if ui == "console":
        from lity.app.controller import AgentController
        from lity.interfaces.cli.console import run_console

        run_console(AgentController(paths=paths))
        return 0

    if ui == "web":
        from lity.interfaces.desktop_web.app import run_desktop_web

        return run_desktop_web(paths, dev=args.dev)

    from lity.interfaces.desktop_qt.app import run_desktop

    return run_desktop(paths)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
