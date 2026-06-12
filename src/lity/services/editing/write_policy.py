from __future__ import annotations

from enum import StrEnum


class WriteMode(StrEnum):
    REVIEWED = "reviewed"
    AUTONOMOUS = "autonomous"


def mode_from_yolo(enabled: bool) -> WriteMode:
    return WriteMode.AUTONOMOUS if enabled else WriteMode.REVIEWED


def write_mode_label(mode: WriteMode) -> str:
    if mode is WriteMode.AUTONOMOUS:
        return "Mode autonome : l'agent peut écrire directement dans le répertoire de travail."
    return "Mode revu : l'IA propose des changements que l'interface doit valider."
