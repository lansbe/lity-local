from __future__ import annotations

from lity.infrastructure.settings import SettingsStore


class VideoSettingsManager:
    """Persists which downloaded video model the in-process engine generates with.

    Deliberately minimal next to ``ImageSettingsManager``: video generation has
    no external WebUI, so there is a single setting — the active model name (a
    folder under ``…/Documents/Lity/Models/Videos/``). Empty = auto-pick the
    first installed one.
    """

    def __init__(self, settings_store: SettingsStore):
        self.settings = settings_store
        self.init_defaults()

    def init_defaults(self) -> None:
        if self.settings.get("video_selected_model") is None:
            self.settings.set("video_selected_model", "")

    @property
    def selected_video_model(self) -> str:
        """Downloaded video model the in-process engine generates with."""
        return self.settings.get("video_selected_model", "")

    def set_selected_video_model(self, name: str) -> None:
        self.settings.set("video_selected_model", (name or "").strip())
