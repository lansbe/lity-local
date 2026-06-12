from __future__ import annotations

from pathlib import Path

from lity.infrastructure.settings import SettingsStore


class ImageSettingsManager:
    def __init__(self, settings_store: SettingsStore):
        self.settings = settings_store
        self.init_defaults()

    def init_defaults(self) -> None:
        defaults = {
            "sd_api_url": "http://127.0.0.1:7860",
            "sd_install_dir": "",
            "sd_checkpoints_dir": "",
            "sd_selected_checkpoint": "",
            # Name of the downloaded image model used by the in-process engine
            # (a folder under …/Documents/Lity/Models/Images/). Empty = auto-pick
            # the first installed one.
            "image_selected_model": "",
        }
        for key, value in defaults.items():
            if self.settings.get(key) is None:
                self.settings.set(key, value)

    @property
    def api_url(self) -> str:
        return self.settings.get("sd_api_url", "http://127.0.0.1:7860")

    @property
    def install_dir(self) -> str:
        return self.settings.get("sd_install_dir", "")

    @property
    def checkpoints_dir(self) -> str:
        return self.settings.get("sd_checkpoints_dir", "")

    @property
    def selected_checkpoint(self) -> str:
        return self.settings.get("sd_selected_checkpoint", "")

    @property
    def selected_image_model(self) -> str:
        """Downloaded image model the in-process engine generates with."""
        return self.settings.get("image_selected_model", "")

    def set_selected_image_model(self, name: str) -> None:
        self.settings.set("image_selected_model", (name or "").strip())

    def set_api_url(self, url: str) -> None:
        self.settings.set("sd_api_url", url.strip())

    def set_install_dir(self, path: str) -> None:
        clean_path = path.strip()
        self.settings.set("sd_install_dir", clean_path)
        if clean_path and not self.checkpoints_dir:
            potential_dir = Path(clean_path) / "models" / "Stable-diffusion"
            if potential_dir.exists():
                self.set_checkpoints_dir(str(potential_dir))

    def set_checkpoints_dir(self, path: str) -> None:
        self.settings.set("sd_checkpoints_dir", path.strip())

    def set_selected_checkpoint(self, checkpoint_name: str) -> None:
        self.settings.set("sd_selected_checkpoint", checkpoint_name.strip())

    def scan_checkpoints(self) -> list[str]:
        checkpoints_path = Path(self.checkpoints_dir).expanduser()
        if checkpoints_path.exists() and checkpoints_path.is_dir():
            checkpoints = sorted(
                item.name
                for item in checkpoints_path.iterdir()
                if item.suffix.lower() in {".safetensors", ".ckpt"}
            )
            if checkpoints:
                return checkpoints
        return [
            "v1-5-pruned-emaonly.safetensors",
            "sd_xl_base_1.0.safetensors",
            "dreamshaper_8.safetensors",
            "realisticVisionV51.safetensors",
        ]
