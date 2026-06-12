from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

APP_NAME = "Lity"
APP_AUTHOR = "Lity"
ENV_HOME = "LITY_HOME"


def _platform_path(kind: str, app_name: str, app_author: str) -> Path:
    try:
        from platformdirs import (
            user_cache_path,
            user_config_path,
            user_data_path,
            user_log_path,
        )

        factories = {
            "cache": user_cache_path,
            "config": user_config_path,
            "data": user_data_path,
            "logs": user_log_path,
        }
        return factories[kind](app_name, app_author, ensure_exists=True)
    except Exception:
        root = Path.home() / ".lity"
        return root / kind


@dataclass(frozen=True)
class AppPaths:
    data_dir: Path
    config_dir: Path
    cache_dir: Path
    log_dir: Path
    resources_dir: Path

    @classmethod
    def create(
        cls,
        home_override: Path | None = None,
        app_name: str = APP_NAME,
        app_author: str = APP_AUTHOR,
    ) -> AppPaths:
        env_home = os.environ.get(ENV_HOME)
        home = Path(home_override).expanduser() if home_override else None
        if home is None and env_home:
            home = Path(env_home).expanduser()

        # Everything lives under ~/Documents/Lity so it is always in the same,
        # user-visible place on every machine (macOS/Windows/Linux), unless a
        # home override / env var explicitly points elsewhere (tests, --home).
        if home is None:
            home = Path.home() / "Documents" / "Lity"
        data_dir = home / "data"
        config_dir = home / "config"
        cache_dir = home / "cache"
        log_dir = home / "logs"

        try:
            resources_dir = Path(resources.files("lity.resources"))
        except Exception:
            resources_dir = Path(__file__).resolve().parents[1] / "resources"

        paths = cls(
            data_dir=data_dir,
            config_dir=config_dir,
            cache_dir=cache_dir,
            log_dir=log_dir,
            resources_dir=resources_dir,
        )
        paths.ensure()
        return paths

    def ensure(self) -> None:
        for directory in (self.data_dir, self.config_dir, self.cache_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_videos_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.image_models_dir.mkdir(parents=True, exist_ok=True)
        self.video_models_dir.mkdir(parents=True, exist_ok=True)
        self.characters_dir.mkdir(parents=True, exist_ok=True)

    @property
    def settings_file(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def conversations_dir(self) -> Path:
        return self.data_dir / "conversations"

    @property
    def vector_index_file(self) -> Path:
        return self.data_dir / "vector_index.json"

    @property
    def memory_index_file(self) -> Path:
        return self.data_dir / "memory_index.json"

    @property
    def fact_index_file(self) -> Path:
        return self.data_dir / "fact_index.json"

    @property
    def mcp_config_file(self) -> Path:
        return self.config_dir / "mcp.json"

    @property
    def skills_dir(self) -> Path:
        """User-authored skills ("Compétences"), one folder per skill. Sits
        next to Models so it is easy to find and drop folders into."""
        return self.app_root / "skills"

    @property
    def builtin_skills_dir(self) -> Path:
        """Skills packaged with the app (read-only built-ins)."""
        return self.resources_dir / "skills"

    @property
    def user_profile_file(self) -> Path:
        return self.data_dir / "user_profile.json"

    @property
    def assistant_profile_file(self) -> Path:
        return self.data_dir / "assistant_profile.json"

    @property
    def facts_file(self) -> Path:
        return self.data_dir / "long_term_facts.json"

    @property
    def output_images_dir(self) -> Path:
        return self.data_dir / "output_images"

    @property
    def output_videos_dir(self) -> Path:
        """Where locally generated video clips (MP4) are saved."""
        return self.data_dir / "output_videos"

    @property
    def app_root(self) -> Path:
        """The …/Documents/Lity root (parent of the data/config/… subdirs)."""
        return self.data_dir.parent

    @property
    def models_dir(self) -> Path:
        return self.app_root / "Models"

    @property
    def image_models_dir(self) -> Path:
        """Where non-Ollama image checkpoints are downloaded, one folder per model."""
        return self.models_dir / "Images"

    @property
    def video_models_dir(self) -> Path:
        """Where local video-generation models are downloaded, one folder per model."""
        return self.models_dir / "Videos"

    @property
    def characters_dir(self) -> Path:
        """User-created characters, profiles, generated portraits and emotion packs."""
        return self.app_root / "characters"

    @property
    def temp_recording_file(self) -> Path:
        return self.cache_dir / "temp_recording.wav"
