from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lity.infrastructure.paths import AppPaths
from lity.infrastructure.settings import SettingsStore
from lity.services.ai.intent_router import IntentRouter
from lity.services.ai.ollama_engine import AIEngine
from lity.services.ai.prompts import DEFAULT_MODEL_NAME
from lity.services.editing.code_editor import CodeEditor
from lity.services.editing.history import WorkspaceHistory
from lity.services.files.manager import FileManager
from lity.services.memory.json_memory import MemoryManager


@dataclass
class AppServices:
    settings: SettingsStore | None
    engine: Any
    memory: Any
    files: Any
    router: Any
    editor: Any
    image_manager: Any
    video_manager: Any = None

    @classmethod
    def create(cls, paths: AppPaths) -> AppServices:
        settings = SettingsStore(paths.settings_file)
        saved_model = settings.get("selected_model", DEFAULT_MODEL_NAME)
        engine = AIEngine(model=saved_model)
        return cls(
            settings=settings,
            engine=engine,
            memory=MemoryManager(paths=paths),
            files=FileManager(),
            router=IntentRouter(model=saved_model),
            editor=CodeEditor(history=WorkspaceHistory()),
            image_manager=None,
            video_manager=None,
        )

    def with_image_manager(self, paths: AppPaths) -> AppServices:
        if self.image_manager is None:
            if self.settings is None:
                self.settings = SettingsStore(paths.settings_file)
            from lity.services.image_generation.manager import ImageGenerationManager

            self.image_manager = ImageGenerationManager(self.engine, paths, self.settings)
        return self

    def with_video_manager(self, paths: AppPaths) -> AppServices:
        if self.video_manager is None:
            if self.settings is None:
                self.settings = SettingsStore(paths.settings_file)
            from lity.services.video_generation.manager import VideoGenerationManager

            self.video_manager = VideoGenerationManager(self.engine, paths, self.settings)
        return self
