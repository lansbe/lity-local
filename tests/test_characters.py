from __future__ import annotations

from pathlib import Path

from lity.app.controller import AgentController
from lity.app.services import AppServices
from lity.infrastructure.paths import AppPaths
from lity.interfaces.desktop_web.api import DesktopApi
from lity.services.characters import (
    CHARACTER_EMOTIONS,
    CharacterImageGenerator,
    CharacterStore,
)
from lity.services.memory.conversation_store import ConversationStore
from lity.services.memory.json_memory import MemoryManager


def test_character_store_creates_local_profile(tmp_path: Path) -> None:
    store = CharacterStore(tmp_path / "characters")

    profile = store.create(
        {
            "name": "Mira",
            "description": "portrait d'une ingénieure calme aux cheveux courts",
            "gender": "femme",
            "style": "illustration semi-réaliste",
            "instructions": "Réponds avec précision et douceur.",
            "voice": "fr_FR-siwis-medium",
            "image_model": "sd15",
            "seed": 1234,
        }
    )

    assert profile["id"]
    assert profile["name"] == "Mira"
    assert profile["description"] == "portrait d'une ingénieure calme aux cheveux courts"
    assert profile["instructions"] == "Réponds avec précision et douceur."
    assert profile["seed"] == 1234
    assert set(profile["emotions"]) == set(CHARACTER_EMOTIONS)
    assert all(item["image_path"] == "" for item in profile["emotions"].values())
    assert (tmp_path / "characters" / profile["id"] / "profile.json").is_file()

    reloaded = CharacterStore(tmp_path / "characters")
    assert [item["id"] for item in reloaded.list()] == [profile["id"]]


def test_character_store_saves_generated_emotion_images_inside_profile(tmp_path: Path) -> None:
    store = CharacterStore(tmp_path / "characters")
    profile = store.create({"name": "Noor", "description": "portrait studio"})
    source = tmp_path / "render.png"
    source.write_bytes(b"png-bytes")

    updated = store.save_emotion_image(profile["id"], "happy", source)

    image_path = Path(updated["emotions"]["happy"]["image_path"])
    assert image_path.name == "happy.png"
    assert image_path.parent == tmp_path / "characters" / profile["id"] / "images"
    assert image_path.read_bytes() == b"png-bytes"


def test_conversation_store_tracks_active_character(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations")
    conversation_id = store.create_conversation()["id"]

    assert store.get_character_id(conversation_id) == ""
    assert store.set_character_id(conversation_id, "character-123")
    assert store.get_character_id(conversation_id) == "character-123"
    assert store.get_meta(conversation_id)["character_id"] == "character-123"
    assert store.set_character_id(conversation_id, "")
    assert store.get_character_id(conversation_id) == ""


class _FakeImageManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.calls: list[dict] = []

    def is_active(self):
        return False

    def execute_generation(self, params):
        emotion = params["character_emotion"]
        self.calls.append(params)
        path = self.output_dir / f"{emotion}.png"
        path.write_bytes(f"image:{emotion}".encode())
        return {
            "type": "image_generation_result",
            "content": {"image_path": str(path), "params": params},
            "message": "ok",
        }


def test_character_generator_creates_requested_emotion_pack(tmp_path: Path) -> None:
    store = CharacterStore(tmp_path / "characters")
    profile = store.create(
        {
            "name": "Ilan",
            "description": "homme avec lunettes rondes, veste verte",
            "style": "portrait cinématique",
            "image_model": "sdxl",
            "seed": 77,
        }
    )
    image_manager = _FakeImageManager(tmp_path)
    generator = CharacterImageGenerator(store, image_manager)

    result = generator.generate(profile["id"], ["neutral", "happy"])

    assert result["ok"] is True
    assert result["generated"] == ["neutral", "happy"]
    assert len(image_manager.calls) == 2
    assert image_manager.calls[0]["checkpoint"] == "sdxl"
    assert image_manager.calls[0]["seed"] == 77
    assert "Ilan" in image_manager.calls[0]["prompt"]
    assert "neutral expression" in image_manager.calls[0]["prompt"]
    assert "warm smile" in image_manager.calls[1]["prompt"]
    saved = store.get(profile["id"])
    assert Path(saved["emotions"]["happy"]["image_path"]).read_bytes() == b"image:happy"


class _Engine:
    model = "fake"

    def __init__(self):
        self.system_prompt_extra = ""
        self.temperature = None

    def get_installed_models(self):
        return ["fake"]


class _Files:
    working_dir = None
    loaded_files: dict = {}

    def get_context_for_ai(self):
        return ""


class _Router:
    model = "fake"

    def process_intent(self, *_args):
        return {"handled": False, "action": "none", "message": "", "system_context": ""}


class _Editor:
    def parse_create_blocks(self, _text):
        return []

    def parse_search_replace_blocks(self, _text):
        return []


def _controller(tmp_path: Path, image_manager=None) -> AgentController:
    paths = AppPaths.create(home_override=tmp_path)
    services = AppServices(
        settings=None,
        engine=_Engine(),
        memory=MemoryManager(paths=paths),
        files=_Files(),
        router=_Router(),
        editor=_Editor(),
        image_manager=image_manager,
    )
    return AgentController(paths=paths, services=services)


def test_controller_applies_active_character_to_current_conversation(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    created = controller.create_character(
        {
            "name": "Mira",
            "description": "portrait studio",
            "instructions": "Réponds comme Mira, en phrases courtes.",
        }
    )
    character_id = created["character"]["id"]

    assert controller.set_active_character(character_id)["ok"] is True
    assert controller.get_active_character()["id"] == character_id
    controller._apply_conversation_instructions()
    assert "Réponds comme Mira" in controller.engine.system_prompt_extra

    previous_conversation = controller.active_conversation_id
    controller.memory.add_message("user", "matérialise cette conversation")
    controller.new_conversation()
    assert controller.get_active_character() is None
    controller.switch_conversation(previous_conversation)
    assert controller.get_active_character()["id"] == character_id


def test_desktop_api_returns_character_images_as_data_urls(tmp_path: Path) -> None:
    image_manager = _FakeImageManager(tmp_path)
    controller = _controller(tmp_path, image_manager=image_manager)
    api = DesktopApi(controller)
    created = api.create_character({"name": "Noor", "description": "portrait studio"})
    character_id = created["character"]["id"]

    result = api.generate_character_emotions(character_id, ["happy"])

    assert result["ok"] is True
    assert result["character"]["emotions"]["happy"]["image"].startswith("data:image/png;base64,")
    assert api.set_conversation_character(character_id)["active_character"]["id"] == character_id
    assert api.get_state()["active_character"]["id"] == character_id
