import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.app.controller import AgentController
from lity.app.services import AppServices
from lity.infrastructure.paths import AppPaths
from lity.interfaces.desktop_web.api import DesktopApi
from lity.services.editing.code_editor import CodeEditor
from lity.services.editing.history import WorkspaceHistory
from lity.services.files.manager import FileManager
from lity.services.memory.json_memory import MemoryManager


class FakeStreamingEngine:
    model = "fake-model"

    def __init__(self):
        self.last_images = "UNSET"
        self.calls: list[str] = []

    def get_installed_models(self):
        return ["fake-model", "other:latest"]

    def stream_response(self, *args, **kwargs):
        self.calls.append("stream_response")
        self.last_images = kwargs.get("images")
        yield from ["Bon", "jour", " !"]

    def get_response(self, *args, **kwargs):
        self.calls.append("get_response")
        return "Bonjour !"

    def extract_fact(self, message):
        self.calls.append("extract_fact")
        return None

    def get_models_detailed(self):
        return [{"name": "fake-model", "size": 123456}]

    def pull_model(self, name, on_progress=None):
        if on_progress:
            on_progress({"status": "downloading", "completed": 50, "total": 100})
        return {"ok": True, "message": f"{name} téléchargé"}

    def delete_model(self, name):
        return {"ok": True, "message": f"{name} supprimé"}

    def model_info(self, name):
        return {"parameters": "num_ctx 4096", "template": "{{ .Prompt }}", "details": {}}


class FakeAgentEngine:
    model = "fake-model"

    def __init__(self, script):
        self.script = list(script)

    def get_installed_models(self):
        return ["fake-model"]

    def extract_fact(self, message):
        return None

    def _build_messages(self, context, **kwargs):
        return [{"role": "system", "content": "sys"}] + [
            {"role": message["role"], "content": message["content"]} for message in context
        ]

    def chat_with_tools(self, messages, tools=None, **kwargs):
        if self.script:
            return self.script.pop(0)
        return {"content": "fin", "tool_calls": []}


class FakeFiles:
    loaded_files: dict = {}
    working_dir = None
    current_file_path = None

    def get_context_for_ai(self):
        return ""

    def set_working_dir(self, path):
        self.working_dir = Path(path)
        return True, f"Répertoire de travail défini sur : {path}"

    def load_file(self, path, user_input=None):
        return True, f"Fichier chargé : {path}"

    def list_files(self):
        return "Aucun fichier"

    def close_file(self, target=None):
        return True, "Fichier fermé."


class FakeRouter:
    model = "fake-model"

    def process_intent(self, user_input, file_manager):
        return {"handled": False, "action": "none", "message": "", "system_context": ""}


class FakeEditor:
    def parse_create_blocks(self, text):
        return []

    def parse_search_replace_blocks(self, text):
        return []


def _build_api(tmp):
    paths = AppPaths.create(home_override=Path(tmp))
    services = AppServices(
        settings=None,
        engine=FakeStreamingEngine(),
        memory=MemoryManager(paths=paths),
        files=FakeFiles(),
        router=FakeRouter(),
        editor=FakeEditor(),
        image_manager=None,
    )
    controller = AgentController(paths=paths, services=services)
    events: list[tuple[str, dict]] = []
    api = DesktopApi(controller, emit=lambda event, payload: events.append((event, payload)))
    return api, events


def _build_api_with_real_files(tmp: str, workdir: Path):
    paths = AppPaths.create(home_override=Path(tmp))
    services = AppServices(
        settings=None,
        engine=FakeStreamingEngine(),
        memory=MemoryManager(paths=paths),
        files=FileManager(),
        router=FakeRouter(),
        editor=CodeEditor(history=WorkspaceHistory()),
        image_manager=None,
    )
    controller = AgentController(paths=paths, services=services)
    api = DesktopApi(controller)
    api.set_workdir(str(workdir))
    return api


class WorkspaceApiTests(unittest.TestCase):
    def test_apply_create_writes_file_and_lists_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            api = _build_api_with_real_files(tmp, workdir)

            result = api.apply_create({"file_path": "hello.py", "content": "print('hi')\n"})

            self.assertTrue(result["success"], result["message"])
            self.assertTrue((workdir / "hello.py").exists())
            self.assertIn("hello.py", result["files"])

    def test_apply_edit_modifies_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            (workdir / "a.py").write_text("a = 1\n", encoding="utf-8")
            api = _build_api_with_real_files(tmp, workdir)

            result = api.apply_edit(
                {"file_path": "a.py", "search_content": "a = 1", "replace_content": "a = 2"}
            )

            self.assertTrue(result["success"], result["message"])
            self.assertIn("a = 2", (workdir / "a.py").read_text(encoding="utf-8"))

    def test_apply_edit_refreshes_loaded_file_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            (workdir / "a.py").write_text("a = 1\n", encoding="utf-8")
            api = _build_api_with_real_files(tmp, workdir)
            self.assertTrue(api.load_context_file("a.py")["success"])

            result = api.apply_edit(
                {"file_path": "a.py", "search_content": "a = 1", "replace_content": "a = 2"}
            )

            self.assertTrue(result["success"], result["message"])
            self.assertEqual(api.controller.files.current_file_content, "a = 2\n")
            self.assertIn("1: a = 2", api.controller.files.get_context_for_ai())

    def test_load_context_file_and_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            api = _build_api_with_real_files(tmp, workdir)

            files = api.list_workspace_files()
            self.assertIn("note.txt", files["files"])

            loaded = api.load_context_file("note.txt")
            self.assertTrue(loaded["success"], loaded["message"])
            self.assertEqual(len(loaded["loaded"]), 1)
            self.assertEqual(loaded["loaded"][0]["name"], "note.txt")

            closed = api.close_context_file("note.txt")
            self.assertTrue(closed["success"], closed["message"])
            self.assertEqual(closed["loaded"], [])


class DraftConversationTests(unittest.TestCase):
    def test_empty_conversation_is_a_hidden_reused_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            # Fresh start: the default conversation is an empty draft, not listed.
            self.assertEqual(api.list_conversations(), [])
            draft_id = api.controller.active_conversation_id

            # Clicking "new" while empty reuses the same draft (no pile-up).
            meta = api.new_conversation()
            self.assertEqual(meta["id"], draft_id)
            self.assertEqual(api.list_conversations(), [])

            # The first message materialises it into the sidebar.
            api.send_message("première question")
            self.assertEqual(len(api.list_conversations()), 1)

            # A new conversation after a real one is a fresh hidden draft.
            api.new_conversation()
            self.assertEqual(len(api.list_conversations()), 1)

    def test_first_message_generates_ai_title_in_background(self):
        class TitlingEngine(FakeStreamingEngine):
            def generate_title(self, text, model_name=None):
                return "Titre généré"

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=TitlingEngine(),
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            events: list[tuple[str, dict]] = []
            api = DesktopApi(
                controller, emit=lambda event, payload: events.append((event, payload))
            )

            api.send_message("Explique-moi les décorateurs Python")
            if api._title_thread is not None:
                api._title_thread.join(timeout=2)  # avoid racing temp-dir cleanup

            title_events = [payload for event, payload in events if event == "title_update"]
            self.assertTrue(title_events, "expected a title_update bus event")
            self.assertEqual(title_events[-1]["title"], "Titre généré")
            # The sidebar listing reflects the AI title (not the derived one).
            self.assertEqual(api.list_conversations()[0]["title"], "Titre généré")

    def test_ai_title_does_not_override_user_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("première question")
            cid = api.controller.active_conversation_id
            api.rename_conversation(cid, "Titre choisi")
            if api._title_thread is not None:
                api._title_thread.join(timeout=2)
            # A late background title must not clobber the user's choice.
            self.assertFalse(api.controller.set_ai_title(cid, "Titre IA"))
            self.assertEqual(api.list_conversations()[0]["title"], "Titre choisi")


class ProjectsApiTests(unittest.TestCase):
    def test_workdir_is_restored_per_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj_a = Path(tmp) / "a"
            proj_a.mkdir()
            (proj_a / "x.py").write_text("ax", encoding="utf-8")
            proj_b = Path(tmp) / "b"
            proj_b.mkdir()
            (proj_b / "y.py").write_text("by", encoding="utf-8")

            api = _build_api_with_real_files(tmp, proj_a)  # workdir proj_a on the draft
            api.send_message("dans A")  # materialise la conversation A (workdir proj_a)
            first = api.controller.active_conversation_id

            api.new_conversation()  # nouveau brouillon, hérite de proj_a
            api.send_message("dans B")  # materialise B
            second = api.controller.active_conversation_id
            self.assertNotEqual(second, first)
            api.set_workdir(str(proj_b))  # B -> project B

            back_to_first = api.switch_conversation(first)
            self.assertEqual(back_to_first["workdir"], str(proj_a.resolve()))
            self.assertIn("x.py", back_to_first["files"])

            to_second = api.switch_conversation(second)
            self.assertEqual(to_second["workdir"], str(proj_b.resolve()))
            self.assertIn("y.py", to_second["files"])


class UndoApiTests(unittest.TestCase):
    def test_undo_restores_edit_and_removes_created_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            (workdir / "a.py").write_text("a = 1\n", encoding="utf-8")
            api = _build_api_with_real_files(tmp, workdir)

            # Edit an existing file, then create a new one.
            api.apply_edit(
                {"file_path": "a.py", "search_content": "a = 1", "replace_content": "a = 2"}
            )
            create_result = api.apply_create({"file_path": "new.py", "content": "x = 0\n"})
            self.assertEqual(create_result["change_count"], 2)
            self.assertTrue((workdir / "new.py").exists())

            # Undo the create -> file removed.
            undo1 = api.undo_change()
            self.assertTrue(undo1["ok"])
            self.assertFalse((workdir / "new.py").exists())
            self.assertEqual(undo1["change_count"], 1)

            # Undo the edit -> original content restored.
            undo2 = api.undo_change()
            self.assertTrue(undo2["ok"])
            self.assertEqual((workdir / "a.py").read_text(encoding="utf-8"), "a = 1\n")
            self.assertEqual(undo2["change_count"], 0)

            # Nothing left to undo.
            self.assertFalse(api.undo_change()["ok"])


class AgentApiTests(unittest.TestCase):
    def test_set_flags_reflected_in_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            self.assertFalse(api.get_state()["agent_mode"])
            self.assertEqual(api.set_agent_mode(True), {"agent_mode": True})
            self.assertEqual(api.set_allow_commands(True), {"allow_commands": True})
            state = api.get_state()
            self.assertTrue(state["agent_mode"])
            self.assertTrue(state["allow_commands"])

    def test_agent_mode_emits_steps_and_returns_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            (workdir / "a.py").write_text("x = 1\n", encoding="utf-8")
            paths = AppPaths.create(home_override=Path(tmp))
            engine = FakeAgentEngine(
                [
                    {"content": None, "tool_calls": [{"name": "list_files", "arguments": {}}]},
                    {"content": "Voici l'analyse du projet.", "tool_calls": []},
                ]
            )
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=FileManager(),
                router=FakeRouter(),
                editor=CodeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            events: list[tuple[str, dict]] = []
            api = DesktopApi(
                controller, emit=lambda event, payload: events.append((event, payload))
            )
            api.set_agent_mode(True)
            api.set_workdir(str(workdir))

            payload = api.send_message("analyse le projet")

            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(payload["content"], "Voici l'analyse du projet.")
            steps = [payload for event, payload in events if event == "step"]
            kinds = [step["kind"] for step in steps]
            self.assertIn("tool_call", kinds)
            self.assertIn("tool_result", kinds)

    def test_agent_no_workspace_hides_file_tools_but_keeps_web(self):
        # With web available by default, the agent decides for itself: without a
        # workspace, file tools (list_files/read_file/search) are NOT offered, but
        # web tools ARE — so the model can search without any manual "Web" toggle.
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))

            class HybridEngine:
                model = "fake-model"

                def __init__(self):
                    self.tools_seen = None

                def get_installed_models(self):
                    return ["fake-model"]

                def extract_fact(self, message):
                    return None

                def _build_messages(self, context, **kwargs):
                    return [{"role": "system", "content": "sys"}]

                def chat_with_tools(self, messages, tools=None, **kwargs):
                    self.tools_seen = [tool["function"]["name"] for tool in (tools or [])]
                    return {"content": "Réponse directe.", "tool_calls": []}

            engine = HybridEngine()
            files = FakeFiles()
            files.working_dir = None
            files.loaded_files = {}
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=files,
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            api = DesktopApi(controller)
            api.set_agent_mode(True)
            api.set_web_search(True)  # web is off by default → opt in explicitly

            payload = api.send_message("Quel est le dernier modèle de Claude ?")

            self.assertEqual(payload["type"], "ai_response")
            self.assertIsNotNone(engine.tools_seen)  # loop entered (autonomous)
            self.assertIn("web_search", engine.tools_seen)
            self.assertIn("fetch_url", engine.tools_seen)
            self.assertNotIn("list_files", engine.tools_seen)  # no workspace → no file tools


class MessageActionTests(unittest.TestCase):
    def test_regenerate_replaces_last_assistant_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("bonjour")
            api.regenerate()

            messages = api.get_messages()
            self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
            self.assertEqual(messages[0]["content"], "bonjour")

    def test_edit_and_resend_rewrites_last_user_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("bonjor")  # typo
            api.edit_and_resend("bonjour les amis")

            messages = api.get_messages()
            self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
            self.assertEqual(messages[0]["content"], "bonjour les amis")

    def test_send_message_forwards_images_to_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("décris cette image", images=["data:image/png;base64,AAAA"])
            self.assertEqual(api.controller.engine.last_images, ["data:image/png;base64,AAAA"])

    def test_agent_mode_with_images_uses_vision_not_tools(self):
        # Regression: an attached image must reach the model even in agent mode.
        # The tool-loop can't forward images, so attachments route to the vision
        # path. A workdir is set so the no-workspace gate is NOT what bypasses it.
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))

            class VisionHybridEngine:
                model = "fake-model"

                def __init__(self):
                    self.tool_loop_calls = 0
                    self.last_images = "UNSET"

                def get_installed_models(self):
                    return ["fake-model"]

                def extract_fact(self, message):
                    return None

                def _build_messages(self, context, **kwargs):
                    return [{"role": "system", "content": "sys"}]

                def chat_with_tools(self, messages, tools=None, **kwargs):
                    self.tool_loop_calls += 1
                    return {"content": "via outils", "tool_calls": []}

                def stream_response(self, *args, **kwargs):
                    self.last_images = kwargs.get("images")
                    yield from ["Je ", "vois ", "l'image."]

            engine = VisionHybridEngine()
            files = FakeFiles()
            files.working_dir = Path(tmp)  # workspace ready → not the gate under test
            files.loaded_files = {}
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=files,
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            api = DesktopApi(controller)
            api.set_agent_mode(True)

            payload = api.send_message("décris cette image", images=["data:image/png;base64,AAAA"])

            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(engine.last_images, ["data:image/png;base64,AAAA"])  # forwarded
            self.assertEqual(engine.tool_loop_calls, 0)  # tool-loop bypassed for vision

    def _routing_api(self, tmp, model, installed, caps=None):
        """Build an API whose engine records the model_name it was asked to use.

        ``caps`` maps model name → vision verdict (True/False/None), mimicking
        Ollama's reported capabilities. ``None`` (or an absent entry) makes the
        engine return None, so the controller falls back to the name heuristic."""
        paths = AppPaths.create(home_override=Path(tmp))

        class RoutingEngine:
            def __init__(self):
                self.model = model
                self.last_model_name = "UNSET"
                self.last_images = "UNSET"

            def get_installed_models(self):
                return list(installed)

            def supports_vision(self, name=None):
                return (caps or {}).get(name or self.model)

            def extract_fact(self, message):
                return None

            def stream_response(self, *args, **kwargs):
                self.last_model_name = kwargs.get("model_name")
                self.last_images = kwargs.get("images")
                yield from ["ok"]

        engine = RoutingEngine()
        services = AppServices(
            settings=None,
            engine=engine,
            memory=MemoryManager(paths=paths),
            files=FakeFiles(),
            router=FakeRouter(),
            editor=FakeEditor(),
            image_manager=None,
        )
        controller = AgentController(paths=paths, services=services)
        return DesktopApi(controller), engine

    def test_image_routes_to_installed_vision_model(self):
        # Text-only active model + an installed vision model → the turn is routed
        # to the vision model so the image is actually read.
        with tempfile.TemporaryDirectory() as tmp:
            api, engine = self._routing_api(tmp, "qwen2.5:7b", ["qwen2.5:7b", "llava:7b"])
            payload = api.send_message("décris", images=["data:image/png;base64,AAAA"])
            self.assertEqual(engine.last_model_name, "llava:7b")
            self.assertEqual(engine.last_images, ["data:image/png;base64,AAAA"])
            self.assertIn("llava", payload.get("system_notification") or "")

    def test_image_without_vision_model_warns_user(self):
        # No vision model installed → no silent drop: the user is told why.
        with tempfile.TemporaryDirectory() as tmp:
            api, engine = self._routing_api(tmp, "qwen2.5:7b", ["qwen2.5:7b"])
            payload = api.send_message("décris", images=["data:image/png;base64,AAAA"])
            self.assertIsNone(engine.last_model_name)  # active model kept
            self.assertIn("ne peut pas voir les images", payload.get("system_notification") or "")

    def test_vision_active_model_is_not_rerouted(self):
        # Already a vision model → no override, no warning noise.
        with tempfile.TemporaryDirectory() as tmp:
            api, engine = self._routing_api(tmp, "llava:7b", ["llava:7b"])
            payload = api.send_message("décris", images=["data:image/png;base64,AAAA"])
            self.assertIsNone(engine.last_model_name)
            self.assertIsNone(payload.get("system_notification"))

    def test_ollama_capability_trumps_name_for_custom_vision_model(self):
        # A custom-named multimodal build that the name
        # heuristic can't recognise is still trusted via Ollama's capabilities:
        # no reroute, no spurious "can't see images" warning.
        with tempfile.TemporaryDirectory() as tmp:
            api, engine = self._routing_api(
                tmp,
                "custom-vision-build:latest",
                ["custom-vision-build:latest"],
                caps={"custom-vision-build:latest": True},
            )
            payload = api.send_message("décris", images=["data:image/png;base64,AAAA"])
            self.assertIsNone(engine.last_model_name)  # current model already sees images
            self.assertIsNone(payload.get("system_notification"))
            self.assertEqual(engine.last_images, ["data:image/png;base64,AAAA"])

    def test_ollama_capability_overrides_optimistic_name(self):
        # A name that LOOKS multimodal but whose build dropped the vision tower
        # (Ollama reports no vision) must NOT be trusted: the user is warned.
        with tempfile.TemporaryDirectory() as tmp:
            api, engine = self._routing_api(
                tmp,
                "gemma4-textonly:latest",
                ["gemma4-textonly:latest"],
                caps={"gemma4-textonly:latest": False},
            )
            payload = api.send_message("décris", images=["data:image/png;base64,AAAA"])
            self.assertIsNone(engine.last_model_name)
            self.assertIn("ne peut pas voir les images", payload.get("system_notification") or "")

    def test_image_bypasses_workspace_file_gate(self):
        # Regression: "analyse cette image" matches the workspace-file gate, but
        # an attached image IS the context — it must reach the model, not error
        # with "Aucun fichier n'est chargé".
        with tempfile.TemporaryDirectory() as tmp:
            api, engine = self._routing_api(tmp, "llava:7b", ["llava:7b"])
            payload = api.send_message(
                "Analyse cette image dis-moi c'est quoi", images=["data:image/png;base64,AAAA"]
            )
            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(engine.last_images, ["data:image/png;base64,AAAA"])

    def test_file_gate_still_fires_without_image(self):
        # The gate must still protect "analyse ce fichier" when nothing is loaded
        # and no image is attached.
        with tempfile.TemporaryDirectory() as tmp:
            api, _engine = self._routing_api(tmp, "qwen2.5:7b", ["qwen2.5:7b"])
            payload = api.send_message("analyse ce fichier stp")
            self.assertEqual(payload["type"], "error")
            self.assertIn("Aucun fichier", payload.get("message") or "")

    def test_attached_image_is_persisted_in_history(self):
        # The attachment is stored on the user message: it redisplays on reload
        # and stays in context on follow-up turns.
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("décris cette image", images=["data:image/png;base64,AAAA"])
            messages = api.get_messages()
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[0]["images"], ["data:image/png;base64,AAAA"])


class SettingsApiTests(unittest.TestCase):
    def _services(self, paths, settings):
        from lity.services.ai.ollama_engine import AIEngine

        return AppServices(
            settings=settings,
            engine=AIEngine(model="x"),
            memory=MemoryManager(paths=paths),
            files=FileManager(),
            router=FakeRouter(),
            editor=CodeEditor(),
            image_manager=None,
        )

    def test_custom_instructions_applied_persisted_and_reloaded(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            controller = AgentController(
                paths=paths, services=self._services(paths, SettingsStore(paths.settings_file))
            )
            api = DesktopApi(controller)

            result = api.update_settings(
                {
                    "custom_instructions": "Réponds toujours en pirate.",
                    "embedding_model": "mxbai-embed-large",
                    "default_yolo": True,
                }
            )
            self.assertEqual(result["custom_instructions"], "Réponds toujours en pirate.")
            self.assertEqual(controller.engine.system_prompt_extra, "Réponds toujours en pirate.")
            self.assertEqual(
                SettingsStore(paths.settings_file).get("embedding_model"), "mxbai-embed-large"
            )

            # A fresh controller applies persisted instructions on init.
            reloaded = AgentController(
                paths=paths, services=self._services(paths, SettingsStore(paths.settings_file))
            )
            self.assertEqual(reloaded.engine.system_prompt_extra, "Réponds toujours en pirate.")
            # And DesktopApi picks up default_yolo at startup.
            self.assertTrue(DesktopApi(reloaded).get_state()["yolo"])

    def test_web_toggle_persists_to_settings(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            controller = AgentController(
                paths=paths, services=self._services(paths, SettingsStore(paths.settings_file))
            )
            api = DesktopApi(controller)

            api.set_web_search(True)
            # Persisted so health, the Settings checkbox and a restart stay in sync.
            self.assertTrue(SettingsStore(paths.settings_file).get("web_search_enabled"))
            self.assertTrue(controller.get_settings()["web_search_enabled"])

            api.set_web_search(False)
            self.assertFalse(SettingsStore(paths.settings_file).get("web_search_enabled"))


class CodexProviderApiTests(unittest.TestCase):
    class FakeLoginProcess:
        returncode = 0

        def __init__(self):
            self.timeout = None

        def communicate(self, timeout=None):
            self.timeout = timeout
            return "Logged in using ChatGPT\n", None

        def kill(self):
            pass

    class FakeCodex:
        def __init__(self, content: str = "Réponse depuis le compte ChatGPT via Codex."):
            self.calls = []
            self.login_calls = 0
            self.login_process = CodexProviderApiTests.FakeLoginProcess()
            self.content = content

        def status(self):
            return {
                "available": True,
                "authenticated": True,
                "message": "Logged in using ChatGPT",
            }

        def run_prompt(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return {
                "ok": True,
                "content": self.content,
                "message": "ok",
            }

        def start_login(self):
            self.login_calls += 1
            return {
                "ok": True,
                "message": "Connexion Codex lancée.",
                "process": self.login_process,
            }

        def model_catalog(self):
            return {
                "ok": True,
                "default_model": "gpt-5.5",
                "message": "2 modèle(s) Codex disponible(s).",
                "models": [
                    {
                        "slug": "gpt-5.5",
                        "display_name": "GPT-5.5",
                        "description": "Frontier model.",
                        "default_reasoning_level": "medium",
                        "supported_reasoning_levels": [
                            {"effort": "low", "description": "Fast"},
                            {"effort": "medium", "description": "Balanced"},
                            {"effort": "high", "description": "Deep"},
                        ],
                        "priority": 10,
                    },
                    {
                        "slug": "gpt-5.4-mini",
                        "display_name": "GPT-5.4-Mini",
                        "description": "Fast model.",
                        "default_reasoning_level": "low",
                        "supported_reasoning_levels": [
                            {"effort": "low", "description": "Fast"},
                            {"effort": "medium", "description": "Balanced"},
                        ],
                        "priority": 20,
                    },
                ],
            }

    class TrackingLocalEngine(FakeStreamingEngine):
        def embed(self, text, model_name=None):
            self.calls.append("embed")
            return [1.0]

        def generate_structured(self, *args, **kwargs):
            self.calls.append("generate_structured")
            return {"standalone": "requête réécrite"}

        def generate_title(self, text, model_name=None):
            self.calls.append("generate_title")
            return "Titre local"

    def test_codex_settings_are_persisted_and_reported(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            controller = AgentController(
                paths=paths, services=SettingsApiTests()._services(paths, settings)
            )
            api = DesktopApi(controller)

            result = api.update_settings(
                {
                    "chat_provider": "codex",
                    "codex_model": "gpt-5.5",
                    "codex_reasoning_effort": "high",
                }
            )

            self.assertEqual(result["chat_provider"], "codex")
            self.assertEqual(result["codex_model"], "gpt-5.5")
            self.assertEqual(result["codex_reasoning_effort"], "high")
            self.assertEqual(SettingsStore(paths.settings_file).get("chat_provider"), "codex")
            self.assertEqual(api.get_state()["chat_provider"], "codex")

    def test_codex_status_is_exposed_without_reading_auth_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            fake = self.FakeCodex()
            api.controller._codex_cli = fake

            self.assertEqual(api.codex_status()["message"], "Logged in using ChatGPT")

    def test_codex_models_exposes_cli_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.controller._codex_cli = self.FakeCodex()

            result = api.codex_models()

            self.assertTrue(result["ok"], result["message"])
            self.assertEqual(result["default_model"], "gpt-5.5")
            self.assertEqual(result["models"][0]["slug"], "gpt-5.5")
            self.assertEqual(
                [level["effort"] for level in result["models"][0]["supported_reasoning_levels"]],
                ["low", "medium", "high"],
            )

    def test_codex_login_starts_background_flow_and_emits_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)
            fake = self.FakeCodex()
            api.controller._codex_cli = fake

            result = api.codex_login()

            self.assertTrue(result["ok"], result["message"])
            self.assertTrue(result["running"])
            self.assertEqual(fake.login_calls, 1)
            deadline = time.time() + 2
            while not any(event == "codex_login" for event, _payload in events):
                if time.time() > deadline:
                    self.fail("expected codex_login event")
                time.sleep(0.01)
            payload = [payload for event, payload in events if event == "codex_login"][-1]
            self.assertTrue(payload["ok"], payload["message"])
            self.assertTrue(payload["status"]["authenticated"])
            self.assertEqual(fake.login_process.timeout, 600)

    def test_codex_provider_uses_codex_cli_for_chat_turns(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "codex")
            settings.set("codex_model", "gpt-5.5")
            settings.set("codex_reasoning_effort", "medium")
            services = SettingsApiTests()._services(paths, settings)
            controller = AgentController(paths=paths, services=services)
            fake = self.FakeCodex()
            controller._codex_cli = fake
            events: list[tuple[str, dict]] = []
            api = DesktopApi(
                controller, emit=lambda event, payload: events.append((event, payload))
            )
            api.set_workdir(str(workdir))

            payload = api.send_message("Analyse ce projet avec Codex")

            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(payload["content"], "Réponse depuis le compte ChatGPT via Codex.")
            self.assertEqual(
                [p["content"] for e, p in events if e == "chunk"], [payload["content"]]
            )
            messages = api.get_messages()
            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertEqual(fake.calls[0][1]["model"], "gpt-5.5")
            self.assertEqual(fake.calls[0][1]["reasoning_effort"], "medium")
            self.assertEqual(fake.calls[0][1]["workdir"], workdir.resolve())

    def test_codex_provider_does_not_call_local_engine_or_inject_local_memory(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "codex")
            settings.set("codex_model", "gpt-5.5")
            services = AppServices(
                settings=settings,
                engine=self.TrackingLocalEngine(),
                memory=MemoryManager(paths=paths),
                files=FileManager(),
                router=FakeRouter(),
                editor=CodeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            controller.memory.process_extracted_fact(
                {
                    "categorie": "user_profile",
                    "cle": "prenom",
                    "valeur": "Alex",
                }
            )
            controller.memory.set_fact("stack", "Lity utilise un RAG local via Ollama")
            fake = self.FakeCodex()
            controller._codex_cli = fake
            api = DesktopApi(controller)
            api.set_workdir(str(workdir))

            payload = api.send_message("je préfère les réponses courtes")
            time.sleep(0.05)

            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(services.engine.calls, [])
            prompt = fake.calls[0][0]
            self.assertNotIn("Alex", prompt)
            self.assertNotIn("Lity utilise un RAG local via Ollama", prompt)
            self.assertNotIn("MÉMOIRE DES CONVERSATIONS PASSÉES", prompt)
            self.assertIn("lity.services.codex_rag", prompt)

    def test_codex_memory_blocks_are_saved_and_hidden_from_chat(self):
        from lity.infrastructure.settings import SettingsStore

        codex_content = (
            "Bien reçu.\n\n"
            "[LITY_MEMORY]\n"
            '{"categorie":"long_term_facts","cle":"style","valeur":"'
            "L'utilisateur préfère les réponses courtes."
            '"}\n'
            "[/LITY_MEMORY]\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "codex")
            services = AppServices(
                settings=settings,
                engine=self.TrackingLocalEngine(),
                memory=MemoryManager(paths=paths),
                files=FileManager(),
                router=FakeRouter(),
                editor=CodeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            controller._codex_cli = self.FakeCodex(content=codex_content)
            api = DesktopApi(controller)

            payload = api.send_message("note ça")

            self.assertEqual(payload["content"], "Bien reçu.")
            self.assertNotIn("LITY_MEMORY", api.get_messages()[-1]["content"])
            self.assertEqual(
                api.get_memory()["facts"]["style"],
                "L'utilisateur préfère les réponses courtes.",
            )
            self.assertEqual(services.engine.calls, [])


class ClaudeProviderApiTests(unittest.TestCase):
    class FakeLoginProcess:
        returncode = 0

        def __init__(self):
            self.timeout = None

        def communicate(self, timeout=None):
            self.timeout = timeout
            return "Logged in as you@example.com\n", None

        def kill(self):
            pass

    class FakeClaude:
        def __init__(self, content: str = "Réponse depuis le compte Claude."):
            self.calls = []
            self.login_calls = 0
            self.login_process = ClaudeProviderApiTests.FakeLoginProcess()
            self.content = content

        def status(self):
            return {
                "available": True,
                "authenticated": True,
                "message": "Logged in as you@example.com",
            }

        def run_prompt(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return {
                "ok": True,
                "content": self.content,
                "usage": {
                    "cost_usd": 0.5,
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "by_model": {
                        "claude-opus-4-8": {
                            "input_tokens": 1000,
                            "output_tokens": 200,
                            "cost_usd": None,
                        }
                    },
                },
                "message": "ok",
            }

        def start_login(self):
            self.login_calls += 1
            return {
                "ok": True,
                "message": "Connexion Claude lancée.",
                "process": self.login_process,
            }

        def model_catalog(self):
            from lity.services.claude_cli import ClaudeCliClient

            return ClaudeCliClient().model_catalog()

    class TrackingLocalEngine(FakeStreamingEngine):
        def embed(self, text, model_name=None):
            self.calls.append("embed")
            return [1.0]

        def generate_structured(self, *args, **kwargs):
            self.calls.append("generate_structured")
            return {"standalone": "requête réécrite"}

        def generate_title(self, text, model_name=None):
            self.calls.append("generate_title")
            return "Titre local"

    def test_claude_settings_are_persisted_and_reported(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            controller = AgentController(
                paths=paths, services=SettingsApiTests()._services(paths, settings)
            )
            api = DesktopApi(controller)

            result = api.update_settings(
                {
                    "chat_provider": "claude",
                    "claude_model": "opus",
                    "claude_effort": "high",
                }
            )

            self.assertEqual(result["chat_provider"], "claude")
            self.assertEqual(result["claude_model"], "opus")
            self.assertEqual(result["claude_effort"], "high")
            self.assertEqual(SettingsStore(paths.settings_file).get("chat_provider"), "claude")
            self.assertEqual(api.get_state()["chat_provider"], "claude")

    def test_claude_status_is_exposed_without_reading_auth_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.controller._claude_cli = self.FakeClaude()

            self.assertEqual(api.claude_status()["message"], "Logged in as you@example.com")

    def test_claude_models_exposes_static_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)

            result = api.claude_models()

            self.assertTrue(result["ok"], result["message"])
            self.assertEqual(result["default_model"], "claude-opus-4-8")
            slugs = [model["slug"] for model in result["models"]]
            self.assertIn("claude-fable-5", slugs)
            self.assertIn("claude-opus-4-8", slugs)
            self.assertIn("claude-haiku-4-5", slugs)

    def test_claude_login_starts_background_flow_and_emits_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)
            fake = self.FakeClaude()
            api.controller._claude_cli = fake

            result = api.claude_login()

            self.assertTrue(result["ok"], result["message"])
            self.assertTrue(result["running"])
            self.assertEqual(fake.login_calls, 1)
            deadline = time.time() + 2
            while not any(event == "claude_login" for event, _payload in events):
                if time.time() > deadline:
                    self.fail("expected claude_login event")
                time.sleep(0.01)
            payload = [payload for event, payload in events if event == "claude_login"][-1]
            self.assertTrue(payload["ok"], payload["message"])
            self.assertTrue(payload["status"]["authenticated"])
            self.assertEqual(fake.login_process.timeout, 600)

    def test_claude_provider_uses_claude_cli_for_chat_turns(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "claude")
            settings.set("claude_model", "opus")
            settings.set("claude_effort", "high")
            services = SettingsApiTests()._services(paths, settings)
            controller = AgentController(paths=paths, services=services)
            fake = self.FakeClaude()
            controller._claude_cli = fake
            events: list[tuple[str, dict]] = []
            api = DesktopApi(
                controller, emit=lambda event, payload: events.append((event, payload))
            )
            api.set_workdir(str(workdir))

            payload = api.send_message("Analyse ce projet avec Claude")

            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(payload["content"], "Réponse depuis le compte Claude.")
            self.assertEqual(
                [p["content"] for e, p in events if e == "chunk"], [payload["content"]]
            )
            messages = api.get_messages()
            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertEqual(fake.calls[0][1]["model"], "opus")
            self.assertEqual(fake.calls[0][1]["reasoning_effort"], "high")
            self.assertEqual(fake.calls[0][1]["workdir"], workdir.resolve())

            # The turn's usage is tracked per model and exposed via api.usage().
            usage = api.usage()
            self.assertEqual(usage["claude"]["turns"], 1)
            self.assertEqual(usage["claude"]["cost_usd"], 0.5)
            self.assertEqual(usage["claude"]["total_tokens"], 1200)
            by_model = {row["model"]: row for row in usage["claude"]["by_model"]}
            self.assertIn("claude-opus-4-8", by_model)
            self.assertEqual(by_model["claude-opus-4-8"]["output_tokens"], 200)

    def test_claude_memory_blocks_are_saved_and_hidden_from_chat(self):
        from lity.infrastructure.settings import SettingsStore

        claude_content = (
            "Bien reçu.\n\n"
            "[LITY_MEMORY]\n"
            '{"categorie":"long_term_facts","cle":"style","valeur":"'
            "L'utilisateur préfère les réponses courtes."
            '"}\n'
            "[/LITY_MEMORY]\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "claude")
            services = AppServices(
                settings=settings,
                engine=self.TrackingLocalEngine(),
                memory=MemoryManager(paths=paths),
                files=FileManager(),
                router=FakeRouter(),
                editor=CodeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            controller._claude_cli = self.FakeClaude(content=claude_content)
            api = DesktopApi(controller)

            payload = api.send_message("note ça")

            self.assertEqual(payload["content"], "Bien reçu.")
            self.assertNotIn("LITY_MEMORY", api.get_messages()[-1]["content"])
            self.assertEqual(
                api.get_memory()["facts"]["style"],
                "L'utilisateur préfère les réponses courtes.",
            )
            self.assertEqual(services.engine.calls, [])


class GrokProviderApiTests(unittest.TestCase):
    class FakeLoginProcess:
        returncode = 0

        def __init__(self):
            self.timeout = None

        def communicate(self, timeout=None):
            self.timeout = timeout
            return "Device auth complete\n", None

        def kill(self):
            pass

    class FakeGrok:
        def __init__(self, content: str = "Réponse depuis le compte Grok."):
            self.calls = []
            self.login_calls: list[bool] = []
            self.login_process = GrokProviderApiTests.FakeLoginProcess()
            self.content = content

        def status(self):
            return {"available": True, "authenticated": True, "message": "Grok est connecté."}

        def run_prompt(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return {
                "ok": True,
                "content": self.content,
                "usage": {
                    "cost_usd": None,
                    "input_tokens": 800,
                    "output_tokens": 150,
                    "by_model": {},
                },
                "message": "ok",
            }

        def start_login(self, *, device_auth=False):
            self.login_calls.append(bool(device_auth))
            return {
                "ok": True,
                "message": "Connexion Grok lancée.",
                "process": self.login_process,
            }

        def model_catalog(self):
            return {
                "ok": True,
                "models": [
                    {
                        "slug": "grok-build",
                        "display_name": "grok-build",
                        "description": "Modèle disponible via `grok models`.",
                        "default_reasoning_level": "",
                        "supported_reasoning_levels": [],
                        "priority": 0,
                    }
                ],
                "default_model": "grok-build",
                "message": "1 modèle(s) Grok disponible(s).",
            }

    def test_grok_settings_are_persisted_and_reported(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            controller = AgentController(
                paths=paths, services=SettingsApiTests()._services(paths, settings)
            )
            api = DesktopApi(controller)

            result = api.update_settings({"chat_provider": "grok", "grok_model": "grok-build"})

            self.assertEqual(result["chat_provider"], "grok")
            self.assertEqual(result["grok_model"], "grok-build")
            self.assertEqual(SettingsStore(paths.settings_file).get("chat_provider"), "grok")
            self.assertEqual(api.get_state()["chat_provider"], "grok")

    def test_grok_models_exposes_cli_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.controller._grok_cli = self.FakeGrok()

            result = api.grok_models()

            self.assertTrue(result["ok"], result["message"])
            self.assertEqual(result["default_model"], "grok-build")
            self.assertIn("grok-build", [m["slug"] for m in result["models"]])

    def test_grok_login_supports_device_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)
            fake = self.FakeGrok()
            api.controller._grok_cli = fake

            result = api.grok_login(device_auth=True)

            self.assertTrue(result["ok"], result["message"])
            self.assertTrue(result["running"])
            self.assertEqual(fake.login_calls, [True])
            deadline = time.time() + 2
            while not any(event == "grok_login" for event, _payload in events):
                if time.time() > deadline:
                    self.fail("expected grok_login event")
                time.sleep(0.01)
            payload = [payload for event, payload in events if event == "grok_login"][-1]
            self.assertTrue(payload["ok"], payload["message"])
            self.assertEqual(fake.login_process.timeout, 600)

    def test_grok_provider_uses_grok_cli_and_tracks_usage(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "grok")
            settings.set("grok_model", "grok-build")
            services = SettingsApiTests()._services(paths, settings)
            controller = AgentController(paths=paths, services=services)
            fake = self.FakeGrok()
            controller._grok_cli = fake
            api = DesktopApi(controller)
            api.set_workdir(str(workdir))

            payload = api.send_message("Analyse ce projet avec Grok")

            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(payload["content"], "Réponse depuis le compte Grok.")
            self.assertEqual(fake.calls[0][1]["model"], "grok-build")
            self.assertEqual(fake.calls[0][1]["workdir"], workdir.resolve())
            self.assertEqual(fake.calls[0][1]["output_format"], "streaming-json")
            self.assertTrue(callable(fake.calls[0][1]["on_chunk"]))

            usage = api.usage()
            self.assertEqual(usage["grok"]["turns"], 1)
            self.assertEqual(usage["grok"]["total_tokens"], 950)
            by_model = {row["model"]: row for row in usage["grok"]["by_model"]}
            self.assertIn("grok-build", by_model)

    def test_grok_casual_prompt_keeps_internal_plumbing_out_of_view(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "grok")
            services = SettingsApiTests()._services(paths, settings)
            controller = AgentController(paths=paths, services=services)
            fake = self.FakeGrok()
            controller._grok_cli = fake
            api = DesktopApi(controller)

            api.send_message("yo")

            prompt = fake.calls[0][0]
            self.assertIn("Message utilisateur :\nyo", prompt)
            self.assertNotIn("Lity", prompt)
            self.assertNotIn("Grok Build", prompt)
            self.assertNotIn("modèle local", prompt)
            self.assertNotIn("RAG", prompt)
            self.assertNotIn("CREATE", prompt)
            self.assertNotIn("SEARCH-REPLACE", prompt)

    def test_grok_file_edit_prompt_still_explains_reviewed_edit_blocks(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "grok")
            services = SettingsApiTests()._services(paths, settings)
            controller = AgentController(paths=paths, services=services)
            fake = self.FakeGrok()
            controller._grok_cli = fake
            api = DesktopApi(controller)

            api.send_message("Modifie le fichier README.md pour ajouter une section usage")

            prompt = fake.calls[0][0]
            self.assertIn("CREATE", prompt)
            self.assertIn("SEARCH-REPLACE", prompt)

    def test_grok_provider_uses_stable_headless_session_per_conversation(self):
        from lity.infrastructure.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            settings = SettingsStore(paths.settings_file)
            settings.set("chat_provider", "grok")
            services = SettingsApiTests()._services(paths, settings)
            controller = AgentController(paths=paths, services=services)
            fake = self.FakeGrok()
            controller._grok_cli = fake
            api = DesktopApi(controller)

            first_conversation = controller.active_conversation_id
            api.send_message("Premier tour")
            api.send_message("Deuxième tour")
            api.new_conversation()
            second_conversation = controller.active_conversation_id
            api.send_message("Autre conversation")

            first_session = f"lity-{first_conversation}"
            second_session = f"lity-{second_conversation}"
            self.assertEqual(fake.calls[0][1]["session_id"], first_session)
            self.assertEqual(fake.calls[1][1]["session_id"], first_session)
            self.assertEqual(fake.calls[2][1]["session_id"], second_session)


class RagAndSearchApiTests(unittest.TestCase):
    def test_index_project_without_embeddings_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            (workdir / "a.py").write_text("print('hi')", encoding="utf-8")
            api = _build_api_with_real_files(tmp, workdir)
            # FakeStreamingEngine has no embed() -> indexing unsupported, no crash.
            result = api.index_project()
            self.assertFalse(result["ok"])
            self.assertIn("rag_enabled", api.set_rag(True))

    def test_search_conversations_matches_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("parle moi des volcans")
            matches = api.search_conversations("volcans")
            self.assertEqual(len(matches), 1)
            self.assertEqual(api.search_conversations("inexistant"), [])


class YoloApiTests(unittest.TestCase):
    def test_set_yolo_enables_agent_and_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            result = api.set_yolo(True)
            self.assertTrue(result["yolo"])
            self.assertTrue(result["agent_mode"])
            self.assertEqual(result["write_mode"], "autonomous")
            self.assertTrue(api.get_state()["yolo"])

    def test_yolo_agent_writes_file_autonomously(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "project"
            workdir.mkdir()
            paths = AppPaths.create(home_override=Path(tmp))
            engine = FakeAgentEngine(
                [
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "name": "write_file",
                                "arguments": {"path": "poeme.md", "content": "Oh concombre géant"},
                            }
                        ],
                    },
                    {"content": "J'ai créé poeme.md.", "tool_calls": []},
                ]
            )
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=FileManager(),
                router=FakeRouter(),
                editor=CodeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)
            api = DesktopApi(controller)
            api.set_yolo(True)
            api.set_workdir(str(workdir))

            payload = api.send_message("écris un poème dans poeme.md")

            self.assertEqual(payload["type"], "ai_response")
            self.assertTrue((workdir / "poeme.md").exists())
            self.assertIn("concombre", (workdir / "poeme.md").read_text(encoding="utf-8"))


class MemoryApiTests(unittest.TestCase):
    def test_view_update_delete_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.update_memory_entry("user_profile", "prénom", "Alex")
            api.update_memory_entry("facts", "projet", "Lity")

            memory = api.get_memory()
            self.assertEqual(memory["user_profile"]["prénom"], "Alex")
            self.assertEqual(memory["facts"]["projet"], "Lity")

            api.delete_memory_entry("user_profile", "prénom")
            self.assertNotIn("prénom", api.get_memory()["user_profile"])

            api.clear_memory()
            self.assertEqual(api.get_memory()["facts"], {})


class HealthApiTests(unittest.TestCase):
    def test_health_lists_services(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            names = [entry["name"] for entry in api.get_health()]
            self.assertIn("Stable Diffusion", names)
            self.assertIn("Embeddings", names)
            self.assertIn("Voix", names)
            self.assertTrue(any("Ollama" in name for name in names))


class FakeSTT:
    is_model_ready = True
    model_load_error = None

    def start_recording(self):
        return True, "Enregistrement démarré."

    def stop_recording_and_transcribe(self, on_done):
        on_done("bonjour le monde")


class FakeTTS:
    current_voice_name = "fr_FR-upmc-medium"

    def __init__(self):
        self.spoken: list[str] = []
        self.downloaded: list[str] = []

    def get_voices(self):
        return ["fr_FR-upmc-medium"]

    def speak(self, text, on_finish=None):
        self.spoken.append(text)
        if on_finish:
            on_finish()

    def stop(self):
        pass

    def download_voice(self, voice_path, on_progress=None):
        self.downloaded.append(voice_path)

    def load_voice(self, name):
        self.current_voice_name = name
        return True


class VoiceApiTests(unittest.TestCase):
    def test_voice_roundtrip_with_fake_managers(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.controller._stt = FakeSTT()
            api.controller._tts = FakeTTS()

            self.assertTrue(api.start_recording()["ok"])
            self.assertEqual(api.stop_recording()["text"], "bonjour le monde")
            self.assertTrue(api.speak("salut")["ok"])
            self.assertEqual(api.controller._tts.spoken, ["salut"])

            status = api.audio_status()
            self.assertTrue(status["stt_ready"])
            self.assertTrue(status["has_voice"])


class FakeImageManager:
    def __init__(self, image_path):
        self._active = False
        self._image_path = image_path

    def is_active(self):
        return self._active

    def start_session(self):
        self._active = True
        return {"type": "image_mode_ready", "status": "ready", "message": "Mode image activé."}

    def poll_launch_status(self):
        return {"status": "ready"}

    def cancel_session(self):
        self._active = False

    def process_user_message(self, user_input, engine):
        return {
            "type": "image_generation_result",
            "content": {"image_path": self._image_path},
            "message": "Image générée.",
        }


class FakeVideoManager:
    def __init__(self, video_path):
        self._active = False
        self._video_path = video_path
        self.engine = type("E", (), {"unload": lambda self: None})()

    def is_active(self):
        return self._active

    def start_session(self):
        self._active = True
        return {"type": "video_mode_ready", "status": "ready", "message": "Mode vidéo activé."}

    def poll_launch_status(self):
        return {"status": "ready"}

    def cancel_session(self):
        self._active = False

    def process_user_message(self, user_input, engine):
        return {
            "type": "video_generation_result",
            "content": {"video_path": self._video_path},
            "message": "Vidéo générée.",
        }


class ImageApiTests(unittest.TestCase):
    def test_image_session_and_result_becomes_data_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_file = Path(tmp) / "out.png"
            image_file.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
            api, _events = _build_api(tmp)
            api.controller.image_manager = FakeImageManager(str(image_file))

            self.assertEqual(api.start_image_session()["status"], "ready")
            self.assertTrue(api.image_active())

            payload = api.send_message("un chat astronaute")
            self.assertEqual(payload["type"], "image_generation_result")
            self.assertTrue(payload["image"].startswith("data:image/png;base64,"))

            api.stop_image_session()
            self.assertFalse(api.image_active())


class VideoApiTests(unittest.TestCase):
    def test_video_session_and_result_becomes_data_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_file = Path(tmp) / "out.mp4"
            video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
            api, _events = _build_api(tmp)
            api.controller.video_manager = FakeVideoManager(str(video_file))

            self.assertEqual(api.start_video_session()["status"], "ready")
            self.assertTrue(api.video_active())
            self.assertTrue(api.get_state()["video_active"])

            payload = api.send_message("un chat qui court")
            self.assertEqual(payload["type"], "video_generation_result")
            self.assertTrue(payload["video"].startswith("data:video/mp4;base64,"))

            api.stop_video_session()
            self.assertFalse(api.video_active())

    def test_download_video_model_unknown_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            result = api.download_video_model("does-not-exist")
            self.assertFalse(result["ok"])
            self.assertFalse(result["running"])

    def test_video_pull_status_and_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            self.assertIsNone(api.video_pull_status()["active"])
            self.assertTrue(api.cancel_video_download()["ok"])
            self.assertTrue(api._video_pull_cancel)

    def test_videoify_wraps_video_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "c.mp4"
            mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            api, _events = _build_api(tmp)
            out = api._videoify(
                {"type": "video_generation_result", "content": {"video_path": str(mp4)}}
            )
            self.assertTrue(out["video"].startswith("data:video/mp4;base64,"))


class PerConversationModelTests(unittest.TestCase):
    def test_model_restored_on_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("salut")
            first = api.controller.active_conversation_id
            api.set_model("other:latest")  # bound to first

            api.new_conversation()
            api.send_message("hello")
            second = api.controller.active_conversation_id
            api.set_model("fake-model")  # bound to second

            api.switch_conversation(first)
            self.assertEqual(api.controller.engine.model, "other:latest")
            api.switch_conversation(second)
            self.assertEqual(api.controller.engine.model, "fake-model")


class ModelMgmtApiTests(unittest.TestCase):
    def test_list_pull_delete_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)
            self.assertEqual(api.list_models_detailed()[0]["name"], "fake-model")

            # pull_model now enqueues and returns the queue snapshot.
            result = api.pull_model("llama3.2")
            self.assertIn("llama3.2", [result["active"], *result["queue"]])
            if api._pull_worker is not None:
                api._pull_worker.join(timeout=3)
            self.assertTrue([p for event, p in events if event == "pull_progress"])
            self.assertTrue([p for event, p in events if event == "pull_done"])

            self.assertTrue(api.delete_model("x")["ok"])
            self.assertIn("parameters", api.model_info("fake-model"))

    def test_generation_stats(self):
        class StatsEngine(FakeStreamingEngine):
            last_stats = {"context_used": 150, "tokens_per_sec": 42.0}

            def context_length(self, name=None):
                return 8192

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=StatsEngine(),
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            api = DesktopApi(AgentController(paths=paths, services=services))
            stats = api.generation_stats()
            self.assertEqual(stats["tokens_per_sec"], 42.0)
            self.assertEqual(stats["context_used"], 150)
            self.assertEqual(stats["context_length"], 8192)
            self.assertEqual(stats["usage_pct"], 2)  # round(150/8192*100)

    def test_model_supports_tools(self):
        class CapsEngine(FakeStreamingEngine):
            def __init__(self, caps):
                super().__init__()
                self._caps = caps

            def model_info(self, name):
                return {"capabilities": self._caps}

        def supports(caps):
            with tempfile.TemporaryDirectory() as tmp:
                paths = AppPaths.create(home_override=Path(tmp))
                services = AppServices(
                    settings=None,
                    engine=CapsEngine(caps),
                    memory=MemoryManager(paths=paths),
                    files=FakeFiles(),
                    router=FakeRouter(),
                    editor=FakeEditor(),
                    image_manager=None,
                )
                api = DesktopApi(AgentController(paths=paths, services=services))
                return api.model_supports_tools()["supports"]

        self.assertTrue(supports(["completion", "tools"]))
        self.assertFalse(supports(["completion"]))
        self.assertIsNone(supports([]))  # unknown (old Ollama) → no false alarm


class GitApiTests(unittest.TestCase):
    def test_status_and_commit(self):
        import shutil
        import subprocess

        if shutil.which("git") is None:
            self.skipTest("git indisponible")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            for args in (
                ["init"],
                ["config", "user.email", "t@t.io"],
                ["config", "user.name", "Test"],
            ):
                subprocess.run(["git", *args], cwd=repo, capture_output=True)
            (repo / "a.txt").write_text("hello", encoding="utf-8")
            api = _build_api_with_real_files(tmp, repo)

            status = api.git_status()
            self.assertTrue(status["is_repo"])
            self.assertTrue(any(entry["path"] == "a.txt" for entry in status["files"]))

            result = api.git_commit("commit initial")
            self.assertTrue(result["ok"], result["message"])
            self.assertEqual(result["status"]["files"], [])


class ExportPinApiTests(unittest.TestCase):
    def test_export_markdown_and_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("bonjour le monde")
            cid = api.controller.active_conversation_id

            export = api.export_conversation(cid, "markdown")
            self.assertTrue(export["ok"])
            self.assertIn("bonjour le monde", export["content"])
            self.assertTrue(export["filename"].endswith(".md"))

            pinned = api.set_pinned(cid, True)
            self.assertTrue(pinned["conversations"][0]["pinned"])


class ModelVoiceCatalogTests(unittest.TestCase):
    def test_model_suggestions_cover_chat_and_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            suggestions = api.model_suggestions()
            categories = {item["category"] for item in suggestions}
            self.assertIn("chat", categories)
            self.assertIn("embedding", categories)

    def test_voice_catalog_download_and_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.controller._tts = FakeTTS()

            voices = api.list_voices()
            self.assertTrue(voices["available"])
            self.assertGreaterEqual(len(voices["catalog"]), 1)
            self.assertIn("fr_FR-upmc-medium", voices["installed"])

            result = api.download_voice("en_US-amy-medium")
            self.assertTrue(result["ok"])
            self.assertIn("en/en_US/amy/medium/en_US-amy-medium", api.controller._tts.downloaded)

            self.assertTrue(api.set_voice("fr_FR-upmc-medium")["ok"])


class DesktopApiTests(unittest.TestCase):
    def test_send_message_streams_chunks_and_returns_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)

            payload = api.send_message("bonjour")

            chunks = [p["content"] for e, p in events if e == "chunk"]
            self.assertEqual(chunks, ["Bon", "jour", " !"])
            self.assertEqual(payload["type"], "ai_response")
            self.assertEqual(payload["content"], "Bonjour !")
            self.assertIn("conversations", payload)
            self.assertIn("active_conversation_id", payload)

    def test_send_message_persists_into_active_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api.send_message("bonjour")

            messages = api.get_messages()
            self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
            self.assertEqual(messages[0]["content"], "bonjour")
            self.assertEqual(messages[1]["content"], "Bonjour !")

    def test_slash_command_returns_slash_payload_without_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)

            payload = api.send_message("/help")

            self.assertEqual(payload["type"], "slash")
            self.assertIn("COMMANDES", payload["message"])
            self.assertEqual([e for e, _ in events if e == "chunk"], [])

    def test_busy_guard_rejects_concurrent_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            api._busy = True

            payload = api.send_message("bonjour")
            self.assertEqual(payload["type"], "error")

    def test_stop_sets_cancel_and_silences_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)

            self.assertEqual(api.stop(), {"stopped": True})
            events.clear()
            api._on_chunk("ignored")
            self.assertEqual(events, [])

    def test_conversation_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            first = api.controller.active_conversation_id

            api.send_message("salut depuis A")
            created = api.new_conversation()
            self.assertNotEqual(created["id"], first)

            api.send_message("salut depuis B")
            switched = api.switch_conversation(first)
            self.assertTrue(switched["success"])
            self.assertEqual(switched["messages"][0]["content"], "salut depuis A")

            listed = api.list_conversations()
            self.assertEqual(len(listed), 2)

            result = api.delete_conversation(created["id"])
            self.assertEqual(len(result["conversations"]), 1)

    def test_get_state_and_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)

            api.send_message("bonjour")  # a draft only appears once it has a message
            state = api.get_state()
            self.assertEqual(state["model"], "fake-model")
            self.assertIn("assistant_name", state)
            self.assertEqual(len(state["conversations"]), 1)

            models = api.list_models()
            self.assertIn("fake-model", models["models"])
            self.assertIsNone(models["error"])

    def test_set_workdir_reports_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            result = api.set_workdir(tmp)
            self.assertTrue(result["success"])
            self.assertEqual(result["workdir"], str(Path(tmp)))

    def test_choose_workdir_uses_folder_picker(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            services = AppServices(
                settings=None,
                engine=FakeStreamingEngine(),
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            controller = AgentController(paths=paths, services=services)

            cancelled = DesktopApi(controller, folder_picker=lambda: None)
            self.assertFalse(cancelled.choose_workdir()["success"])

            chosen = DesktopApi(controller, folder_picker=lambda: tmp)
            result = chosen.choose_workdir()
            self.assertTrue(result["success"])
            self.assertEqual(result["workdir"], str(Path(tmp)))

    def test_choose_workdir_without_picker_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _events = _build_api(tmp)
            result = api.choose_workdir()
            self.assertFalse(result["success"])


class PullQueueTests(unittest.TestCase):
    def test_queues_and_runs_pulls_in_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)
            first = api.pull_model("gemma2:2b")
            self.assertIn("gemma2:2b", [first["active"], *first["queue"]])
            api.pull_model("phi3")  # second one is queued behind the first

            if api._pull_worker is not None:
                api._pull_worker.join(timeout=3)

            done = [payload["name"] for event, payload in events if event == "pull_done"]
            self.assertIn("gemma2:2b", done)
            self.assertIn("phi3", done)

            status = api.pull_status()
            self.assertIsNone(status["active"])
            self.assertEqual(status["queue"], [])

    def test_pull_status_snapshot_survives_reopen(self):
        # A blocking engine keeps the first pull running so we can observe the
        # queue snapshot the modal reads on reopen.
        class BlockingPullEngine(FakeStreamingEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def pull_model(self, name, on_progress=None):
                self.started.set()
                self.release.wait(timeout=5)
                return {"ok": True, "message": f"{name} ok"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            engine = BlockingPullEngine()
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            api = DesktopApi(AgentController(paths=paths, services=services))

            api.pull_model("gemma2:9b")
            api.pull_model("qwen2.5:7b")
            self.assertTrue(engine.started.wait(timeout=3))

            # What a freshly reopened modal would see: one active + one queued.
            status = api.pull_status()
            self.assertEqual(status["active"], "gemma2:9b")
            self.assertEqual(status["queue"], ["qwen2.5:7b"])

            engine.release.set()
            if api._pull_worker is not None:
                api._pull_worker.join(timeout=3)
            self.assertIsNone(api.pull_status()["active"])

    def test_pull_done_event_clears_active_banner(self):
        # Regression: the modal banner kept showing "Téléchargement" after a pull
        # finished because pull_done was emitted with the just-done model still
        # marked active. The event must report active=None when nothing is left.
        with tempfile.TemporaryDirectory() as tmp:
            api, events = _build_api(tmp)
            api.pull_model("solo:latest")
            if api._pull_worker is not None:
                api._pull_worker.join(timeout=3)

            done = [payload for event, payload in events if event == "pull_done"]
            self.assertEqual(done[-1]["name"], "solo:latest")
            self.assertIsNone(done[-1]["active"])
            self.assertEqual(done[-1]["queue"], [])

    def test_cancel_queued_model_drops_it_without_pulling(self):
        # A blocking engine keeps the first pull busy so the second stays queued
        # long enough to cancel it before it ever starts.
        class BlockingPullEngine(FakeStreamingEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def pull_model(self, name, on_progress=None):
                self.started.set()
                self.release.wait(timeout=5)
                return {"ok": True, "message": f"{name} ok"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            engine = BlockingPullEngine()
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            events: list[tuple[str, dict]] = []
            api = DesktopApi(
                AgentController(paths=paths, services=services),
                emit=lambda event, payload: events.append((event, payload)),
            )

            api.pull_model("first:latest")
            api.pull_model("second:latest")
            self.assertTrue(engine.started.wait(timeout=3))

            status = api.cancel_pull("second:latest")
            self.assertNotIn("second:latest", status["queue"])
            self.assertEqual(status["active"], "first:latest")

            engine.release.set()
            if api._pull_worker is not None:
                api._pull_worker.join(timeout=3)

            pulled = [payload["name"] for event, payload in events if event == "pull_done"]
            self.assertIn("first:latest", pulled)
            self.assertNotIn("second:latest", pulled)

    def test_cancel_active_download_aborts_it(self):
        # An engine that honours should_cancel: it spins until cancelled (or a
        # release event lets it finish), so we can abort the in-flight pull.
        class CancellablePullEngine(FakeStreamingEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.finish = threading.Event()

            def pull_model(self, name, on_progress=None, should_cancel=None):
                self.started.set()
                while not (should_cancel and should_cancel()):
                    if self.finish.is_set():
                        return {"ok": True, "message": f"{name} ok"}
                    if on_progress:
                        on_progress({"status": "downloading", "completed": 1, "total": 100})
                    time.sleep(0.005)
                return {"ok": False, "cancelled": True, "message": f"{name} annulé"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            engine = CancellablePullEngine()
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            events: list[tuple[str, dict]] = []
            api = DesktopApi(
                AgentController(paths=paths, services=services),
                emit=lambda event, payload: events.append((event, payload)),
            )

            api.pull_model("big:latest")
            self.assertTrue(engine.started.wait(timeout=3))

            api.cancel_pull()  # no name → cancel whatever is active
            if api._pull_worker is not None:
                api._pull_worker.join(timeout=3)

            done = [payload for event, payload in events if event == "pull_done"]
            self.assertTrue(done[-1]["cancelled"])
            self.assertFalse(done[-1]["ok"])
            self.assertIsNone(done[-1]["active"])
            self.assertIsNone(api.pull_status()["active"])

    def test_pull_done_promotes_next_queued_for_seamless_handoff(self):
        # When one download finishes with another queued, the completion event
        # should already report the next model as active (no empty "flash" in
        # the banner), then clear to None once the queue drains.
        class BlockFirstEngine(FakeStreamingEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def pull_model(self, name, on_progress=None):
                if name.startswith("first"):
                    self.started.set()
                    self.release.wait(timeout=5)
                return {"ok": True, "message": f"{name} ok"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            engine = BlockFirstEngine()
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            events: list[tuple[str, dict]] = []
            api = DesktopApi(
                AgentController(paths=paths, services=services),
                emit=lambda event, payload: events.append((event, payload)),
            )

            api.pull_model("first:latest")
            self.assertTrue(engine.started.wait(timeout=3))
            api.pull_model("second:latest")  # queued behind the blocked first
            engine.release.set()
            if api._pull_worker is not None:
                api._pull_worker.join(timeout=3)

            handoff = [(p["name"], p["active"]) for e, p in events if e == "pull_done"]
            # first done → second promoted into active; second done → cleared.
            self.assertIn(("first:latest", "second:latest"), handoff)
            self.assertIn(("second:latest", None), handoff)

    def test_cancel_during_handoff_window_is_not_lost(self):
        # Regression for the promotion-window race: a cancel aimed at the model
        # that was just promoted (during first's pull_done broadcast) must stick,
        # not be wiped by the worker re-entering its loop. We fire the cancel
        # synchronously from inside the broadcast — the exact race window.
        class HandoffEngine(FakeStreamingEngine):
            def __init__(self):
                super().__init__()
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.second_pulled_fully = False

            def pull_model(self, name, on_progress=None, should_cancel=None):
                if name.startswith("first"):
                    self.first_started.set()
                    self.release_first.wait(timeout=5)
                    return {"ok": True, "message": "first ok"}
                # second: spin until cancelled; only "completes" if never cancelled
                for _ in range(400):
                    if should_cancel and should_cancel():
                        return {"ok": False, "cancelled": True, "message": "second annulé"}
                    time.sleep(0.005)
                self.second_pulled_fully = True
                return {"ok": True, "message": "second ok"}

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            engine = HandoffEngine()
            services = AppServices(
                settings=None,
                engine=engine,
                memory=MemoryManager(paths=paths),
                files=FakeFiles(),
                router=FakeRouter(),
                editor=FakeEditor(),
                image_manager=None,
            )
            events: list[tuple[str, dict]] = []
            box: dict[str, DesktopApi] = {}

            def emit(event, payload):
                events.append((event, payload))
                # Inside first's completion broadcast, second is already the active
                # model — cancel it right now (the worker holds no lock here).
                if event == "pull_done" and payload.get("name") == "first:latest":
                    box["api"].cancel_pull("second:latest")

            api = DesktopApi(AgentController(paths=paths, services=services), emit=emit)
            box["api"] = api

            api.pull_model("first:latest")
            self.assertTrue(engine.first_started.wait(timeout=3))
            api.pull_model("second:latest")  # queued behind the blocked first
            engine.release_first.set()
            if api._pull_worker is not None:
                api._pull_worker.join(timeout=5)

            # The cancel landed in the handoff window and must have taken effect.
            self.assertFalse(engine.second_pulled_fully)
            done = {p["name"]: p for e, p in events if e == "pull_done"}
            self.assertTrue(done["second:latest"]["cancelled"])
            self.assertFalse(done["second:latest"]["ok"])
            self.assertIsNone(api.pull_status()["active"])


if __name__ == "__main__":
    unittest.main()
