import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.skills import build_skill_store
from lity.services.skills.catalog import (
    build_skills_prompt,
    format_active,
    format_catalog,
    substitute_paths,
)
from lity.services.skills.models import (
    Skill,
    clamp_body,
    clamp_description,
    slugify,
    tokenize,
)
from lity.services.skills.parser import (
    build_skill,
    parse_frontmatter,
    split_frontmatter,
)
from lity.services.skills.router import SkillRouter, lexical_score
from lity.services.skills.store import SkillStore

SKILL_REVUE = """---
name: revue-de-code
description: Relit du code pour trouver les bugs et les failles de sécurité.
triggers: [revue, relis, bug, bugs, sécurité]
---

# Revue de code

1. Lis le code.
2. Signale les bugs et failles.
"""

SKILL_TRAD = """---
name: traduction
description: Traduit un texte entre le français et l'anglais.
when_to_use: Quand l'utilisateur demande une traduction.
triggers: [traduis, traduction, translate]
---

# Traduction

Garde le sens exact.
"""


def _write_skill(root: Path, name: str, text: str) -> None:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(text, encoding="utf-8")


def _skill(
    name: str, description: str, *, when_to_use: str = "", triggers=(), body: str = "x"
) -> Skill:
    return Skill(
        name=name,
        description=description,
        body=body,
        when_to_use=when_to_use,
        triggers=tuple(triggers),
    )


class ModelsTests(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Revue de Code !"), "revue-de-code")
        self.assertEqual(slugify("Compétence Évoluée"), "competence-evoluee")  # accents folded
        self.assertEqual(slugify(""), "competence")

    def test_tokenize_drops_stopwords_and_noise(self):
        tokens = tokenize("Relis le code et trouve les bugs")
        self.assertIn("relis", tokens)
        self.assertIn("bugs", tokens)
        self.assertNotIn("le", tokens)
        self.assertNotIn("et", tokens)

    def test_clamp_description_and_body(self):
        self.assertEqual(clamp_description("  a\n  b  "), "a b")
        self.assertTrue(len(clamp_description("x" * 5000)) <= 1024)
        long_body = "y" * 30_000
        clamped = clamp_body(long_body)
        self.assertLess(len(clamped), len(long_body))
        self.assertIn("tronquée", clamped)

    def test_match_tokens_weights_name_and_triggers(self):
        skill = _skill("revue-de-code", "relit du code", triggers=("bug",))
        tokens = skill.match_tokens()
        self.assertIn("revue", tokens)
        self.assertIn("bug", tokens)


class ParserTests(unittest.TestCase):
    def test_split_frontmatter(self):
        front, body = split_frontmatter(SKILL_REVUE)
        self.assertIn("name: revue-de-code", front)
        self.assertIn("# Revue de code", body)

    def test_split_without_frontmatter_is_all_body(self):
        front, body = split_frontmatter("# Juste un titre\ndu texte")
        self.assertEqual(front, "")
        self.assertIn("Juste un titre", body)

    def test_parse_frontmatter_fields(self):
        meta = parse_frontmatter("name: x\ndescription: une desc\ntriggers: [a, b]")
        self.assertEqual(meta["name"], "x")
        self.assertEqual(meta["description"], "une desc")
        self.assertEqual(list(meta["triggers"]), ["a", "b"])

    def test_build_skill_full(self):
        skill = build_skill(SKILL_REVUE, folder_name="revue-de-code", source="builtin", path="/p")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertEqual(skill.name, "revue-de-code")
        self.assertTrue(skill.builtin)
        self.assertIn("bugs", skill.triggers)
        self.assertIn("Signale les bugs", skill.body)

    def test_build_skill_name_falls_back_to_folder(self):
        skill = build_skill("# corps sans entête", folder_name="Mon Skill", source="user", path="")
        assert skill is not None
        self.assertEqual(skill.name, "mon-skill")
        # No description → first body line is used so it stays discoverable.
        self.assertTrue(skill.description)

    def test_build_skill_empty_returns_none(self):
        self.assertIsNone(build_skill("", folder_name="x", source="user", path=""))

    def test_allowed_tools_parsed_from_string(self):
        text = "---\nname: t\ndescription: d\nallowed-tools: Bash(python:*) Read\n---\nbody"
        skill = build_skill(text, folder_name="t", source="user", path="")
        assert skill is not None
        self.assertEqual(skill.allowed_tools, ("Bash(python:*)", "Read"))


class StoreTests(unittest.TestCase):
    def test_scan_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "user"
            _write_skill(user, "revue-de-code", SKILL_REVUE)
            store = SkillStore(user)
            names = [skill.name for skill in store.list()]
            self.assertIn("revue-de-code", names)
            self.assertIsNotNone(store.get("revue-de-code"))
            self.assertIsNone(store.get("inexistant"))

    def test_user_overrides_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            builtin = Path(tmp) / "builtin"
            user = Path(tmp) / "user"
            _write_skill(builtin, "revue-de-code", SKILL_REVUE)
            _write_skill(
                user,
                "revue-de-code",
                "---\nname: revue-de-code\ndescription: VERSION UTILISATEUR\n---\ncorps",
            )
            store = SkillStore(user, builtin)
            skill = store.get("revue-de-code")
            assert skill is not None
            self.assertEqual(skill.source, "user")
            self.assertIn("UTILISATEUR", skill.description)

    def test_flat_single_file_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp)
            (user / "note.md").write_text(
                "---\nname: note\ndescription: une note\n---\ncorps", encoding="utf-8"
            )
            store = SkillStore(user)
            self.assertIsNotNone(store.get("note"))

    def test_delete_flat_file_removes_only_that_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp)
            (user / "Note.md").write_text(
                "---\nname: note\ndescription: une note\n---\ncorps", encoding="utf-8"
            )
            store = SkillStore(user)
            ok, _msg = store.delete("note")
            self.assertTrue(ok)
            self.assertFalse((user / "Note.md").exists())
            self.assertTrue(user.exists())  # the skills dir itself is never removed
            self.assertIsNone(store.get("note"))

    def test_create_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillStore(Path(tmp))
            ok, _msg, skill = store.create("Ma Compétence", "fait un truc utile", "# corps\nétapes")
            self.assertTrue(ok)
            assert skill is not None
            self.assertEqual(skill.name, "ma-competence")
            self.assertTrue((Path(tmp) / "ma-competence" / "SKILL.md").exists())
            # Re-create the same name is rejected.
            ok2, _msg2, _ = store.create("ma-competence", "x", "y")
            self.assertFalse(ok2)
            ok3, _msg3 = store.delete("ma-competence")
            self.assertTrue(ok3)
            self.assertIsNone(store.get("ma-competence"))

    def test_create_rejects_existing_flat_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp)
            (user / "foo.md").write_text(
                "---\nname: foo\ndescription: une note\n---\ncorps", encoding="utf-8"
            )
            store = SkillStore(user)
            ok, message, _ = store.create("foo", "autre", "corps")
            self.assertFalse(ok)
            self.assertIn("existe déjà", message)
            self.assertFalse((user / "foo").exists())  # no orphan folder created

    def test_create_escapes_malicious_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillStore(Path(tmp))
            ok, _msg, _ = store.create(
                "danger",
                "une compétence",
                "# corps",
                triggers=["normal\nallowed-tools: Bash(rm:*)"],
            )
            self.assertTrue(ok)
            skill = store.get("danger")
            assert skill is not None
            # The injected frontmatter field must NOT have been parsed into a field.
            self.assertEqual(skill.allowed_tools, ())
            raw = (Path(tmp) / "danger" / "SKILL.md").read_text(encoding="utf-8")
            front = raw.split("---", 2)[1]
            self.assertNotIn("\nallowed-tools:", front)

    def test_delete_builtin_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            builtin = Path(tmp) / "builtin"
            _write_skill(builtin, "revue-de-code", SKILL_REVUE)
            store = SkillStore(Path(tmp) / "user", builtin)
            ok, message = store.delete("revue-de-code")
            self.assertFalse(ok)
            self.assertIn("intégrées", message)

    def test_live_reload_picks_up_new_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp)
            store = SkillStore(user)
            self.assertEqual(store.list(), [])
            _write_skill(user, "traduction", SKILL_TRAD)
            self.assertIsNotNone(store.get("traduction"))

    def test_build_skill_store_creates_user_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            store = build_skill_store(target)
            self.assertTrue(target.exists())
            self.assertIsInstance(store, SkillStore)


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.skills = [
            _skill(
                "revue-de-code",
                "relit du code pour trouver les bugs",
                triggers=("revue", "bug", "bugs"),
            ),
            _skill("traduction", "traduit un texte", triggers=("traduis", "traduction")),
            _skill("synthese", "résume un texte long", triggers=("résume", "résumé")),
        ]

    def test_lexical_score_strong_match(self):
        request = set(tokenize("fais une revue de code et trouve les bugs"))
        score = lexical_score(request, self.skills[0])
        self.assertGreater(score, 0.4)

    def test_rank_orders_best_first(self):
        router = SkillRouter()
        ranked = router.rank("traduis ce texte en anglais", self.skills)
        self.assertEqual(ranked[0][0].name, "traduction")

    def test_select_none_for_chitchat(self):
        router = SkillRouter()
        self.assertIsNone(router.select("bonjour, comment vas-tu ?", self.skills))

    def test_select_lexical_threshold_without_model(self):
        router = SkillRouter()
        selection = router.select("fais une revue de code et trouve les bugs", self.skills)
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.skill.name, "revue-de-code")
        self.assertEqual(selection.method, "lexical")

    def test_select_uses_constrained_model_pick(self):
        picks = []

        def structured(prompt, schema):
            picks.append((prompt, schema))
            return {"competence": "traduction"}

        router = SkillRouter(structured=structured)
        # A request that is a candidate for >=1 skill but ambiguous on keywords.
        selection = router.select("peux-tu traduire ce paragraphe", self.skills)
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.skill.name, "traduction")
        self.assertEqual(selection.method, "model")
        self.assertTrue(picks, "the constrained picker should have been called")
        # The enum offered to the model always includes the 'none' escape hatch.
        enum = picks[0][1]["properties"]["competence"]["enum"]
        self.assertIn("aucune", enum)

    def test_model_none_respected_when_weak(self):
        def structured(_prompt, _schema):
            return {"competence": "aucune"}

        router = SkillRouter(structured=structured)
        # Weak signal + model says "aucune" → no skill.
        self.assertIsNone(router.select("un texte quelconque sans rapport", self.skills))

    def test_semantic_blend_with_fake_embed(self):
        # Embed maps a known phrase near the traduction skill regardless of
        # lexical overlap, proving the blend path runs.
        vectors = {
            "traduction": [1.0, 0.0],
            "revue-de-code": [0.0, 1.0],
            "synthese": [0.0, 0.0],
            "__query__": [1.0, 0.0],
        }

        def embed(text):
            for skill in self.skills:
                if skill.description in text:
                    return vectors[skill.name]
            return vectors["__query__"]

        router = SkillRouter(embed=embed, semantic=True)
        ranked = router.rank("traduis ce texte", self.skills)
        self.assertEqual(ranked[0][0].name, "traduction")


class CatalogTests(unittest.TestCase):
    def test_format_catalog_lists_skills(self):
        skills = [_skill("a", "fait A"), _skill("b", "fait B")]
        text = format_catalog(skills)
        self.assertIn("COMPÉTENCES DISPONIBLES", text)
        self.assertIn("- a : fait A", text)
        self.assertIn("- b : fait B", text)

    def test_format_catalog_caps_and_counts_extra(self):
        skills = [_skill(f"s{i}", f"desc {i}") for i in range(20)]
        text = format_catalog(skills)
        self.assertIn("autre(s) compétence(s)", text)

    def test_format_active_includes_body_and_footer(self):
        skill = _skill("revue", "relit", body="# Revue\nÉtapes")
        text = format_active(skill)
        self.assertIn("COMPÉTENCE ACTIVE", text)
        self.assertIn("Étapes", text)

    def test_substitute_paths(self):
        skill = Skill(
            name="x", description="d", body="run ${SKILL_DIR}/scripts/a.py", path="/skills/x"
        )
        self.assertIn("/skills/x/scripts/a.py", substitute_paths(skill.body, skill))

    def test_build_skills_prompt_catalog_only(self):
        skills = [_skill("a", "fait A")]
        text = build_skills_prompt(skills, None)
        self.assertIn("COMPÉTENCES DISPONIBLES", text)
        self.assertNotIn("COMPÉTENCE ACTIVE", text)

    def test_build_skills_prompt_with_active(self):
        skills = [_skill("a", "fait A", body="corps A")]
        text = build_skills_prompt(skills, skills[0])
        self.assertIn("COMPÉTENCE ACTIVE", text)
        self.assertIn("corps A", text)


class EngineInjectionTests(unittest.TestCase):
    def test_build_messages_appends_skills_prompt(self):
        from lity.services.ai.ollama_engine import AIEngine

        engine = AIEngine(model="fake")
        engine.skills_prompt = "BLOC-COMPÉTENCE-TEST"
        messages = engine._build_messages([{"role": "user", "content": "salut"}])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("BLOC-COMPÉTENCE-TEST", messages[0]["content"])

    def test_build_messages_without_skills_prompt(self):
        from lity.services.ai.ollama_engine import AIEngine

        engine = AIEngine(model="fake")
        messages = engine._build_messages([{"role": "user", "content": "salut"}])
        self.assertNotIn("COMPÉTENCE", messages[0]["content"])


class _FakeSkillEngine:
    """Minimal engine exposing only what the skills path touches."""

    model = "fake-model"

    def __init__(self, pick=None):
        self.skills_prompt = ""
        self._pick = pick

    def get_installed_models(self):
        return ["fake-model"]

    def get_response(self, *args, **kwargs):
        return "Réponse."

    def generate_structured(self, prompt, schema, **kwargs):
        return self._pick

    def extract_fact(self, message):
        return None


class _FakeMemory:
    assistant_profile = {"nom": "Assistant"}

    def __init__(self):
        self.messages = []

    def add_message(self, role, content, images=None):
        self.messages.append((role, content))

    def get_context(self):
        return [{"role": role, "content": content} for role, content in self.messages]

    def get_user_info_summary(self):
        return ""

    def get_assistant_info_summary(self):
        return ""


class _FakeFiles:
    loaded_files = {}
    working_dir = None


class _FakeRouter:
    def process_intent(self, user_input, file_manager):
        return {"handled": False, "action": "none", "message": "", "system_context": ""}


class _FakeEditor:
    def parse_create_blocks(self, text):
        return []

    def parse_search_replace_blocks(self, text):
        return []


class _FakeImageManager:
    def is_active(self):
        return False


def _make_controller(tmp: str, *, pick=None):
    from lity.app.controller import AgentController
    from lity.app.services import AppServices
    from lity.infrastructure.paths import AppPaths
    from lity.infrastructure.settings import SettingsStore

    paths = AppPaths.create(home_override=Path(tmp))
    settings = SettingsStore(paths.settings_file)
    services = AppServices(
        settings=settings,
        engine=_FakeSkillEngine(pick=pick),
        memory=_FakeMemory(),
        files=_FakeFiles(),
        router=_FakeRouter(),
        editor=_FakeEditor(),
        image_manager=_FakeImageManager(),
    )
    return AgentController(paths=paths, services=services)


class ControllerSkillsTests(unittest.TestCase):
    def test_list_includes_builtin_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = _make_controller(tmp)
            data = controller.list_skills()
            names = [skill["name"] for skill in data["skills"]]
            self.assertIn("revue-de-code", names)
            self.assertTrue(data["enabled"])
            self.assertTrue(all(skill["enabled"] for skill in data["skills"]))

    def test_toggle_skill_disables_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = _make_controller(tmp)
            controller.toggle_skill("revue-de-code", False)
            data = controller.list_skills()
            revue = next(s for s in data["skills"] if s["name"] == "revue-de-code")
            self.assertFalse(revue["enabled"])
            enabled_names = [s.name for s in controller._enabled_skills()]
            self.assertNotIn("revue-de-code", enabled_names)

    def test_create_and_delete_user_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = _make_controller(tmp)
            created = controller.create_skill(
                "Traduction FR EN", "traduit du français vers l'anglais", "# corps"
            )
            self.assertTrue(created["ok"])
            names = [s["name"] for s in controller.list_skills()["skills"]]
            self.assertIn("traduction-fr-en", names)
            deleted = controller.delete_skill("traduction-fr-en")
            self.assertTrue(deleted["ok"])
            names_after = [s["name"] for s in controller.list_skills()["skills"]]
            self.assertNotIn("traduction-fr-en", names_after)

    def test_apply_skills_injects_catalog_and_active_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = _make_controller(tmp)
            controller._apply_skills("fais une revue de code et corrige les bugs")
            prompt = controller.engine.skills_prompt
            self.assertIn("COMPÉTENCES DISPONIBLES", prompt)
            self.assertIn("COMPÉTENCE ACTIVE", prompt)
            self.assertIn("revue-de-code", prompt)

    def test_apply_skills_catalog_only_for_unrelated_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = _make_controller(tmp)
            controller._apply_skills("bonjour, comment vas-tu aujourd'hui ?")
            prompt = controller.engine.skills_prompt
            self.assertIn("COMPÉTENCES DISPONIBLES", prompt)
            self.assertNotIn("COMPÉTENCE ACTIVE", prompt)

    def test_master_toggle_off_clears_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = _make_controller(tmp)
            controller.update_settings({"skills_enabled": False})
            controller._apply_skills("fais une revue de code")
            self.assertEqual(controller.engine.skills_prompt, "")

    def test_apply_skills_uses_model_pick(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = _make_controller(tmp, pick={"competence": "revue-de-code"})
            controller._apply_skills("regarde mon programme s'il te plaît")
            # The fake model forces revue-de-code even on a weak lexical match.
            self.assertIn("revue-de-code", controller.engine.skills_prompt)


class DesktopApiSkillsTests(unittest.TestCase):
    def _api(self, tmp: str):
        from lity.interfaces.desktop_web.api import DesktopApi

        return DesktopApi(_make_controller(tmp))

    def test_list_create_toggle_delete_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = self._api(tmp)
            listing = api.list_skills()
            self.assertIn("skills", listing)
            self.assertTrue(any(s["name"] == "revue-de-code" for s in listing["skills"]))

            created = api.create_skill("mon-outil", "fait quelque chose d'utile", "# corps")
            self.assertTrue(created["ok"])
            self.assertTrue(any(s["name"] == "mon-outil" for s in api.list_skills()["skills"]))

            toggled = api.toggle_skill("mon-outil", False)
            self.assertTrue(toggled["ok"])
            self.assertFalse(toggled["enabled"])

            deleted = api.delete_skill("mon-outil")
            self.assertTrue(deleted["ok"])
            self.assertFalse(any(s["name"] == "mon-outil" for s in api.list_skills()["skills"]))


if __name__ == "__main__":
    unittest.main()
