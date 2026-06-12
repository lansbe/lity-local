import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.infrastructure.paths import AppPaths
from lity.services.memory.conversation_store import ConversationStore, derive_title
from lity.services.memory.json_memory import MemoryManager


class ConversationStoreTests(unittest.TestCase):
    def _store(self, tmp: str) -> ConversationStore:
        return ConversationStore(Path(tmp) / "conversations")

    def test_create_list_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            meta = store.create_conversation()

            self.assertIn("id", meta)
            self.assertEqual(meta["title"], "Nouvelle conversation")
            listed = store.list_conversations()
            self.assertEqual([item["id"] for item in listed], [meta["id"]])
            conversation = store.get_conversation(meta["id"])
            self.assertEqual(conversation["messages"], [])

    def test_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            ids = {store.create_conversation()["id"] for _ in range(5)}
            self.assertEqual(len(ids), 5)

    def test_add_message_and_persist_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]
            store.add_message(cid, "user", "bonjour")
            store.add_message(cid, "assistant", "salut")

            reloaded = self._store(tmp)
            messages = reloaded.get_messages(cid)
            self.assertEqual([m["content"] for m in messages], ["bonjour", "salut"])
            self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
            self.assertIn("timestamp", messages[0])

    def test_auto_title_from_first_user_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]
            store.add_message(cid, "user", "Explique-moi les décorateurs Python")

            self.assertEqual(
                store.get_conversation(cid)["title"], "Explique-moi les décorateurs Python"
            )
            # A second user message must not overwrite the title.
            store.add_message(cid, "user", "et les générateurs ?")
            self.assertEqual(
                store.get_conversation(cid)["title"], "Explique-moi les décorateurs Python"
            )

    def test_set_ai_title_applies_then_respects_user_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]
            store.add_message(cid, "user", "salut")

            # AI title overrides the message-derived default.
            self.assertTrue(store.set_ai_title(cid, "Discussion de test"))
            self.assertEqual(store.get_conversation(cid)["title"], "Discussion de test")

            # A manual rename locks the title against further AI titles.
            self.assertTrue(store.rename_conversation(cid, "Mon titre"))
            self.assertFalse(store.set_ai_title(cid, "Autre titre"))
            self.assertEqual(store.get_conversation(cid)["title"], "Mon titre")
            self.assertEqual(self._store(tmp).get_conversation(cid)["title"], "Mon titre")

    def test_instructions_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]
            self.assertEqual(store.get_instructions(cid), {"instructions": "", "temperature": None})
            store.set_instructions(cid, "Sois concise.", 0.4)
            self.assertEqual(
                store.get_instructions(cid), {"instructions": "Sois concise.", "temperature": 0.4}
            )
            self.assertEqual(
                self._store(tmp).get_instructions(cid),
                {"instructions": "Sois concise.", "temperature": 0.4},
            )

    def test_summary_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]
            self.assertEqual(store.get_summary(cid), ("", 0))
            store.set_summary(cid, "Résumé des points clés.", 3)
            self.assertEqual(store.get_summary(cid), ("Résumé des points clés.", 3))
            self.assertEqual(self._store(tmp).get_summary(cid), ("Résumé des points clés.", 3))

    def test_clear_messages_keeps_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]
            store.add_message(cid, "user", "salut")
            store.clear_messages(cid)

            self.assertEqual(store.get_messages(cid), [])
            self.assertTrue(store.exists(cid))

    def test_pop_last_assistant_and_replace_last_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]
            store.add_message(cid, "user", "première question")
            store.add_message(cid, "assistant", "première réponse")

            self.assertTrue(store.pop_last_assistant(cid))
            self.assertEqual([m["role"] for m in store.get_messages(cid)], ["user"])
            # Popping again when the tail is a user message does nothing.
            self.assertFalse(store.pop_last_assistant(cid))

            self.assertTrue(store.replace_last_user(cid, "question corrigée"))
            self.assertEqual(store.get_messages(cid)[-1]["content"], "question corrigée")

    def test_rename_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cid = store.create_conversation()["id"]

            self.assertTrue(store.rename_conversation(cid, "Mon titre"))
            self.assertEqual(store.get_conversation(cid)["title"], "Mon titre")
            self.assertFalse(store.rename_conversation(cid, "   "))

            self.assertTrue(store.delete_conversation(cid))
            self.assertFalse(store.exists(cid))
            self.assertEqual(store.list_conversations(), [])

    def test_search_matches_title_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            first = store.create_conversation()["id"]
            store.add_message(first, "user", "parle moi de la photosynthèse")
            second = store.create_conversation()["id"]
            store.add_message(second, "user", "recette de cuisine")

            ids = [meta["id"] for meta in store.search("photosynthèse")]
            self.assertEqual(ids, [first])
            self.assertEqual(len(store.search("")), 2)
            self.assertEqual(store.search("introuvable"), [])

    def test_ensure_default_creates_then_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            first = store.ensure_default()
            again = store.ensure_default()
            self.assertEqual(first, again)
            self.assertEqual(len(store.list_conversations()), 1)

    def test_missing_conversation_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            self.assertIsNone(store.get_conversation("nope"))
            self.assertEqual(store.get_messages("nope"), [])
            store.add_message("nope", "user", "x")  # no raise
            self.assertFalse(store.rename_conversation("nope", "x"))
            self.assertFalse(store.delete_conversation("nope"))

    def test_corrupt_index_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            (store.root / "index.json").write_text("not json", encoding="utf-8")
            self.assertEqual(store.list_conversations(), [])
            meta = store.create_conversation()
            self.assertEqual(len(store.list_conversations()), 1)
            self.assertEqual(store.list_conversations()[0]["id"], meta["id"])

    def test_derive_title_truncates(self):
        self.assertEqual(derive_title("  hello   world  "), "hello world")
        self.assertEqual(derive_title(""), "Nouvelle conversation")
        long = "x" * 100
        self.assertTrue(derive_title(long).endswith("…"))
        self.assertLessEqual(len(derive_title(long)), 48)


class MemoryConversationTests(unittest.TestCase):
    def test_active_conversation_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            memory = MemoryManager(paths=paths)

            first_id = memory.active_conversation_id
            self.assertTrue(first_id)
            memory.add_message("user", "dans la première")

            second = memory.new_conversation()
            self.assertNotEqual(second["id"], first_id)
            self.assertEqual(memory.get_context(), [])  # fresh conversation
            memory.add_message("user", "dans la seconde")

            self.assertTrue(memory.switch_conversation(first_id))
            self.assertEqual(memory.get_context()[-1]["content"], "dans la première")
            self.assertFalse(memory.switch_conversation("unknown-id"))

    def test_delete_active_falls_back_to_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            memory = MemoryManager(paths=paths)
            first_id = memory.active_conversation_id
            memory.add_message("user", "garde cette conversation")  # make it non-empty
            second = memory.new_conversation()  # distinct empty draft
            self.assertNotEqual(second["id"], first_id)

            new_active = memory.delete_conversation(second["id"])
            self.assertEqual(new_active, first_id)
            self.assertEqual(memory.active_conversation_id, first_id)

    def test_context_summary_for_long_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            memory = MemoryManager(paths=paths)
            for index in range(25):
                memory.add_message("user", f"message {index}")

            # 25 messages, window 20 → the first 5 scrolled out and need a summary.
            pending = memory.pending_summary()
            self.assertIsNotNone(pending)
            self.assertEqual(pending["count"], 5)
            self.assertEqual(len(pending["messages"]), 5)

            memory.set_conversation_summary("Résumé des 5 premiers messages.", 5)
            context = memory.get_context()
            self.assertEqual(context[0]["role"], "system")
            self.assertIn("RÉSUMÉ", context[0]["content"])
            # Nothing left to summarize once the covered count catches up.
            self.assertIsNone(memory.pending_summary())


if __name__ == "__main__":
    unittest.main()
