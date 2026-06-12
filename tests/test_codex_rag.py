import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.infrastructure.paths import AppPaths


class CodexRagTests(unittest.TestCase):
    def test_search_codex_rag_reads_memory_without_embeddings(self):
        from lity.services.codex_rag import search_codex_rag

        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths.create(home_override=Path(tmp))
            paths.facts_file.write_text(
                json.dumps(
                    {
                        "style": {
                            "value": "L'utilisateur préfère les réponses courtes en français.",
                            "count": 1,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (paths.conversations_dir / "conv.json").write_text(
                json.dumps(
                    {
                        "title": "Préférences",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Pour ce projet, garde les réponses courtes.",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            hits = search_codex_rag("réponses courtes", paths=paths, top_k=3)

            self.assertTrue(hits)
            self.assertTrue(any("réponses courtes" in hit["text"] for hit in hits))
            self.assertTrue(all(hit["source"] in {"facts", "conversation"} for hit in hits))
