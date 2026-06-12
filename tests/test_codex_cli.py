import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from lity.services.codex_cli import CodexCliClient


class CodexCliClientTests(unittest.TestCase):
    def test_status_reports_missing_cli(self):
        client = CodexCliClient(command="missing-codex", which=lambda _name: None)

        status = client.status()

        self.assertFalse(status["available"])
        self.assertFalse(status["authenticated"])
        self.assertIn("Codex CLI introuvable", status["message"])

    def test_status_reports_chatgpt_login(self):
        def run(command, **_kwargs):
            self.assertEqual(command, ["codex", "login", "status"])
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT\n", "")

        client = CodexCliClient(runner=run, which=lambda _name: "codex")

        status = client.status()

        self.assertTrue(status["available"])
        self.assertTrue(status["authenticated"])
        self.assertIn("ChatGPT", status["message"])

    def test_start_login_launches_codex_login_without_api_key(self):
        class FakeProcess:
            pass

        process = FakeProcess()
        calls = []

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return process

        client = CodexCliClient(popen=popen, which=lambda _name: "codex")

        result = client.start_login()

        self.assertTrue(result["ok"], result["message"])
        self.assertIs(result["process"], process)
        command, kwargs = calls[0]
        self.assertEqual(command, ["codex", "login"])
        self.assertNotIn("--with-api-key", command)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(kwargs["text"])

    def test_model_catalog_uses_codex_debug_models_and_sanitizes_output(self):
        payload = {
            "models": [
                {
                    "slug": "hidden",
                    "display_name": "Hidden",
                    "visibility": "hidden",
                    "priority": 1,
                    "base_instructions": "do not leak me",
                },
                {
                    "slug": "gpt-5.4-mini",
                    "display_name": "GPT-5.4-Mini",
                    "description": "Fast model.",
                    "default_reasoning_level": "medium",
                    "supported_reasoning_levels": [
                        {"effort": "low", "description": "Fast"},
                        {"effort": "medium", "description": "Balanced"},
                        {"effort": "turbo", "description": "Invalid"},
                    ],
                    "visibility": "list",
                    "priority": 20,
                    "base_instructions": "do not leak me",
                },
                {
                    "slug": "gpt-5.5",
                    "display_name": "GPT-5.5",
                    "description": "Frontier model.",
                    "default_reasoning_level": "high",
                    "supported_reasoning_levels": [
                        {"effort": "medium", "description": "Balanced"},
                        {"effort": "high", "description": "Deep"},
                    ],
                    "visibility": "list",
                    "priority": 10,
                },
            ]
        }

        def run(command, **kwargs):
            self.assertEqual(command, ["codex", "debug", "models"])
            self.assertEqual(kwargs["timeout"], 20)
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        client = CodexCliClient(runner=run, which=lambda _name: "codex")

        catalog = client.model_catalog()

        self.assertTrue(catalog["ok"], catalog["message"])
        self.assertEqual(catalog["default_model"], "gpt-5.5")
        self.assertEqual(
            [model["slug"] for model in catalog["models"]], ["gpt-5.5", "gpt-5.4-mini"]
        )
        self.assertNotIn("base_instructions", catalog["models"][0])
        self.assertEqual(
            [level["effort"] for level in catalog["models"][1]["supported_reasoning_levels"]],
            ["low", "medium"],
        )

    def test_run_prompt_uses_codex_exec_with_read_only_sandbox(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("Réponse via Codex\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, '{"event":"done"}\n', "")

        client = CodexCliClient(runner=run, which=lambda _name: "codex")
        with tempfile.TemporaryDirectory() as tmp:
            result = client.run_prompt(
                "Analyse ce projet.",
                model="gpt-5.5",
                reasoning_effort="high",
                workdir=Path(tmp),
            )

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "Réponse via Codex")
        command, kwargs = calls[0]
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertIn("--sandbox", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertIn("--model", command)
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.5")
        self.assertIn("model_reasoning_effort='high'", command)
        self.assertIn("--cd", command)
        self.assertEqual(kwargs["timeout"], 600)

    def test_run_prompt_rejects_invalid_reasoning_effort(self):
        client = CodexCliClient(which=lambda _name: "codex")

        result = client.run_prompt("Bonjour", reasoning_effort="turbo")

        self.assertFalse(result["ok"])
        self.assertIn("raisonnement invalide", result["message"].lower())

    def test_run_prompt_reports_token_usage_from_json_events(self):
        events = "\n".join(
            [
                '{"type":"event_msg","payload":{"type":"agent_message","message":"hi"}}',
                (
                    '{"type":"event_msg","payload":{"type":"token_count","info":'
                    '{"total_token_usage":{"input_tokens":900,"cached_input_tokens":100,'
                    '"output_tokens":250,"reasoning_output_tokens":50,"total_tokens":1300}},'
                    '"rate_limits":null}}'
                ),
            ]
        )

        def run(command, **_kwargs):
            self.assertIn("--json", command)
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("Réponse\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, events, "")

        client = CodexCliClient(runner=run, which=lambda _name: "codex")
        result = client.run_prompt("Analyse", model="gpt-5.5")

        self.assertEqual(result["content"], "Réponse")
        usage = result["usage"]
        self.assertEqual(usage["input_tokens"], 1000)  # input + cached
        self.assertEqual(usage["output_tokens"], 300)  # output + reasoning

    def test_run_prompt_reconstructs_message_when_output_file_empty(self):
        events = '{"type":"event_msg","payload":{"type":"agent_message","message":"Salut !"}}'

        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, events, "")

        client = CodexCliClient(runner=run, which=lambda _name: "codex")
        result = client.run_prompt("Bonjour")

        self.assertEqual(result["content"], "Salut !")


if __name__ == "__main__":
    unittest.main()
