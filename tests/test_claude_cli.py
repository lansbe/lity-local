import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from lity.services.claude_cli import ClaudeCliClient


class ClaudeCliClientTests(unittest.TestCase):
    def test_status_reports_missing_cli(self):
        client = ClaudeCliClient(command="missing-claude", which=lambda _name: None, environ={})

        status = client.status()

        self.assertFalse(status["available"])
        self.assertFalse(status["authenticated"])
        self.assertIn("Claude CLI introuvable", status["message"])

    def test_status_reports_logged_in_account(self):
        def run(command, **_kwargs):
            self.assertEqual(command, ["claude", "auth", "status"])
            return subprocess.CompletedProcess(command, 0, "Logged in as you@example.com\n", "")

        client = ClaudeCliClient(runner=run, which=lambda _name: "claude", environ={})

        status = client.status()

        self.assertTrue(status["available"])
        self.assertTrue(status["authenticated"])
        self.assertIn("Logged in", status["message"])

    def test_status_falls_back_to_env_credentials(self):
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 1, "", "not logged in")

        client = ClaudeCliClient(
            runner=run,
            which=lambda _name: "claude",
            environ={"ANTHROPIC_API_KEY": "sk-test"},
        )

        status = client.status()

        self.assertTrue(status["available"])
        self.assertTrue(status["authenticated"])

    def test_start_login_launches_setup_token_without_api_key(self):
        class FakeProcess:
            pass

        process = FakeProcess()
        calls = []

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return process

        client = ClaudeCliClient(popen=popen, which=lambda _name: "claude")

        result = client.start_login()

        self.assertTrue(result["ok"], result["message"])
        self.assertIs(result["process"], process)
        command, kwargs = calls[0]
        self.assertEqual(command, ["claude", "setup-token"])
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(kwargs["text"])

    def test_model_catalog_returns_full_lineup(self):
        client = ClaudeCliClient(which=lambda _name: "claude")

        catalog = client.model_catalog()

        self.assertTrue(catalog["ok"], catalog["message"])
        self.assertEqual(catalog["default_model"], "claude-opus-4-8")
        self.assertEqual(
            [model["slug"] for model in catalog["models"]],
            [
                "claude-fable-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-haiku-4-5",
            ],
        )
        by_slug = {model["slug"]: model for model in catalog["models"]}
        # Fable / Opus 4.8 expose the full effort range including xhigh + max.
        self.assertEqual(
            [level["effort"] for level in by_slug["claude-opus-4-8"]["supported_reasoning_levels"]],
            ["low", "medium", "high", "xhigh", "max"],
        )
        # Sonnet 4.6 has max but not xhigh.
        self.assertEqual(
            [
                level["effort"]
                for level in by_slug["claude-sonnet-4-6"]["supported_reasoning_levels"]
            ],
            ["low", "medium", "high", "max"],
        )
        # Haiku exposes no effort levels at all.
        self.assertEqual(by_slug["claude-haiku-4-5"]["supported_reasoning_levels"], [])
        self.assertEqual(by_slug["claude-opus-4-8"]["default_reasoning_level"], "high")

    def test_run_prompt_uses_print_mode_with_plan_permission(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"result": "Réponse via Claude"}), ""
            )

        client = ClaudeCliClient(runner=run, which=lambda _name: "claude")
        with tempfile.TemporaryDirectory() as tmp:
            result = client.run_prompt(
                "Analyse ce projet.",
                model="opus",
                reasoning_effort="high",
                workdir=Path(tmp),
            )

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "Réponse via Claude")
        command, kwargs = calls[0]
        self.assertEqual(command[:2], ["claude", "-p"])
        self.assertIn("--output-format", command)
        self.assertEqual(command[command.index("--output-format") + 1], "json")
        self.assertIn("--permission-mode", command)
        self.assertEqual(command[command.index("--permission-mode") + 1], "plan")
        self.assertEqual(command[command.index("--model") + 1], "opus")
        self.assertEqual(command[command.index("--effort") + 1], "high")
        # The prompt is the trailing positional argument.
        self.assertEqual(command[-1], "Analyse ce projet.")
        self.assertEqual(kwargs["cwd"], tmp)
        self.assertEqual(kwargs["timeout"], 600)

    def test_run_prompt_rejects_invalid_reasoning_effort(self):
        client = ClaudeCliClient(which=lambda _name: "claude")

        result = client.run_prompt("Bonjour", reasoning_effort="turbo")

        self.assertFalse(result["ok"])
        self.assertIn("raisonnement invalide", result["message"].lower())

    def test_run_prompt_drops_effort_for_models_without_effort(self):
        calls = []

        def run(command, **_kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, json.dumps({"result": "ok"}), "")

        client = ClaudeCliClient(runner=run, which=lambda _name: "claude")

        # Haiku has no effort parameter — it must not reach the command line.
        client.run_prompt("Salut", model="claude-haiku-4-5", reasoning_effort="high")
        self.assertNotIn("--effort", calls[0])

    def test_run_prompt_drops_unsupported_effort_level(self):
        calls = []

        def run(command, **_kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, json.dumps({"result": "ok"}), "")

        client = ClaudeCliClient(runner=run, which=lambda _name: "claude")

        # Sonnet 4.6 supports max but not xhigh — xhigh must be dropped.
        client.run_prompt("Salut", model="claude-sonnet-4-6", reasoning_effort="xhigh")
        self.assertNotIn("--effort", calls[0])
        calls.clear()
        # max is supported on Sonnet 4.6 — it stays.
        client.run_prompt("Salut", model="claude-sonnet-4-6", reasoning_effort="max")
        self.assertIn("--effort", calls[0])
        self.assertEqual(calls[0][calls[0].index("--effort") + 1], "max")

    def test_run_prompt_tolerates_plain_text_output(self):
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, "Texte brut sans JSON", "")

        client = ClaudeCliClient(runner=run, which=lambda _name: "claude")

        result = client.run_prompt("Salut")

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "Texte brut sans JSON")

    def test_run_prompt_reports_cost_and_per_model_usage(self):
        payload = {
            "result": "Réponse",
            "total_cost_usd": 0.42,
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 300,
                "cache_read_input_tokens": 200,
            },
            "modelUsage": {
                "claude-opus-4-8": {"input_tokens": 1000, "output_tokens": 300},
            },
        }

        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        client = ClaudeCliClient(runner=run, which=lambda _name: "claude")

        usage = client.run_prompt("Salut")["usage"]

        self.assertEqual(usage["cost_usd"], 0.42)
        self.assertEqual(usage["input_tokens"], 1200)  # input + cache_read
        self.assertEqual(usage["output_tokens"], 300)
        self.assertIn("claude-opus-4-8", usage["by_model"])
        self.assertEqual(usage["by_model"]["claude-opus-4-8"]["output_tokens"], 300)


if __name__ == "__main__":
    unittest.main()
