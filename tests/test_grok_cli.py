import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from lity.services.grok_cli import GrokCliClient


class GrokCliClientTests(unittest.TestCase):
    def test_status_reports_missing_cli(self):
        client = GrokCliClient(
            command="missing-grok",
            which=lambda _name: None,
            environ={},
            auth_file=Path("/nonexistent/auth.json"),
        )

        status = client.status()

        self.assertFalse(status["available"])
        self.assertFalse(status["authenticated"])
        self.assertIn("Grok CLI introuvable", status["message"])

    def test_status_authenticated_via_env_key(self):
        client = GrokCliClient(
            which=lambda _name: "grok",
            environ={"XAI_API_KEY": "xai-test"},
            auth_file=Path("/nonexistent/auth.json"),
        )

        status = client.status()

        self.assertTrue(status["available"])
        self.assertTrue(status["authenticated"])

    def test_status_authenticated_via_auth_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth = Path(tmp) / "auth.json"
            auth.write_text("{}", encoding="utf-8")
            client = GrokCliClient(which=lambda _name: "grok", environ={}, auth_file=auth)

            status = client.status()

            self.assertTrue(status["available"])
            self.assertTrue(status["authenticated"])

    def test_start_login_launches_grok_login(self):
        process = object()
        calls = []

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return process

        client = GrokCliClient(popen=popen, which=lambda _name: "grok")

        result = client.start_login()

        self.assertTrue(result["ok"], result["message"])
        self.assertIs(result["process"], process)
        command, kwargs = calls[0]
        self.assertEqual(command, ["grok", "login"])
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)

    def test_start_login_supports_device_auth(self):
        process = object()
        calls = []

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return process

        client = GrokCliClient(popen=popen, which=lambda _name: "grok")

        result = client.start_login(device_auth=True)

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(calls[0][0], ["grok", "login", "--device-auth"])

    def test_model_catalog_reads_available_models_from_grok_cli(self):
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                """
\x1b[2m2026-06-12T18:58:03Z\x1b[0m \x1b[33m WARN\x1b[0m config warning
You are logged in with grok.com.

Default model: grok-build

Available models:
  * grok-build (default)
  - grok-composer-2.5-fast
""",
                "",
            )

        client = GrokCliClient(runner=run, which=lambda _name: "grok")

        catalog = client.model_catalog()

        self.assertTrue(catalog["ok"], catalog["message"])
        self.assertEqual(catalog["default_model"], "grok-build")
        self.assertEqual(
            [model["slug"] for model in catalog["models"]],
            ["grok-build", "grok-composer-2.5-fast"],
        )
        # Grok has no reasoning-effort levels.
        self.assertEqual(catalog["models"][0]["supported_reasoning_levels"], [])

    def test_model_catalog_falls_back_when_grok_models_fails(self):
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 1, "", "not logged in")

        client = GrokCliClient(runner=run, which=lambda _name: "grok")

        catalog = client.model_catalog()

        self.assertTrue(catalog["ok"], catalog["message"])
        self.assertEqual(catalog["default_model"], "grok-build")
        self.assertIn("grok-build", [model["slug"] for model in catalog["models"]])

    def test_model_catalog_includes_custom_models_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                """
[model.team-coder]
model = "grok-build-0.1"
base_url = "https://api.example.com/v1"
name = "Team Coder"
env_key = "TEAM_XAI_KEY"

[models]
default = "team-coder"
""".strip(),
                encoding="utf-8",
            )
            client = GrokCliClient(which=lambda _name: "grok", config_files=[config])

            catalog = client.model_catalog()

            slugs = [model["slug"] for model in catalog["models"]]
            self.assertIn("team-coder", slugs)
            self.assertEqual(catalog["default_model"], "team-coder")
            custom = next(model for model in catalog["models"] if model["slug"] == "team-coder")
            self.assertEqual(custom["display_name"], "Team Coder")
            self.assertIn("https://api.example.com/v1", custom["description"])

    def test_run_prompt_uses_headless_print_mode(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {"text": "Réponse via Grok", "tokenUsage": {"input": 900, "output": 200}}
                ),
                "",
            )

        client = GrokCliClient(runner=run, which=lambda _name: "grok")
        with tempfile.TemporaryDirectory() as tmp:
            result = client.run_prompt("Analyse ce projet.", model="grok-build", workdir=Path(tmp))

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "Réponse via Grok")
        self.assertEqual(result["usage"]["input_tokens"], 900)
        self.assertEqual(result["usage"]["output_tokens"], 200)
        command, kwargs = calls[0]
        self.assertEqual(command[:2], ["grok", "-p"])
        self.assertEqual(command[2], "Analyse ce projet.")  # prompt is -p's value
        self.assertEqual(command[command.index("--output-format") + 1], "json")
        self.assertIn("--no-alt-screen", command)
        self.assertIn("--no-auto-update", command)
        self.assertEqual(command[command.index("--model") + 1], "grok-build")
        self.assertEqual(command[command.index("--cwd") + 1], tmp)
        self.assertEqual(kwargs["timeout"], 600)

    def test_run_prompt_remaps_removed_static_model_ids_to_grok_build(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, json.dumps({"text": "ok"}), "")

        client = GrokCliClient(runner=run, which=lambda _name: "grok")

        for old_slug in ("grok-build-0.1", "grok-4", "grok-3"):
            result = client.run_prompt("Salut", model=old_slug)
            self.assertTrue(result["ok"], result["message"])

        for command, _kwargs in calls:
            self.assertEqual(command[command.index("--model") + 1], "grok-build")

    def test_run_prompt_supports_headless_sessions_plugins_and_streaming_json(self):
        calls = []
        stream = "\n".join(
            [
                json.dumps({"type": "message_delta", "delta": {"text": "Bon"}}),
                json.dumps(
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "jour"},
                    }
                ),
                json.dumps({"usage": {"input_tokens": 10, "output_tokens": 5}}),
            ]
        )

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stream, "")

        client = GrokCliClient(runner=run, which=lambda _name: "grok")

        result = client.run_prompt(
            "Continue.",
            session_id="lity-abc123",
            output_format="streaming-json",
            plugin_dirs=[Path("/tmp/grok-plugin")],
        )

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "Bonjour")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 5)
        command = calls[0][0]
        self.assertEqual(command[command.index("--session-id") + 1], "lity-abc123")
        self.assertEqual(command[command.index("--output-format") + 1], "streaming-json")
        self.assertEqual(command[command.index("--plugin-dir") + 1], "/tmp/grok-plugin")
        self.assertNotIn("--always-approve", command)

    def test_run_prompt_streams_json_chunks_live_when_callback_is_provided(self):
        calls = []
        chunks = []

        class Stream:
            def __iter__(self):
                return iter(
                    [
                        json.dumps({"delta": {"text": "Bon"}}) + "\n",
                        json.dumps({"content": {"text": "jour"}}) + "\n",
                    ]
                )

        class Process:
            returncode = 0
            stdout = Stream()

            def __init__(self):
                self.stderr = self
                self.timeout = None
                self.killed = False

            def read(self):
                return ""

            def wait(self, timeout=None):
                self.timeout = timeout
                return self.returncode

            def kill(self):
                self.killed = True

        process = Process()

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return process

        client = GrokCliClient(popen=popen, which=lambda _name: "grok")

        result = client.run_prompt(
            "Salut",
            output_format="streaming-json",
            on_chunk=chunks.append,
        )

        self.assertTrue(result["ok"], result["message"])
        self.assertTrue(result["streamed"])
        self.assertEqual(result["content"], "Bonjour")
        self.assertEqual(chunks, ["Bon", "jour"])
        self.assertEqual(calls[0][0][calls[0][0].index("--output-format") + 1], "streaming-json")
        self.assertEqual(process.timeout, 600)

    def test_run_prompt_streaming_json_reads_grok_type_text_data_events(self):
        chunks = []

        class Stream:
            def __iter__(self):
                return iter(
                    [
                        json.dumps({"type": "thought", "data": "Thinking"}) + "\n",
                        json.dumps({"type": "text", "data": "OK"}) + "\n",
                        json.dumps({"type": "end", "stopReason": "EndTurn"}) + "\n",
                    ]
                )

        class Process:
            returncode = 0
            stdout = Stream()
            stderr = None

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                pass

        client = GrokCliClient(
            popen=lambda _command, **_kwargs: Process(), which=lambda _name: "grok"
        )

        result = client.run_prompt(
            "Salut",
            output_format="streaming-json",
            on_chunk=chunks.append,
        )

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "OK")
        self.assertEqual(chunks, ["OK"])

    def test_run_prompt_streaming_json_preserves_text_chunk_whitespace(self):
        chunks = []

        class Stream:
            def __iter__(self):
                return iter(
                    [
                        json.dumps({"type": "text", "data": "Salut"}) + "\n",
                        json.dumps({"type": "text", "data": " !"}) + "\n",
                    ]
                )

        class Process:
            returncode = 0
            stdout = Stream()
            stderr = None

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                pass

        client = GrokCliClient(
            popen=lambda _command, **_kwargs: Process(), which=lambda _name: "grok"
        )

        result = client.run_prompt(
            "Salut",
            output_format="streaming-json",
            on_chunk=chunks.append,
        )

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "Salut !")
        self.assertEqual(chunks, ["Salut", " !"])

    def test_run_prompt_can_resume_or_continue_existing_headless_sessions(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, json.dumps({"text": "ok"}), "")

        client = GrokCliClient(runner=run, which=lambda _name: "grok")

        resumed = client.run_prompt("Reprends.", resume="session-42")
        continued = client.run_prompt("Continue.", continue_session=True)

        self.assertTrue(resumed["ok"], resumed["message"])
        self.assertTrue(continued["ok"], continued["message"])
        self.assertEqual(calls[0][0][calls[0][0].index("--resume") + 1], "session-42")
        self.assertIn("--continue", calls[1][0])

    def test_run_prompt_tolerates_plain_text_output(self):
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, "Texte brut sans JSON", "")

        client = GrokCliClient(runner=run, which=lambda _name: "grok")

        result = client.run_prompt("Salut")

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["content"], "Texte brut sans JSON")
        self.assertIsNone(result["usage"])


if __name__ == "__main__":
    unittest.main()
