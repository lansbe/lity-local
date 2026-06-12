import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lity.app.controller import AgentController
from lity.app.services import AppServices
from lity.infrastructure.paths import AppPaths
from lity.interfaces.desktop_web.api import DesktopApi
from lity.services.memory.json_memory import MemoryManager
from lity.services.web.searxng_setup import CONTAINER_NAME, SearxngInstaller


class FakeDocker:
    """Scriptable docker CLI: records commands, returns canned outputs."""

    def __init__(self, *, daemon_up=True, container_state="absent", port="0.0.0.0:8080"):
        self.daemon_up = daemon_up
        self.container_state = container_state
        self.port = port
        self.commands: list[list[str]] = []
        self.fail_run = False

    def __call__(self, cmd, timeout):
        self.commands.append(list(cmd))
        if cmd[:2] == ["docker", "ps"] and "-a" not in cmd:
            return (0, "") if self.daemon_up else (1, "daemon down")
        if cmd[:3] == ["docker", "ps", "-a"]:
            return 0, "" if self.container_state == "absent" else self.container_state
        if cmd[:2] == ["docker", "start"]:
            self.container_state = "running"
            return 0, CONTAINER_NAME
        if cmd[:2] == ["docker", "run"]:
            if self.fail_run:
                return 1, "port already allocated"
            self.container_state = "running"
            return 0, "abc123"
        if cmd[:2] == ["docker", "port"]:
            return 0, self.port
        return 0, ""


def _installer(tmp, docker, *, reachable_after=0, free_ports=None):
    """Installer with injected docker/probe/clock — no real subprocess/network."""
    probes = {"count": 0}

    def probe(url):
        probes["count"] += 1
        return probes["count"] > reachable_after

    ports = set(free_ports if free_ports is not None else range(8080, 8090))
    clock = {"now": 0.0}

    installer = SearxngInstaller(
        Path(tmp),
        run_fn=docker,
        probe_fn=probe,
        sleep_fn=lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
        clock=lambda: clock["now"],
        port_free_fn=lambda port: port in ports,
    )
    return installer


class InstallerStatusTests(unittest.TestCase):
    def test_reachable_instance_short_circuits(self):
        docker = FakeDocker()
        installer = _installer(tempfile.gettempdir(), docker, reachable_after=0)
        status = installer.status("http://localhost:8080")
        self.assertTrue(status["reachable"])
        self.assertEqual(docker.commands, [])  # no docker calls when already up

    def test_unreachable_reports_docker_and_container(self):
        docker = FakeDocker(container_state="exited")
        installer = _installer(tempfile.gettempdir(), docker, reachable_after=99)
        status = installer.status("http://localhost:8080")
        self.assertFalse(status["reachable"])
        self.assertTrue(status["docker"])
        self.assertEqual(status["container"], "exited")


class InstallerInstallTests(unittest.TestCase):
    def _events(self):
        events = []
        return events, lambda kind, payload: events.append(payload)

    def test_fresh_install_writes_config_runs_container_and_persists_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            docker = FakeDocker(container_state="absent")
            installer = _installer(tmp, docker, reachable_after=2)
            events, on_event = self._events()
            persisted = []

            result = installer.install(on_event, persisted.append)

            self.assertTrue(result["ok"], result["message"])
            self.assertEqual(result["url"], "http://localhost:8080")
            self.assertEqual(persisted, ["http://localhost:8080"])
            # The auto-config enables the JSON API (without it: 403 for the app).
            config = (Path(tmp) / "searxng" / "settings.yml").read_text(encoding="utf-8")
            self.assertIn("- json", config)
            self.assertIn("secret_key", config)
            run_cmd = next(cmd for cmd in docker.commands if cmd[:2] == ["docker", "run"])
            self.assertIn("8080:8080", " ".join(run_cmd))
            self.assertIn(CONTAINER_NAME, run_cmd)

    def test_busy_port_falls_back_to_next_free_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            docker = FakeDocker(container_state="absent")
            installer = _installer(tmp, docker, reachable_after=1, free_ports={8082})
            events, on_event = self._events()
            persisted = []

            result = installer.install(on_event, persisted.append)

            self.assertTrue(result["ok"])
            self.assertEqual(result["url"], "http://localhost:8082")
            run_cmd = next(cmd for cmd in docker.commands if cmd[:2] == ["docker", "run"])
            self.assertIn("8082:8080", " ".join(run_cmd))

    def test_stopped_container_is_restarted_not_recreated(self):
        with tempfile.TemporaryDirectory() as tmp:
            docker = FakeDocker(container_state="exited")
            installer = _installer(tmp, docker, reachable_after=1)
            events, on_event = self._events()

            result = installer.install(on_event, lambda url: None)

            self.assertTrue(result["ok"])
            self.assertTrue(any(cmd[:2] == ["docker", "start"] for cmd in docker.commands))
            self.assertFalse(any(cmd[:2] == ["docker", "run"] for cmd in docker.commands))

    def test_without_docker_fails_with_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            docker = FakeDocker(daemon_up=False)
            installer = _installer(tmp, docker, reachable_after=99)
            installer.docker_ready = lambda: False  # docker binary absent
            events, on_event = self._events()

            result = installer.install(on_event, lambda url: None)

            self.assertFalse(result["ok"])
            self.assertIn("Docker", result["message"])

    def test_never_reachable_times_out_with_log_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            docker = FakeDocker(container_state="absent")
            installer = _installer(tmp, docker, reachable_after=10_000)
            events, on_event = self._events()

            result = installer.install(on_event, lambda url: None)

            self.assertFalse(result["ok"])
            self.assertIn("docker logs", result["message"])

    def test_existing_config_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "searxng" / "settings.yml"
            config.parent.mkdir(parents=True)
            config.write_text("# config personnalisée\n", encoding="utf-8")
            docker = FakeDocker(container_state="absent")
            installer = _installer(tmp, docker, reachable_after=1)

            installer.install(lambda *a: None, lambda url: None)

            self.assertEqual(config.read_text(encoding="utf-8"), "# config personnalisée\n")

    def test_install_succeeds_even_if_persist_callback_raises(self):
        # SearXNG is genuinely up; a failing persist must NOT flip the verdict to
        # failure (which would make the UI lie about a working install).
        with tempfile.TemporaryDirectory() as tmp:
            docker = FakeDocker(container_state="absent")
            installer = _installer(tmp, docker, reachable_after=1)

            def boom(_url: str) -> None:
                raise RuntimeError("settings locked")

            result = installer.install(lambda *a: None, boom)

            self.assertTrue(result["ok"])
            self.assertEqual(result["url"], "http://localhost:8080")


class _StubInstaller:
    def __init__(self, *, reachable=False, docker=True, ok=True):
        self.reachable = reachable
        self.docker = docker
        self.ok = ok
        self.persisted: list[str] = []

    def status(self, url):
        return {
            "url": url,
            "reachable": self.reachable,
            "docker": self.docker,
            "container": "absent",
        }

    def install(self, on_event, persist):
        on_event("searxng_setup", {"stage": "run", "message": "création…", "done": False})
        if self.ok:
            persist("http://localhost:8081")
            self.persisted.append("http://localhost:8081")
            return {"ok": True, "url": "http://localhost:8081", "message": "opérationnel"}
        return {"ok": False, "url": "", "message": "échec docker"}


class WebSetupApiTests(unittest.TestCase):
    def _api(self, installer):
        import time

        from test_desktop_web_api import _build_api

        tmp = tempfile.mkdtemp()
        api, events = _build_api(tmp)
        api._searxng_installer = installer
        return api, events, time

    def test_web_status_reports_reality_not_the_toggle(self):
        api, _events, _time = self._api(_StubInstaller(reachable=False, docker=True))
        status = api.web_status()
        self.assertFalse(status["reachable"])
        self.assertTrue(status["docker"])
        self.assertIn("fallback_ddg", status)
        self.assertFalse(status["setup_resolved"])
        self.assertFalse(status["setup_running"])

    def test_setup_runs_in_background_and_emits_done_event(self):
        installer = _StubInstaller(ok=True)
        api, events, time = self._api(installer)

        launched = api.setup_searxng()
        self.assertTrue(launched["ok"])
        deadline = time.time() + 5
        while time.time() < deadline:
            if any(p.get("done") for k, p in events if k == "searxng_setup"):
                break
            time.sleep(0.02)

        done = [p for k, p in events if k == "searxng_setup" and p.get("done")]
        self.assertEqual(len(done), 1)
        self.assertTrue(done[0]["ok"])
        self.assertEqual(done[0]["url"], "http://localhost:8081")
        self.assertEqual(installer.persisted, ["http://localhost:8081"])
        self.assertFalse(api.web_status()["setup_running"])  # flag released

    def test_failed_setup_reports_message(self):
        api, events, time = self._api(_StubInstaller(ok=False))
        api.setup_searxng()
        deadline = time.time() + 5
        while time.time() < deadline:
            if any(p.get("done") for k, p in events if k == "searxng_setup"):
                break
            time.sleep(0.02)
        done = [p for k, p in events if k == "searxng_setup" and p.get("done")][0]
        self.assertFalse(done["ok"])
        self.assertIn("échec", done["message"])

    def test_mark_resolved_is_safe_without_settings(self):
        api, _events, _time = self._api(_StubInstaller())
        self.assertTrue(api.mark_web_setup_resolved()["ok"])


class WebSetupResolvedPersistenceTests(unittest.TestCase):
    """The 'resolved' flag is the dismiss-bug fix: it persists ONLY on a real
    decision, so dismissing the modal never suppresses the offer."""

    def _services(self, tmp, settings):
        from test_desktop_web_api import FakeEditor, FakeFiles, FakeRouter, FakeStreamingEngine

        return AppServices(
            settings=settings,
            engine=FakeStreamingEngine(),
            memory=MemoryManager(paths=AppPaths.create(home_override=Path(tmp))),
            files=FakeFiles(),
            router=FakeRouter(),
            editor=FakeEditor(),
            image_manager=None,
        )

    def _api_with_settings(self, tmp):
        from lity.infrastructure.settings import SettingsStore

        paths = AppPaths.create(home_override=Path(tmp))
        settings = SettingsStore(paths.settings_file)
        controller = AgentController(paths=paths, services=self._services(tmp, settings))
        api = DesktopApi(controller, emit=lambda event, payload: None)
        api._searxng_installer = _StubInstaller(reachable=False, ok=True)
        return api, settings

    def test_web_search_defaults_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, _settings = self._api_with_settings(tmp)
            self.assertFalse(api.get_state()["web_search"])  # nothing installed → off
            self.assertFalse(api.web_status()["setup_resolved"])

    def test_mark_resolved_persists_and_status_reflects_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            api, settings = self._api_with_settings(tmp)
            self.assertFalse(api.web_status()["setup_resolved"])

            api.mark_web_setup_resolved()

            self.assertTrue(settings.get("web_setup_resolved"))
            self.assertTrue(api.web_status()["setup_resolved"])

    def test_stale_prompted_key_is_ignored(self):
        # A user stuck by the old "mark on show" bug had web_setup_prompted=True.
        # The new logic reads web_setup_resolved, so they get the offer again.
        with tempfile.TemporaryDirectory() as tmp:
            api, settings = self._api_with_settings(tmp)
            settings.set("web_setup_prompted", True)  # stale value from an earlier build
            self.assertFalse(api.web_status()["setup_resolved"])  # not suppressed

    def test_successful_install_marks_resolved_but_does_not_auto_enable_web(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            api, settings = self._api_with_settings(tmp)
            api.setup_searxng()
            deadline = time.time() + 5
            while time.time() < deadline:
                if settings.get("web_setup_resolved"):
                    break
                time.sleep(0.02)
            self.assertEqual(settings.get("searxng_url"), "http://localhost:8081")
            self.assertTrue(settings.get("web_setup_resolved"))
            # Installing makes web AVAILABLE; it must NOT silently turn the toggle
            # on — the user opts in (the UI flips it for the current session).
            self.assertFalse(settings.get("web_search_enabled"))
            self.assertFalse(api.get_state()["web_search"])

    def test_web_toggle_starts_off_even_if_persisted_on(self):
        # A previous session (or an older build) left web_search_enabled=True.
        # The toolbar must still start OFF, and the stale flag is cleared so the
        # health panel agrees.
        with tempfile.TemporaryDirectory() as tmp:
            _api, settings = self._api_with_settings(tmp)
            settings.set("web_search_enabled", True)
            api2 = DesktopApi(
                AgentController(
                    paths=AppPaths.create(home_override=Path(tmp)),
                    services=self._services(tmp, settings),
                ),
                emit=lambda event, payload: None,
            )
            self.assertFalse(api2.get_state()["web_search"])  # off on launch
            self.assertFalse(settings.get("web_search_enabled"))  # stale flag cleared

    def test_install_done_event_is_ok_even_if_settings_write_raises(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            api, _settings = self._api_with_settings(tmp)
            events: list[tuple[str, dict]] = []
            api.set_emit(lambda event, payload: events.append((event, payload)))
            # A settings write that blows up mid-install must not corrupt the
            # result the UI is told about (SearXNG really did come up).
            api.controller.update_settings = lambda patch: (_ for _ in ()).throw(
                RuntimeError("disk full")
            )

            api.setup_searxng()
            deadline = time.time() + 5
            while time.time() < deadline:
                if any(p.get("done") for k, p in events if k == "searxng_setup"):
                    break
                time.sleep(0.02)

            done = [p for k, p in events if k == "searxng_setup" and p.get("done")][0]
            self.assertTrue(done["ok"])  # persist failure didn't flip the verdict


if __name__ == "__main__":
    unittest.main()
