import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DesktopQtContractTests(unittest.TestCase):
    def test_chat_worker_exposes_cancel_contract_when_qt_available(self):
        try:
            from lity.interfaces.desktop_qt.workers import ChatWorker
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        worker = ChatWorker(controller=object(), message="hello")

        self.assertTrue(hasattr(worker, "cancel"))
        self.assertTrue(hasattr(worker, "is_cancelled"))

    def test_chat_worker_passes_cancellation_callback_to_compatible_controller(self):
        try:
            from lity.interfaces.desktop_qt.workers import ChatWorker
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        class Controller:
            def __init__(self):
                self.should_cancel = None

            def process_user_message_sync(self, message, should_cancel=None):
                self.should_cancel = should_cancel
                return {"type": "text", "content": message}

        controller = Controller()
        worker = ChatWorker(controller=controller, message="hello")

        worker.run()

        self.assertTrue(callable(controller.should_cancel))
        self.assertFalse(controller.should_cancel())
        worker.cancel()
        self.assertTrue(controller.should_cancel())

    def test_chat_worker_streams_partial_chunks_when_controller_supports_streaming(self):
        try:
            from lity.interfaces.desktop_qt.workers import ChatWorker
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        class Controller:
            def process_user_message_stream(self, message, on_chunk, should_cancel=None):
                on_chunk("Bon")
                on_chunk("jour")
                return {"type": "ai_response", "content": "Bonjour"}

        chunks = []
        results = []
        worker = ChatWorker(controller=Controller(), message="salut")
        worker.chunk_ready.connect(chunks.append)
        worker.result_ready.connect(results.append)

        worker.run()

        self.assertEqual(chunks, ["Bon", "jour"])
        self.assertEqual(results, [{"type": "ai_response", "content": "Bonjour"}])

    def test_chat_panel_shows_animated_busy_state(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication

            from lity.interfaces.desktop_qt.widgets.chat_panel import ChatPanel
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        app = QApplication.instance() or QApplication([])
        self.addCleanup(app.processEvents)
        panel = ChatPanel()

        self.assertTrue(panel.busy_label.isHidden())

        panel.set_busy(True)
        first_frame = panel.busy_label.text()
        panel._advance_busy_animation()

        self.assertFalse(panel.busy_label.isHidden())
        self.assertIn("Réponse en cours", first_frame)
        self.assertNotEqual(panel.busy_label.text(), first_frame)

        panel.set_busy(False)

        self.assertTrue(panel.busy_label.isHidden())
        self.assertEqual(panel.send_button.text(), "Envoyer")

    def test_model_list_worker_returns_models_without_main_window_blocking_contract(self):
        try:
            from lity.interfaces.desktop_qt import workers
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        class Engine:
            model = "llama3"

            def get_installed_models(self):
                return ["llama3", "mistral"]

        received = []
        worker = workers.ModelListWorker(Engine())
        worker.models_ready.connect(lambda models, current: received.append((models, current)))

        worker.run()

        self.assertEqual(received, [(["llama3", "mistral"], "llama3")])

    def test_sidebar_lists_only_installed_models_when_selected_model_is_missing(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication

            from lity.interfaces.desktop_qt.widgets.sidebar import Sidebar
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        app = QApplication.instance() or QApplication([])
        self.addCleanup(app.processEvents)
        sidebar = Sidebar()

        sidebar.set_models(["mistral:latest"], "llama3")

        items = [
            sidebar.model_combo.itemText(index) for index in range(sidebar.model_combo.count())
        ]
        self.assertEqual(items, ["mistral:latest"])
        self.assertEqual(sidebar.model_combo.currentText(), "mistral:latest")

    def test_sidebar_does_not_display_unavailable_model_when_ollama_has_no_models(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication

            from lity.interfaces.desktop_qt.widgets.sidebar import Sidebar
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        app = QApplication.instance() or QApplication([])
        self.addCleanup(app.processEvents)
        sidebar = Sidebar()

        sidebar.set_models([], "llama3")

        self.assertEqual(sidebar.model_combo.count(), 0)

    def test_main_window_shutdown_is_explicit_and_idempotent(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication

            from lity.interfaces.desktop_qt.main_window import MainWindow
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        class Engine:
            model = "llama3"

            def get_installed_models(self):
                return []

        class Files:
            working_dir = Path.cwd()

        class ImageManager:
            def is_active(self):
                return False

            def cancel_session(self):
                return None

        class Controller:
            def __init__(self):
                self.engine = Engine()
                self.files = Files()
                self.image_manager = ImageManager()
                self.shutdown_calls = 0

            def sync_available_models(self, models):
                return models[0] if models else ""

            def shutdown(self):
                self.shutdown_calls += 1

        app = QApplication.instance() or QApplication([])
        controller = Controller()
        window = MainWindow(controller)
        self.addCleanup(window.deleteLater)
        self.addCleanup(app.processEvents)

        window.shutdown()
        window.shutdown()

        self.assertNotIn("closeEvent", MainWindow.__dict__)
        self.assertEqual(controller.shutdown_calls, 1)

    def test_main_window_shutdown_tolerates_already_deleted_worker_thread(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication

            from lity.interfaces.desktop_qt.main_window import MainWindow
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        class DeletedThread:
            def isRunning(self):
                raise RuntimeError(
                    "libshiboken: Internal C++ object (PySide6.QtCore.QThread) already deleted."
                )

        class Worker:
            def __init__(self):
                self.cancelled = False

            def cancel(self):
                self.cancelled = True

        class Engine:
            model = "llama3"

            def get_installed_models(self):
                return []

        class Files:
            working_dir = Path.cwd()

        class Controller:
            def __init__(self):
                self.engine = Engine()
                self.files = Files()
                self.image_manager = None
                self.shutdown_calls = 0

            def sync_available_models(self, models):
                return ""

            def shutdown(self):
                self.shutdown_calls += 1

        app = QApplication.instance() or QApplication([])
        controller = Controller()
        window = MainWindow(controller)
        self.addCleanup(window.deleteLater)
        self.addCleanup(app.processEvents)
        worker = Worker()
        window.worker = worker
        window.worker_thread = DeletedThread()

        window.shutdown()

        self.assertTrue(worker.cancelled)
        self.assertEqual(controller.shutdown_calls, 1)
        self.assertIsNone(window.worker_thread)

    def test_main_window_clears_chat_worker_refs_when_thread_finishes(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication

            from lity.interfaces.desktop_qt.main_window import MainWindow
        except ImportError:
            self.skipTest("PySide6 is not installed in this environment")

        class Engine:
            model = "llama3"

            def get_installed_models(self):
                return []

        class Files:
            working_dir = Path.cwd()

        class Controller:
            assistant_name = "Assistant"

            def __init__(self):
                self.engine = Engine()
                self.files = Files()
                self.image_manager = None

            def sync_available_models(self, models):
                return ""

            def process_slash_command(self, _message):
                return None

            def process_user_message_stream(self, message, on_chunk, should_cancel=None):
                on_chunk(message)
                return {"type": "ai_response", "content": message}

            def shutdown(self):
                return None

        app = QApplication.instance() or QApplication([])
        window = MainWindow(Controller())
        self.addCleanup(window.shutdown)
        self.addCleanup(window.deleteLater)
        self.addCleanup(app.processEvents)

        window.handle_send("salut")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and (window.busy or window.worker_thread is not None):
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

        self.assertFalse(window.busy)
        self.assertIsNone(window.worker)
        self.assertIsNone(window.worker_thread)

    def test_desktop_import_error_message_mentions_desktop_extra(self):
        from lity.interfaces.desktop_qt.app import _missing_pyside_message

        message = _missing_pyside_message("No module named PySide6")

        self.assertIn("--extra desktop", message)
        self.assertIn("PySide6", message)


if __name__ == "__main__":
    unittest.main()
