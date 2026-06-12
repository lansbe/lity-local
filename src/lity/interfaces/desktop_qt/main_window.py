from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from lity.interfaces.desktop_qt.widgets.chat_panel import ChatPanel
from lity.interfaces.desktop_qt.widgets.diff_review import DiffReviewWidget
from lity.interfaces.desktop_qt.widgets.image_result import ImageResultWidget
from lity.interfaces.desktop_qt.widgets.sidebar import Sidebar
from lity.interfaces.desktop_qt.workers import ChatWorker, ModelListWorker


class MainWindow(QMainWindow):
    def __init__(self, controller: Any):
        super().__init__()
        self.controller = controller
        self.worker_thread = None
        self.worker = None
        self.model_thread = None
        self.model_worker = None
        self.image_launch_timer = QTimer(self)
        self.image_launch_timer.setInterval(1500)
        self.image_launch_timer.timeout.connect(self._poll_image_launch)
        self.busy = False
        self._shutdown_complete = False
        self._streaming_response = False

        self.setWindowTitle("Lity")
        self.sidebar = Sidebar()
        self.chat = ChatPanel()
        self.activity_area = QScrollArea()
        self.activity_area.setWidgetResizable(True)
        self.activity_content = QWidget()
        self.activity_layout = QVBoxLayout(self.activity_content)
        self.activity_layout.addWidget(QLabel("Revue"))
        self.activity_layout.addStretch(1)
        self.activity_area.setWidget(self.activity_content)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self.sidebar, 0)
        layout.addWidget(self.chat, 1)
        layout.addWidget(self.activity_area, 0)
        self.setCentralWidget(root)

        self._connect_signals()
        self._apply_style()
        self.refresh_models()

    def _connect_signals(self) -> None:
        self.chat.send_requested.connect(self.handle_send)
        self.sidebar.model_selected.connect(self.handle_model_selected)
        self.sidebar.refresh_models_requested.connect(self.refresh_models)
        self.sidebar.workdir_selected.connect(self.handle_workdir_selected)
        self.sidebar.image_mode_toggled.connect(self.toggle_image_mode)
        self.sidebar.clear_requested.connect(self.clear_history)

    def refresh_models(self) -> None:
        if self._thread_is_running("model_thread"):
            return
        self.model_thread = QThread()
        self.model_worker = ModelListWorker(self.controller.engine)
        self.model_worker.moveToThread(self.model_thread)
        self.model_thread.started.connect(self.model_worker.run)
        self.model_worker.models_ready.connect(self._handle_models_ready)
        self.model_worker.error_ready.connect(
            lambda text: self.chat.append_system(f"Modèles indisponibles : {text}")
        )
        self.model_worker.finished.connect(self.model_thread.quit)
        self.model_thread.finished.connect(self.model_worker.deleteLater)
        self.model_thread.finished.connect(self.model_thread.deleteLater)
        self.model_thread.finished.connect(self._clear_model_worker)
        self.model_thread.start()

    def _handle_models_ready(self, models: list[str], _current_model: str) -> None:
        selected = self.controller.sync_available_models(models)
        self.sidebar.set_models(models, selected if models else "")

    def handle_model_selected(self, model_name: str) -> None:
        selected = self.controller.change_model(model_name)
        self.chat.append_system(f"Modèle actif : {selected}")

    def handle_workdir_selected(self, path: str) -> None:
        success, message = self.controller.files.set_working_dir(path)
        self.chat.append_system(message)
        if success:
            self.sidebar.set_workdir(str(self.controller.files.working_dir))

    def toggle_image_mode(self) -> None:
        if self.controller.is_image_session_active():
            self.image_launch_timer.stop()
            if self.controller.image_manager is not None:
                self.controller.image_manager.cancel_session()
            self.chat.append_system("Mode image désactivé.")
        elif self.image_launch_timer.isActive():
            self.chat.append_system(
                "La préparation du mode image est déjà en cours. "
                "Je l'active dès qu'elle est terminée."
            )
        else:
            result = self.controller.start_image_session()
            self.chat.append_system(result.get("message", "Mode image indisponible."))
            if result.get("status") in {"installing", "launched", "launching", "waiting"}:
                self.image_launch_timer.start()

    def _poll_image_launch(self) -> None:
        result = self.controller.poll_image_launch_status()
        if result.get("status") == "ready":
            self.image_launch_timer.stop()
            self.chat.append_system(result.get("message", "Mode image activé."))
        elif result.get("status") in {"error", "missing", "stopped", "no_model"}:
            self.image_launch_timer.stop()
            self.chat.append_system(result.get("message", "Le mode image n'a pas pu démarrer."))

    def clear_history(self) -> None:
        self.controller.clear_history()
        self.chat.append_system("Historique effacé.")

    def handle_send(self, message: str) -> None:
        if self.busy:
            self.chat.append_system("Une requête est déjà en cours.")
            return
        slash_result = self.controller.process_slash_command(message)
        self.chat.append_message("Vous", message, css_class="user")
        if slash_result:
            if slash_result.get("action") == "quit":
                self.close()
                return
            self.chat.append_system(str(slash_result.get("message", "")))
            return

        self.chat.set_busy(True)
        self.busy = True
        self.worker_thread = QThread()
        self.worker = ChatWorker(self.controller, message)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.chunk_ready.connect(self._append_stream_chunk)
        self.worker.result_ready.connect(self.render_result)
        self.worker.error_ready.connect(lambda text: self.chat.append_system(f"Erreur : {text}"))
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self._mark_idle)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_chat_worker)
        self.worker_thread.start()

    def _mark_idle(self) -> None:
        self.busy = False
        self.chat.set_busy(False)

    def _clear_model_worker(self) -> None:
        self.model_worker = None
        self.model_thread = None

    def _clear_chat_worker(self) -> None:
        self.worker = None
        self.worker_thread = None

    def _clear_thread_refs(self, attr_name: str) -> None:
        if attr_name == "worker_thread":
            self._clear_chat_worker()
        elif attr_name == "model_thread":
            self._clear_model_worker()
        else:
            setattr(self, attr_name, None)

    def _thread_is_running(self, attr_name: str) -> bool:
        thread = getattr(self, attr_name)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            self._clear_thread_refs(attr_name)
            return False

    def _cancel_chat_worker(self) -> None:
        if self.worker is None or not hasattr(self.worker, "cancel"):
            return
        try:
            self.worker.cancel()
        except RuntimeError:
            self._clear_chat_worker()

    def _stop_thread(self, attr_name: str) -> None:
        thread = getattr(self, attr_name)
        if thread is None:
            return
        try:
            is_running = thread.isRunning()
        except RuntimeError:
            self._clear_thread_refs(attr_name)
            return
        try:
            if not is_running:
                self._clear_thread_refs(attr_name)
                return
            thread.quit()
            stopped = thread.wait(500)
        except RuntimeError:
            self._clear_thread_refs(attr_name)
            return
        if stopped:
            self._clear_thread_refs(attr_name)

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self.image_launch_timer.stop()
        self._cancel_chat_worker()
        self._stop_thread("worker_thread")
        self._stop_thread("model_thread")
        if hasattr(self.controller, "shutdown"):
            self.controller.shutdown()

    def render_result(self, result: dict) -> None:
        result_type = result.get("type")
        if result.get("system_notification"):
            self.chat.append_system(result["system_notification"])
        if result_type in {"ai_response", "text"}:
            if self._streaming_response:
                self.chat.finish_stream_message(result.get("content", ""))
                self._streaming_response = False
            else:
                self.chat.append_message(self.controller.assistant_name, result.get("content", ""))
            self._render_file_blocks(result)
        elif result_type == "intent_handled":
            self.chat.append_system(result.get("message", ""))
        elif result_type == "error":
            self.chat.append_message(self.controller.assistant_name, result.get("message", ""))
        elif result_type == "image_parameters_proposal":
            self.chat.append_system(result.get("message", ""))
            self._add_activity_text(
                json.dumps(result.get("content", {}), indent=2, ensure_ascii=False)
            )
        elif result_type == "image_generation_result":
            self.chat.append_system(result.get("message", "Image générée."))
            self._add_activity_widget(ImageResultWidget(result.get("content", {})))
        elif result_type in {"image_cancelled", "image_normal_chat"}:
            self.chat.append_system(result.get("message", ""))
        else:
            self.chat.append_system(str(result))

    def _append_stream_chunk(self, chunk: str) -> None:
        if not self._streaming_response:
            self.chat.start_stream_message(self.controller.assistant_name)
            self._streaming_response = True
        self.chat.append_stream_chunk(chunk, self.controller.assistant_name)

    def _render_file_blocks(self, result: dict) -> None:
        for block in result.get("create_blocks", []):
            widget = DiffReviewWidget(block, "create")
            widget.apply_requested.connect(self._apply_create)
            self._add_activity_widget(widget)
        for block in result.get("edit_blocks", []):
            widget = DiffReviewWidget(block, "edit")
            widget.apply_requested.connect(self._apply_edit)
            self._add_activity_widget(widget)

    def _apply_create(self, block: dict) -> None:
        success, message = self.controller.apply_create_block(block)
        self._show_apply_result(success, message)

    def _apply_edit(self, block: dict) -> None:
        success, message = self.controller.apply_edit_block(block)
        self._show_apply_result(success, message)

    def _show_apply_result(self, success: bool, message: str) -> None:
        self.chat.append_system(message)
        if not success:
            QMessageBox.warning(self, "Application impossible", message)

    def _add_activity_text(self, text: str) -> None:
        label = QLabel(text)
        label.setWordWrap(True)
        self._add_activity_widget(label)

    def _add_activity_widget(self, widget: QWidget) -> None:
        self.activity_layout.insertWidget(max(0, self.activity_layout.count() - 1), widget)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #101214; color: #ecf0f1; font-size: 14px; }
            #Sidebar { background: #171a1f; min-width: 250px; max-width: 300px; }
            #AppTitle { font-size: 24px; font-weight: 700; }
            #PanelTitle, #CardTitle { font-size: 16px; font-weight: 700; }
            QTextBrowser, QPlainTextEdit, QTextEdit, QLineEdit, QComboBox {
                background: #1e2329; border: 1px solid #303844; border-radius: 6px; padding: 8px;
            }
            #BusyIndicator {
                background: #162032; color: #bfdbfe; border: 1px solid #29405f;
                border-radius: 6px; padding: 8px 10px; font-weight: 600;
            }
            QPushButton {
                background: #3b82f6; border: 0; border-radius: 6px; padding: 10px 14px; font-weight: 600;
            }
            QPushButton:disabled { background: #3b4655; color: #aab4c0; }
            QScrollArea { border: 0; }
            """
        )
