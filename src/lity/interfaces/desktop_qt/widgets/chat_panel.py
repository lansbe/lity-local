from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ChatPanel(QWidget):
    send_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self._messages: list[tuple[str, str, str]] = []
        self._stream_index: int | None = None
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(True)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Écris un message...")
        self.input.setFixedHeight(92)
        self.send_button = QPushButton("Envoyer")
        self.send_button.clicked.connect(self._send)
        self.busy_label = QLabel("")
        self.busy_label.setObjectName("BusyIndicator")
        self.busy_label.hide()
        self._busy_frames = [
            "Réponse en cours",
            "Réponse en cours.",
            "Réponse en cours..",
            "Réponse en cours...",
        ]
        self._busy_frame_index = 0
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(360)
        self._busy_timer.timeout.connect(self._advance_busy_animation)

        header = QLabel("Conversation")
        header.setObjectName("PanelTitle")

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input, 1)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(self.transcript, 1)
        layout.addWidget(self.busy_label)
        layout.addLayout(input_layout)

    def append_message(self, sender: str, message: str, css_class: str = "assistant") -> None:
        self._messages.append((sender, message, css_class))
        self._render_transcript()

    def append_system(self, message: str) -> None:
        self.append_message("Système", message, css_class="system")

    def start_stream_message(self, sender: str) -> None:
        self._stream_index = len(self._messages)
        self._messages.append((sender, "", "assistant"))
        self._render_transcript()

    def append_stream_chunk(self, chunk: str, sender: str) -> None:
        if self._stream_index is None:
            self.start_stream_message(sender)
        stream_index = self._stream_index
        current_sender, current_message, css_class = self._messages[stream_index]
        self._messages[stream_index] = (current_sender, current_message + chunk, css_class)
        self._render_transcript()

    def finish_stream_message(self, final_message: str | None = None) -> None:
        if self._stream_index is not None and final_message is not None:
            sender, _message, css_class = self._messages[self._stream_index]
            self._messages[self._stream_index] = (sender, final_message, css_class)
        self._stream_index = None
        self._render_transcript()

    def set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.send_button.setText("Réflexion..." if busy else "Envoyer")
        if busy:
            self._busy_frame_index = 0
            self._render_busy_frame()
            self.busy_label.show()
            if not self._busy_timer.isActive():
                self._busy_timer.start()
        else:
            self._busy_timer.stop()
            self.busy_label.clear()
            self.busy_label.hide()

    def _advance_busy_animation(self) -> None:
        self._busy_frame_index = (self._busy_frame_index + 1) % len(self._busy_frames)
        self._render_busy_frame()

    def _render_busy_frame(self) -> None:
        self.busy_label.setText(self._busy_frames[self._busy_frame_index])

    def _send(self) -> None:
        message = self.input.toPlainText().strip()
        if not message:
            return
        self.input.clear()
        self.send_requested.emit(message)

    def _render_transcript(self) -> None:
        blocks = []
        for sender, message, css_class in self._messages:
            safe_message = _html_escape(message).replace("\n", "<br>")
            blocks.append(
                f'<div class="{css_class}"><b>{_html_escape(sender)}</b><br>{safe_message}</div><br>'
            )
        self.transcript.setHtml("".join(blocks))


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
