from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class DiffReviewWidget(QWidget):
    apply_requested = Signal(dict)

    def __init__(self, block: dict, block_type: str):
        super().__init__()
        self.block = block
        self.block_type = block_type

        title = QLabel(
            f"{'Création' if block_type == 'create' else 'Modification'}: {block['file_path']}"
        )
        title.setObjectName("CardTitle")
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setMaximumHeight(180)
        preview.setPlainText(self._preview_text())

        apply_button = QPushButton("Appliquer")
        apply_button.clicked.connect(lambda: self.apply_requested.emit(self.block))

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(preview)
        layout.addWidget(apply_button)

    def _preview_text(self) -> str:
        if self.block_type == "create":
            return self.block.get("content", "")
        return (
            "SEARCH\n"
            f"{self.block.get('search_content', '')}\n\n"
            "REPLACE\n"
            f"{self.block.get('replace_content', '')}"
        )
