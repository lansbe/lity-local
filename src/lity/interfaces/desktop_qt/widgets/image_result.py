from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ImageResultWidget(QWidget):
    def __init__(self, result: dict):
        super().__init__()
        title = QLabel("Image générée")
        title.setObjectName("CardTitle")
        layout = QVBoxLayout(self)
        layout.addWidget(title)

        image_path = result.get("image_path")
        if image_path and Path(image_path).exists():
            pixmap = QPixmap(image_path)
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaledToWidth(420))
            layout.addWidget(image_label)
        else:
            layout.addWidget(QLabel("Aucune image physique à afficher."))

        params = result.get("params", {})
        if params:
            layout.addWidget(QLabel(f"Prompt: {params.get('prompt', '')}"))
