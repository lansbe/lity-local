from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class Sidebar(QWidget):
    model_selected = Signal(str)
    refresh_models_requested = Signal()
    workdir_selected = Signal(str)
    image_mode_toggled = Signal()
    clear_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Sidebar")

        title = QLabel("Lity")
        title.setObjectName("AppTitle")

        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        refresh_button = QPushButton("Rafraîchir modèles")
        refresh_button.clicked.connect(self.refresh_models_requested.emit)

        self.workdir_input = QLineEdit()
        self.workdir_input.setPlaceholderText("Répertoire de travail")
        browse_button = QPushButton("Choisir dossier")
        browse_button.clicked.connect(self._choose_folder)

        image_button = QPushButton("Mode image")
        image_button.clicked.connect(self.image_mode_toggled.emit)

        clear_button = QPushButton("Effacer historique")
        clear_button.clicked.connect(self.clear_requested.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addSpacing(16)
        layout.addWidget(QLabel("Modèle Ollama"))
        layout.addWidget(self.model_combo)
        layout.addWidget(refresh_button)
        layout.addSpacing(16)
        layout.addWidget(QLabel("Espace de travail"))
        layout.addWidget(self.workdir_input)
        layout.addWidget(browse_button)
        layout.addSpacing(16)
        layout.addWidget(image_button)
        layout.addWidget(clear_button)
        layout.addStretch(1)

    def set_models(self, models: list[str], selected: str) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
        index = self.model_combo.findText(selected)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

    def set_workdir(self, path: str) -> None:
        self.workdir_input.setText(path)

    def _on_model_changed(self, model_name: str) -> None:
        if model_name:
            self.model_selected.emit(model_name)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choisir un répertoire")
        if path:
            self.workdir_input.setText(path)
            self.workdir_selected.emit(path)
