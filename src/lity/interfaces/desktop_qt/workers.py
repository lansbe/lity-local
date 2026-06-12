from __future__ import annotations

import inspect
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from lity.app.results import result_to_dict


class ChatWorker(QObject):
    chunk_ready = Signal(str)
    result_ready = Signal(dict)
    error_ready = Signal(str)
    finished = Signal()

    def __init__(self, controller: Any, message: str):
        super().__init__()
        self.controller = controller
        self.message = message
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    @Slot()
    def run(self) -> None:
        if self._cancelled:
            self.finished.emit()
            return
        try:
            result = self._process_message()
            if not self._cancelled:
                self.result_ready.emit(result_to_dict(result))
        except Exception as exc:
            self.error_ready.emit(str(exc))
        finally:
            self.finished.emit()

    def _process_message(self) -> Any:
        if hasattr(self.controller, "process_user_message_stream"):
            return self._call_streaming_handler(self.controller.process_user_message_stream)
        return self._call_sync_handler(self.controller.process_user_message_sync)

    def _call_sync_handler(self, handler: Any) -> Any:
        try:
            signature = inspect.signature(handler)
            accepts_callback = "should_cancel" in signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_callback = False

        if accepts_callback:
            return handler(self.message, should_cancel=self.is_cancelled)
        return handler(self.message)

    def _call_streaming_handler(self, handler: Any) -> Any:
        try:
            signature = inspect.signature(handler)
            accepts_callback = "should_cancel" in signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_callback = False

        if accepts_callback:
            return handler(self.message, self._emit_chunk, should_cancel=self.is_cancelled)
        return handler(self.message, self._emit_chunk)

    def _emit_chunk(self, chunk: str) -> None:
        if chunk and not self._cancelled:
            self.chunk_ready.emit(chunk)


class ModelListWorker(QObject):
    models_ready = Signal(list, str)
    error_ready = Signal(str)
    finished = Signal()

    def __init__(self, engine: Any):
        super().__init__()
        self.engine = engine

    @Slot()
    def run(self) -> None:
        try:
            models = self.engine.get_installed_models()
            self.models_ready.emit(models, getattr(self.engine, "model", ""))
        except Exception as exc:
            self.error_ready.emit(str(exc))
        finally:
            self.finished.emit()
