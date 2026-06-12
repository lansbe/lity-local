from __future__ import annotations

import threading


class BackgroundTaskMixin:
    """Post-turn background work for AgentController."""

    def _trigger_background_memory_task(self, user_msg: str) -> None:
        from lity.app.controller import should_extract_fact

        if not should_extract_fact(user_msg):
            return

        def task() -> None:
            try:
                fact = self.engine.extract_fact(user_msg)
                if fact:
                    self.memory.process_extracted_fact(fact)
                    self._index_fact_to_store(fact)
            except Exception:
                return

        threading.Thread(target=task, daemon=True).start()

    def _trigger_background_summary(self) -> None:
        """Post-turn background tasks: cross-session memory indexing + summary."""
        self._index_active_conversation_to_memory()
        if not hasattr(self.engine, "summarize_context") or not hasattr(
            self.memory, "pending_summary"
        ):
            return
        pending = self.memory.pending_summary()
        if not pending:
            return

        def task() -> None:
            try:
                summary = self.engine.summarize_context(pending["prior"], pending["messages"])
                if summary:
                    self.memory.set_conversation_summary(summary, pending["count"])
            except Exception:
                return

        threading.Thread(target=task, daemon=True).start()
