from __future__ import annotations

from collections.abc import Callable
from typing import Any

# loop_factory(**overrides) -> AgentLoop-like object. Injected so the
# orchestration logic is unit-testable without a model or a real loop.
LoopFactory = Callable[..., Any]

# Each sub-task gets a tight step budget: it has ONE plan step to accomplish.
_SUBTASK_MAX_STEPS = 6

# How much of each sub-task's outcome is carried into the next context.
_CARRY_CHARS = 700


class TaskOrchestrator:
    """Executes a multi-step plan as a sequence of FRESH agent loops.

    One long tool loop overflows a small local model's context and attention —
    by the eighth step it has forgotten the goal and re-reads the same files.
    Instead, each plan step runs in its own loop with a short, purpose-built
    context: the global goal, a digest of what previous steps accomplished, and
    the current step only. This is the local-model equivalent of sub-agents:
    the orchestration lives in code, the model only ever sees one small task.
    """

    def __init__(
        self,
        loop_factory: LoopFactory,
        *,
        max_steps_per_subtask: int = _SUBTASK_MAX_STEPS,
    ):
        self.loop_factory = loop_factory
        self.max_steps_per_subtask = max_steps_per_subtask

    def run(
        self,
        *,
        goal: str,
        plan: list[str],
        base_messages: list[dict[str, Any]],
        on_event: Callable[[str, dict], None],
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Run every plan step in its own loop; returns (answer, receipts)."""
        system = next((m for m in base_messages if m.get("role") == "system"), None)
        completed: list[str] = []
        receipts_items: list[dict[str, Any]] = []
        final = ""

        for index, step in enumerate(plan, 1):
            if should_cancel and should_cancel():
                break
            on_event("subtask", {"index": index, "total": len(plan), "step": step})

            messages: list[dict[str, Any]] = []
            if system is not None:
                messages.append(dict(system))
            messages.append(
                {"role": "user", "content": self._subtask_prompt(goal, plan, completed, index)}
            )

            loop = self.loop_factory(max_steps=self.max_steps_per_subtask)
            if index < len(plan) and getattr(loop, "verify_command", None):
                # Intermediate states are legitimately red (half-done refactors);
                # the definition-of-done check only gates the FINAL step.
                loop.verify_command = None

            result = loop.run(messages, on_event=on_event, should_cancel=should_cancel)
            summary = loop.receipts_summary() if hasattr(loop, "receipts_summary") else None
            if summary:
                receipts_items.extend(summary.get("items", []))

            digest = (result or "").strip()
            completed.append(f"Étape {index} — {step} : {digest[:_CARRY_CHARS]}")
            if digest:
                final = digest

        receipts: dict[str, Any] | None = None
        if receipts_items:
            receipts = {
                "items": receipts_items,
                "grounded": any(item.get("ok") for item in receipts_items),
                "tools_used": sorted({str(item.get("name")) for item in receipts_items}),
            }
        if not final:
            return "Je n'ai pas pu finaliser la réponse.", receipts
        if len(completed) > 1:
            footer = "\n\n---\nPlan exécuté :\n" + "\n".join(
                f"{i}. {step}" for i, step in enumerate(plan[: len(completed)], 1)
            )
            final += footer
        return final, receipts

    @staticmethod
    def _subtask_prompt(goal: str, plan: list[str], completed: list[str], index: int) -> str:
        parts = [f"OBJECTIF GLOBAL : {goal}", "", "PLAN COMPLET :"]
        for i, step in enumerate(plan, 1):
            marker = " (déjà fait)" if i <= len(completed) else ""
            parts.append(f"{i}. {step}{marker}")
        if completed:
            parts.append("")
            parts.append("RÉSUMÉ DE CE QUI A DÉJÀ ÉTÉ FAIT :")
            parts.extend(completed)
        parts.append("")
        parts.append(
            f"TA TÂCHE MAINTENANT — réalise UNIQUEMENT l'étape {index} : {plan[index - 1]}"
        )
        parts.append(
            "Utilise les outils nécessaires, puis termine par un court résumé de ce "
            "que tu as réellement fait (fichiers modifiés, commandes lancées, résultats)."
        )
        return "\n".join(parts)
