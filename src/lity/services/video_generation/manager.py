from __future__ import annotations

import logging
import random
import threading
from typing import Any

from lity.infrastructure.paths import AppPaths
from lity.infrastructure.settings import SettingsStore
from lity.services.video_generation.checkpoints import (
    installed_video_models,
    partial_video_models,
    resolve_checkpoint,
)
from lity.services.video_generation.engine_install import (
    install_mlx_video_engine,
    install_video_engine,
)
from lity.services.video_generation.local_engine import (
    LocalVideoEngine,
    dependencies_available,
)
from lity.services.video_generation.mlx_engine import (
    MlxVideoEngine,
    mlx_video_dependencies_available,
    mlx_video_supported_platform,
)
from lity.services.video_generation.prompt_builder import VideoPromptBuilder
from lity.services.video_generation.settings import VideoSettingsManager
from lity.services.video_generation.update_interpreter import VideoParamUpdateInterpreter

# Samplers the in-process diffusers engine knows how to reproduce.
LOCAL_SAMPLERS: tuple[str, ...] = ("UniPC", "Euler", "DDIM")

_NO_MODEL_MESSAGE = (
    "Aucun modèle vidéo complet n'est encore téléchargé. Ouvre « Modèles » → onglet "
    "« Vidéos », télécharge par exemple « Wan 2.1 T2V 1.3B », puis réactive le "
    "mode vidéo."
)

logger = logging.getLogger(__name__)


class VideoGenerationManager:
    """Drives local, server-less video generation from a downloaded model.

    Mirrors :class:`ImageGenerationManager`: diffusers models load in-process;
    MLX/LTX models render through an isolated ``ltx-2-mlx`` subprocess installed
    on first use.
    """

    def __init__(self, ai_engine: Any, paths: AppPaths, settings_store: SettingsStore):
        self.paths = paths
        self.settings = VideoSettingsManager(settings_store)
        self.engine = LocalVideoEngine(paths)
        self.mlx_engine = MlxVideoEngine(paths)
        self.prompt_builder = VideoPromptBuilder(ai_engine)
        self.interpreter = VideoParamUpdateInterpreter(ai_engine)
        self.state: dict[str, Any] = {
            "active": False,
            "step": "idle",
            "current_params": None,
        }
        # First-run engine install progress; None when no install is in flight.
        self._install: dict[str, Any] | None = None
        self._install_thread: threading.Thread | None = None

    # --------------------------------------------------------------- session
    def is_active(self) -> bool:
        return bool(self.state["active"])

    def start_session(self) -> dict[str, Any]:
        backend = self._required_backend()
        if not self._backend_ready(backend):
            return self._begin_engine_install(backend)
        if not self._has_model():
            return self._no_model_result()
        return self._activate_ready_session()

    def poll_launch_status(self) -> dict[str, Any]:
        install = self._install
        if install is not None and install.get("running"):
            pct = int(install.get("pct", 0))
            return {
                "type": "video_dependency",
                "status": "installing",
                "progress": pct,
                "message": f"Installation du moteur vidéo… {pct}%",
            }
        if install is not None and install.get("ok") is False:
            message = str(install.get("message") or "Échec de l'installation du moteur vidéo.")
            self._install = None
            return {"type": "video_dependency", "status": "error", "message": message}

        # Install finished (or was never needed): verify the engine then the model.
        backend = str(install.get("backend") if install else "") or self._required_backend()
        if not self._backend_ready(backend):
            return {
                "type": "video_dependency",
                "status": "error",
                "message": "Le moteur vidéo reste indisponible après l'installation.",
            }
        if not self._has_model():
            self._install = None
            return self._no_model_result()
        self._install = None
        return self._activate_ready_session(
            "Moteur vidéo prêt. Mode vidéo activé. Décris la vidéo à générer."
        )

    def shutdown(self) -> dict[str, Any]:
        self.cancel_session()
        self.engine.unload()
        self.mlx_engine.unload()
        return {
            "type": "video_dependency",
            "status": "stopped",
            "message": "Mode vidéo arrêté.",
        }

    def cancel_session(self) -> None:
        self.state["active"] = False
        self.state["step"] = "idle"
        self.state["current_params"] = None

    def select_video_model(self, name: str) -> dict[str, Any]:
        """Persist which downloaded model the engine should generate with."""
        self.settings.set_selected_video_model(name)
        return {"ok": True, "selected": self.settings.selected_video_model}

    # ----------------------------------------------------------- engine setup
    def _has_model(self) -> bool:
        return bool(installed_video_models(self.paths.video_models_dir))

    def _active_model_name(self) -> str:
        selected = self.settings.selected_video_model
        models = installed_video_models(self.paths.video_models_dir)
        names = [name for name, _ in models]
        if selected in names:
            return selected
        return names[0] if names else ""

    def _active_backend(self) -> str:
        return str(self._catalog_entry_for(self._active_model_name()).get("backend", "diffusers"))

    def _required_backend(self) -> str:
        return self._active_backend() if self._has_model() else "diffusers"

    def _backend_ready(self, backend: str) -> bool:
        if backend == "mlx":
            entry = self._catalog_entry_for(self._active_model_name())
            mlx = entry.get("mlx") if isinstance(entry, dict) else {}
            command = str(mlx.get("command") or "") if isinstance(mlx, dict) else ""
            return mlx_video_dependencies_available(self.paths, command or "ltx-2-mlx")
        return dependencies_available()

    @staticmethod
    def _catalog_entry_for(model_name: str) -> dict[str, Any]:
        from lity.core.video_model_advisor import VIDEO_MODEL_CATALOG

        for entry in VIDEO_MODEL_CATALOG:
            if str(entry.get("name")) == model_name:
                return entry
        return {}

    @classmethod
    def _gen_defaults_for(cls, model_name: str) -> dict[str, Any]:
        """Generation defaults a model was tuned for (resolution/frames/steps)."""
        gen = cls._catalog_entry_for(model_name).get("gen")
        return dict(gen) if isinstance(gen, dict) else {}

    def _no_model_result(self) -> dict[str, Any]:
        partials = partial_video_models(self.paths.video_models_dir)
        if partials:
            names = ", ".join(partials)
            message = (
                f"Téléchargement vidéo incomplet ({names}). Ouvre « Modèles » → onglet "
                "« Vidéos » et clique « Télécharger » à nouveau : Lity reprend les fichiers "
                "manquants au lieu de repartir de zéro."
            )
            return {"type": "video_dependency", "status": "no_model", "message": message}
        return {"type": "video_dependency", "status": "no_model", "message": _NO_MODEL_MESSAGE}

    def _begin_engine_install(self, backend: str = "diffusers") -> dict[str, Any]:
        if backend == "mlx" and not mlx_video_supported_platform():
            return {
                "type": "video_dependency",
                "status": "error",
                "message": "Le runtime vidéo MLX nécessite un Mac Apple Silicon (arm64) avec Python 3.11+.",
            }
        if self._install is not None and self._install.get("running"):
            pct = int(self._install.get("pct", 0))
            return {
                "type": "video_dependency",
                "status": "installing",
                "progress": pct,
                "message": f"Installation du moteur vidéo déjà en cours… {pct}%",
            }

        self._install = {
            "running": True,
            "pct": 2,
            "message": "Préparation…",
            "ok": None,
            "backend": backend,
        }
        installer = install_mlx_video_engine if backend == "mlx" else install_video_engine
        engine_label = "MLX (ltx-2-mlx)" if backend == "mlx" else "local"
        size_hint = "~1 Go runtime + dépendances" if backend == "mlx" else "~2,5 Go"

        def task() -> None:
            def on_progress(pct: int, line: str) -> None:
                if self._install is not None:
                    self._install["pct"] = pct
                    self._install["message"] = line

            try:
                if backend == "mlx":
                    result = installer(self.paths, on_progress=on_progress)
                else:
                    result = installer(on_progress=on_progress)
            except Exception as exc:  # pragma: no cover - defensive
                result = {"ok": False, "message": str(exc)}
            if self._install is not None:
                self._install["running"] = False
                self._install["ok"] = bool(result.get("ok"))
                self._install["message"] = str(result.get("message", ""))

        self._install_thread = threading.Thread(target=task, daemon=True)
        self._install_thread.start()
        return {
            "type": "video_dependency",
            "status": "installing",
            "progress": 2,
            "message": (
                f"Premier lancement du mode vidéo : installation du moteur {engine_label} "
                f"({size_hint}, une seule fois). Lity l'active automatiquement à la fin."
            ),
        }

    def _activate_ready_session(self, message: str | None = None) -> dict[str, Any]:
        self.state["active"] = True
        self.state["step"] = "waiting_for_prompt"
        self.state["current_params"] = None
        return {
            "type": "video_mode_ready",
            "status": "ready",
            "message": message or "Mode vidéo activé. Décris la vidéo à générer.",
        }

    # --------------------------------------------------------------- dialogue
    def process_user_message(self, user_input: str, engine: Any) -> dict[str, Any]:
        self.prompt_builder.engine = engine
        self.interpreter.engine = engine
        step = self.state["step"]

        if step == "waiting_for_prompt":
            proposal = self.prompt_builder.build_initial_proposal(
                user_input,
                available_samplers=list(LOCAL_SAMPLERS),
            )
            active_model = self._active_model_name()
            proposal["checkpoint"] = active_model
            # The model's own profile overrides the LLM's generic defaults.
            proposal.update(self._gen_defaults_for(active_model))
            self.state["current_params"] = proposal
            # Generate straight away — no "ok" confirmation round-trip. A failed
            # render leaves the session on ``waiting_for_prompt`` so the next
            # description simply retries.
            result = self.execute_generation(proposal)
            if result.get("type") == "video_generation_result":
                self.cancel_session()
            return result

        if step == "waiting_for_confirmation":
            interpretation = self.interpreter.interpret_correction(
                self.state["current_params"],
                user_input,
            )
            action = interpretation.get("action", "normal_message")
            if action == "update_video_params":
                current = self.state["current_params"]
                for key, value in interpretation.get("updates", {}).items():
                    if key in current:
                        current[key] = value
                return {
                    "type": "video_parameters_proposal",
                    "content": current,
                    "message": interpretation.get(
                        "user_message", "Modifications prises en compte."
                    ),
                }
            if action == "confirm_generation":
                final_params = self.state["current_params"]
                if not isinstance(final_params, dict):
                    return {"type": "error", "message": "Aucun paramètre vidéo à générer."}
                result = self.execute_generation(final_params)
                if result.get("type") == "video_generation_result":
                    self.cancel_session()
                return result
            if action == "cancel_generation":
                self.cancel_session()
                return {"type": "video_cancelled", "message": "Session vidéo annulée."}
            return {
                "type": "video_normal_chat",
                "message": "Mode vidéo actif. Ajuste les paramètres, confirme avec 'ok', ou annule.",
            }

        return {"type": "error", "message": "Étape de génération vidéo non valide."}

    # ------------------------------------------------------------ generation
    def execute_generation(self, params: dict[str, Any]) -> dict[str, Any]:
        checkpoint = resolve_checkpoint(
            self.paths.video_models_dir, self.settings.selected_video_model
        )
        if checkpoint is None:
            return {"type": "error", "message": _NO_MODEL_MESSAGE}

        active_model = self._active_model_name()
        backend = self._active_backend()

        # The active model changed since the proposal was built: its own defaults
        # beat the stale proposal ones.
        if params.get("checkpoint") != active_model:
            params = {**params, **self._gen_defaults_for(active_model), "checkpoint": active_model}

        seed = _resolve_seed(params.get("seed", -1))
        config_hint = str(self._catalog_entry_for(active_model).get("hf_config", ""))
        try:
            if backend == "mlx":
                entry = self._catalog_entry_for(active_model)
                mlx = entry.get("mlx") if isinstance(entry, dict) else {}
                video_path = self.mlx_engine.generate(checkpoint, params, seed, mlx=mlx or {})
            else:
                video_path = self.engine.generate(
                    checkpoint, params, seed, backend=backend, config_hint=config_hint
                )
        except Exception as exc:
            logger.exception("Local video generation failed")
            return {
                "type": "error",
                "message": f"La génération vidéo a échoué : {exc}",
            }

        return {
            "type": "video_generation_result",
            "content": {
                "status": "success",
                "params": {**params, "seed": seed, "checkpoint": checkpoint.name},
                "video_path": str(video_path),
            },
            "message": f"Vidéo générée en local avec {checkpoint.name}.",
        }


def _resolve_seed(value: Any) -> int:
    try:
        seed = int(value)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        return random.randint(1, 2_147_483_647)
    return seed
