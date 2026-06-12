from __future__ import annotations

import random
import threading
from typing import Any

from lity.infrastructure.paths import AppPaths
from lity.infrastructure.settings import SettingsStore
from lity.services.image_generation.checkpoints import (
    InstalledImageModel,
    installed_image_records,
)
from lity.services.image_generation.engine_install import (
    install_engine,
    install_mlx_engine,
)
from lity.services.image_generation.local_engine import (
    LocalImageEngine,
    dependencies_available,
)
from lity.services.image_generation.mlx_engine import (
    MlxImageEngine,
    mlx_dependencies_available,
    mlx_supported_platform,
)
from lity.services.image_generation.prompt_builder import ImagePromptBuilder
from lity.services.image_generation.settings import ImageSettingsManager
from lity.services.image_generation.update_interpreter import ImageParamUpdateInterpreter

# Samplers the in-process diffusers engine knows how to reproduce. Offered to the
# prompt builder so it never proposes one the local engine can't honour.
LOCAL_SAMPLERS: tuple[str, ...] = (
    "Euler a",
    "Euler",
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DDIM",
    "Heun",
    "UniPC",
)

_NO_MODEL_MESSAGE = (
    "Aucun modèle image n'est encore téléchargé. Ouvre « Modèles » → onglet "
    "« Images », télécharge par exemple « Stable Diffusion 1.5 », puis réactive "
    "le mode image."
)


class ImageGenerationManager:
    """Drives local, server-less image generation from a downloaded checkpoint.

    No external Stable Diffusion WebUI: the heavy ``torch``/``diffusers`` engine
    is installed on first use, the user's downloaded ``.safetensors`` is loaded
    in-process, and images are rendered straight on the local GPU.
    """

    def __init__(self, ai_engine: Any, paths: AppPaths, settings_store: SettingsStore):
        self.paths = paths
        self.settings = ImageSettingsManager(settings_store)
        self.engine = LocalImageEngine(paths)
        # Second renderer for MLX (mflux) models — diffusers can't load them.
        self.mlx_engine = MlxImageEngine(paths)
        self.prompt_builder = ImagePromptBuilder(ai_engine)
        self.interpreter = ImageParamUpdateInterpreter(ai_engine)
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
        # Install the engine the *active* model needs (diffusers vs mlx).
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
                "type": "image_dependency",
                "status": "installing",
                "progress": pct,
                "message": f"Installation du moteur image… {pct}%",
            }
        if install is not None and install.get("ok") is False:
            message = str(install.get("message") or "Échec de l'installation du moteur image.")
            self._install = None
            return {"type": "image_dependency", "status": "error", "message": message}

        # Install finished (or was never needed): verify the engine then the model.
        backend = str(install.get("backend") if install else "") or self._required_backend()
        if not self._backend_ready(backend):
            return {
                "type": "image_dependency",
                "status": "error",
                "message": "Le moteur image reste indisponible après l'installation.",
            }
        if not self._has_model():
            self._install = None
            return self._no_model_result()
        self._install = None
        return self._activate_ready_session(
            "Moteur image prêt. Mode image activé. Décris l'image à générer."
        )

    def shutdown(self) -> dict[str, Any]:
        self.cancel_session()
        self.engine.unload()
        self.mlx_engine.unload()
        return {
            "type": "image_dependency",
            "status": "stopped",
            "message": "Mode image arrêté.",
        }

    def cancel_session(self) -> None:
        self.state["active"] = False
        self.state["step"] = "idle"
        self.state["current_params"] = None

    def select_image_model(self, name: str) -> dict[str, Any]:
        """Persist which downloaded model the engine should generate with."""
        self.settings.set_selected_image_model(name)
        return {"ok": True, "selected": self.settings.selected_image_model}

    # ----------------------------------------------------------- engine setup
    def _has_model(self) -> bool:
        return bool(installed_image_records(self.paths.image_models_dir))

    def _active_record(self) -> InstalledImageModel | None:
        """The downloaded model the engine will render with (selected, else first)."""
        records = installed_image_records(self.paths.image_models_dir)
        if not records:
            return None
        selected = self.settings.selected_image_model
        for record in records:
            if record.name == selected:
                return record
        return records[0]

    def _active_model_name(self) -> str:
        record = self._active_record()
        return record.name if record else ""

    def _required_backend(self) -> str:
        """Backend the active model needs; default diffusers when none installed."""
        record = self._active_record()
        return record.backend if record is not None else "automatic1111"

    def _backend_ready(self, backend: str) -> bool:
        if backend == "mlx":
            record = self._active_record()
            mlx = record.meta.get("mlx") if record and isinstance(record.meta, dict) else {}
            command = str(mlx.get("command") or "") if isinstance(mlx, dict) else ""
            return mlx_dependencies_available(command or "mflux-generate")
        return dependencies_available()

    @staticmethod
    def _catalog_entry_for(model_name: str) -> dict[str, Any]:
        from lity.core.image_model_advisor import IMAGE_MODEL_CATALOG

        for entry in IMAGE_MODEL_CATALOG:
            if str(entry.get("name")) == model_name:
                return entry
        return {}

    @classmethod
    def _gen_defaults_for(cls, model_name: str) -> dict[str, Any]:
        """Generation defaults a distilled checkpoint was trained for.

        Few-step models (turbo/lightning) need their own steps/cfg/sampler —
        the classic steps=25 / cfg=7.5 the prompt builder proposes produces
        burnt, oversaturated renders on them."""
        gen = cls._catalog_entry_for(model_name).get("gen")
        return dict(gen) if isinstance(gen, dict) else {}

    def _no_model_result(self) -> dict[str, Any]:
        return {"type": "image_dependency", "status": "no_model", "message": _NO_MODEL_MESSAGE}

    def _begin_engine_install(self, backend: str = "automatic1111") -> dict[str, Any]:
        if backend == "mlx" and not mlx_supported_platform():
            return {
                "type": "image_dependency",
                "status": "error",
                "message": "Le moteur MLX nécessite un Mac Apple Silicon (arm64).",
            }
        if self._install is not None and self._install.get("running"):
            pct = int(self._install.get("pct", 0))
            return {
                "type": "image_dependency",
                "status": "installing",
                "progress": pct,
                "message": f"Installation du moteur image déjà en cours… {pct}%",
            }

        self._install = {
            "running": True,
            "pct": 2,
            "message": "Préparation…",
            "ok": None,
            "backend": backend,
        }
        installer = install_mlx_engine if backend == "mlx" else install_engine
        engine_label = "MLX (mflux)" if backend == "mlx" else "local"
        size_hint = "~0,5 Go" if backend == "mlx" else "~2,5 Go"

        def task() -> None:
            def on_progress(pct: int, line: str) -> None:
                if self._install is not None:
                    self._install["pct"] = pct
                    self._install["message"] = line

            try:
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
            "type": "image_dependency",
            "status": "installing",
            "progress": 2,
            "message": (
                f"Premier lancement du mode image : installation du moteur {engine_label} "
                f"({size_hint}, une seule fois). Lity l'active automatiquement à la fin."
            ),
        }

    def _activate_ready_session(self, message: str | None = None) -> dict[str, Any]:
        self.state["active"] = True
        self.state["step"] = "waiting_for_prompt"
        self.state["current_params"] = None
        return {
            "type": "image_mode_ready",
            "status": "ready",
            "message": message or "Mode image activé. Décris l'image à générer.",
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
            # Distilled checkpoints override the LLM's classic defaults.
            proposal.update(self._gen_defaults_for(active_model))
            self.state["current_params"] = proposal
            # Generate straight away — no "ok" confirmation round-trip. The user
            # asked for an image, so the app renders it directly with
            # auto-tuned parameters. A failed render leaves the session on
            # ``waiting_for_prompt`` so the next description simply retries.
            result = self.execute_generation(proposal)
            if result.get("type") == "image_generation_result":
                self.cancel_session()
            return result

        if step == "waiting_for_confirmation":
            interpretation = self.interpreter.interpret_correction(
                self.state["current_params"],
                user_input,
            )
            action = interpretation.get("action", "normal_message")
            if action == "update_image_params":
                current = self.state["current_params"]
                for key, value in interpretation.get("updates", {}).items():
                    if key in current:
                        current[key] = value
                return {
                    "type": "image_parameters_proposal",
                    "content": current,
                    "message": interpretation.get(
                        "user_message", "Modifications prises en compte."
                    ),
                }
            if action == "confirm_generation":
                final_params = self.state["current_params"]
                if not isinstance(final_params, dict):
                    return {"type": "error", "message": "Aucun paramètre image à générer."}
                result = self.execute_generation(final_params)
                if result.get("type") == "image_generation_result":
                    self.cancel_session()
                return result
            if action == "cancel_generation":
                self.cancel_session()
                return {"type": "image_cancelled", "message": "Session image annulée."}
            return {
                "type": "image_normal_chat",
                "message": "Mode image actif. Ajuste les paramètres, confirme avec 'ok', ou annule.",
            }

        return {"type": "error", "message": "Étape de génération d'image non valide."}

    # ------------------------------------------------------------ generation
    def execute_generation(self, params: dict[str, Any]) -> dict[str, Any]:
        record = self._active_record()
        if record is None:
            return {"type": "error", "message": _NO_MODEL_MESSAGE}

        # The active model changed since the proposal was built (e.g. switched
        # in the Models modal): its own defaults beat the stale proposal ones.
        active_model = record.name
        if params.get("checkpoint") != active_model:
            params = {**params, **self._gen_defaults_for(active_model), "checkpoint": active_model}

        seed = _resolve_seed(params.get("seed", -1))
        try:
            if record.backend == "mlx":
                mlx_meta = record.meta.get("mlx") if isinstance(record.meta, dict) else {}
                image_path = self.mlx_engine.generate(record.path, params, seed, mlx=mlx_meta or {})
            else:
                # SD2.x-arch single files need an explicit diffusers config repo.
                config_hint = str(self._catalog_entry_for(active_model).get("hf_config", ""))
                image_path = self.engine.generate(
                    record.path, params, seed, config_hint=config_hint
                )
        except Exception as exc:
            return {
                "type": "error",
                "message": f"La génération d'image a échoué : {exc}",
            }

        return {
            "type": "image_generation_result",
            "content": {
                "status": "success",
                "params": {**params, "seed": seed, "checkpoint": active_model},
                "image_path": str(image_path),
            },
            "message": f"Image générée en local avec {active_model}.",
        }


def _resolve_seed(value: Any) -> int:
    try:
        seed = int(value)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        return random.randint(1, 2_147_483_647)
    return seed
