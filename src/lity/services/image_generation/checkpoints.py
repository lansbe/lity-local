"""Discover and resolve the image checkpoints the user has downloaded.

Lity's auto-downloader drops each model in its own folder under
``…/Documents/Lity/Models/Images/<name>/`` and writes a small
``.lity_model.json`` marker recording which *backend* renders it. Two backends
coexist:

* **diffusers** (``automatic1111`` / no marker) — a single ``.safetensors``/
  ``.ckpt`` loaded in-process by :mod:`local_engine`.
* **mlx** — an mflux model *directory* rendered out-of-process by
  :mod:`mlx_engine`. Its weights are also ``.safetensors``, so without the
  marker the diffusers engine would try to load them and crash; the marker is
  what keeps the two apart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Single-file checkpoints the in-process diffusers engine can load. ``.gguf`` is
# intentionally excluded: those target stable-diffusion.cpp, not diffusers.
_LOADABLE_EXT = (".safetensors", ".ckpt")

# Marker file written next to a downloaded model recording its catalog backend.
MODEL_MARKER = ".lity_model.json"

# Backends rendered by the in-process diffusers engine (single-file checkpoint).
# Anything not listed here (currently only ``mlx``) routes to another engine.
_DIFFUSERS_BACKENDS = {"", "automatic1111", "diffusers"}


@dataclass(frozen=True)
class InstalledImageModel:
    """A downloaded image model, tagged with the engine that can render it."""

    name: str
    backend: str
    # For diffusers: the checkpoint file. For mlx: the model *directory*.
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)


def write_model_marker(folder: Path, model: dict[str, Any]) -> None:
    """Record a downloaded model's backend/metadata so discovery can route it."""
    payload: dict[str, Any] = {
        "name": str(model.get("name") or folder.name),
        "display_name": str(model.get("display_name") or model.get("name") or folder.name),
        "backend": str(model.get("backend") or "automatic1111"),
    }
    # mlx models carry the CLI invocation details the engine needs to render.
    if isinstance(model.get("mlx"), dict):
        payload["mlx"] = dict(model["mlx"])
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / MODEL_MARKER).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover - never fail a download on this
        logger.warning("Could not write model marker in %s: %s", folder, exc)


def read_model_marker(folder: Path) -> dict[str, Any]:
    """Read a folder's marker (``{}`` when absent or unreadable)."""
    marker = folder / MODEL_MARKER
    if not marker.is_file():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pick_checkpoint(folder: Path) -> Path | None:
    """The main weight file in a model folder (largest ``.safetensors``).

    Picking the largest avoids grabbing a small companion file (a standalone
    VAE, a config-only safetensors) instead of the actual checkpoint.
    """
    files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in _LOADABLE_EXT]
    if not files:
        return None
    safetensors = [f for f in files if f.suffix.lower() == ".safetensors"]
    pool = safetensors or files
    return max(pool, key=lambda f: f.stat().st_size)


def installed_image_records(image_models_dir: Path) -> list[InstalledImageModel]:
    """Every downloaded image model, backend-tagged, for routing + the UI."""
    if not image_models_dir.is_dir():
        return []
    found: list[InstalledImageModel] = []
    for sub in sorted(image_models_dir.iterdir()):
        if not sub.is_dir():
            continue
        marker = read_model_marker(sub)
        backend = str(marker.get("backend") or "").lower()
        if backend == "mlx":
            # mflux loads the whole directory (``--path``); presence of weights
            # is enough. Path is the folder itself, not a single file.
            if any(sub.rglob("*.safetensors")):
                found.append(InstalledImageModel(sub.name, "mlx", sub, marker))
            continue
        checkpoint = _pick_checkpoint(sub)
        if checkpoint is not None:
            found.append(
                InstalledImageModel(sub.name, backend or "automatic1111", checkpoint, marker)
            )
    return found


def installed_image_models(image_models_dir: Path) -> list[tuple[str, Path]]:
    """``[(model_name, checkpoint_file)]`` for diffusers-loadable models only.

    Excludes mlx models on purpose: the in-process diffusers engine can't load
    them, and silently picking one up would crash a render (or hijack the active
    model). Callers that need every backend use :func:`installed_image_records`.
    """
    return [
        (record.name, record.path)
        for record in installed_image_records(image_models_dir)
        if record.backend in _DIFFUSERS_BACKENDS
    ]


def resolve_checkpoint(image_models_dir: Path, selected_name: str = "") -> Path | None:
    """Diffusers checkpoint file to generate with.

    Honours the user's selected model name; otherwise falls back to the first
    installed diffusers checkpoint so a single download just works without any
    extra choice.
    """
    models = installed_image_models(image_models_dir)
    if not models:
        return None
    if selected_name:
        for name, path in models:
            if name == selected_name:
                return path
    return models[0][1]
