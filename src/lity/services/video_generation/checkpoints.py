"""Discover and resolve the video models the user has downloaded.

Lity's auto-downloader drops each model in its own folder under
``…/Documents/Lity/Models/Videos/<name>/``. Unlike the single-file image
checkpoints, video models are multi-file repositories (a diffusers repo with
``model_index.json`` + ``transformer/``, ``vae/``, ``text_encoder/`` subfolders,
or an MLX weight folder). So these helpers resolve a model *name* to the model
*directory* to load — not a single weight file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Weight extensions that mark a folder as holding a real model (used as a
# fallback when ``model_index.json`` is absent, e.g. an MLX export).
_WEIGHT_EXT = (".safetensors", ".gguf", ".pth", ".bin")
MODEL_MANIFEST = ".lity-video-model.json"
_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})(?P<suffix>\.[^.]+)$")
_TOKENIZER_MARKERS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
    "vocab.json",
    "merges.txt",
)


def _is_model_dir(folder: Path) -> bool:
    """True when ``folder`` looks like a complete downloaded video model.

    A diffusers repo is identified by its ``model_index.json``; any other layout
    (e.g. MLX) qualifies as soon as it carries at least one weight file anywhere
    inside it.
    """
    if _has_part_file(folder):
        return False
    manifest = folder / MODEL_MANIFEST
    if manifest.is_file():
        return _manifest_is_complete(folder, manifest)
    if (folder / "model_index.json").is_file():
        return _diffusers_repo_is_complete(folder)
    return any(f.is_file() and f.suffix.lower() in _WEIGHT_EXT for f in folder.rglob("*"))


def installed_video_models(video_models_dir: Path) -> list[tuple[str, Path]]:
    """``[(model_name, model_dir)]`` for every downloaded video model."""
    if not video_models_dir.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for sub in sorted(video_models_dir.iterdir()):
        if sub.is_dir() and _is_model_dir(sub):
            found.append((sub.name, sub))
    return found


def partial_video_models(video_models_dir: Path) -> list[str]:
    """Names of model folders that contain files but are not complete enough."""
    if not video_models_dir.is_dir():
        return []
    found: list[str] = []
    for sub in sorted(video_models_dir.iterdir()):
        if sub.is_dir() and not _is_model_dir(sub) and _looks_partial(sub):
            found.append(sub.name)
    return found


def resolve_checkpoint(video_models_dir: Path, selected_name: str = "") -> Path | None:
    """Model *directory* to generate with.

    Honours the user's selected model name; otherwise falls back to the first
    installed one so a single download just works without any extra choice.
    """
    models = installed_video_models(video_models_dir)
    if not models:
        return None
    if selected_name:
        for name, path in models:
            if name == selected_name:
                return path
    return models[0][1]


def _has_part_file(folder: Path) -> bool:
    return any(f.is_file() and f.name.endswith(".part") for f in folder.rglob("*"))


def _looks_partial(folder: Path) -> bool:
    if (folder / MODEL_MANIFEST).is_file() or (folder / "model_index.json").is_file():
        return True
    return any(f.is_file() and f.suffix.lower() in _WEIGHT_EXT for f in folder.rglob("*"))


def _manifest_is_complete(folder: Path, manifest: Path) -> bool:
    try:
        data = json.loads(manifest.read_text())
    except Exception:
        return False
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list) or not files:
        return False
    return _all_files_exist(folder, [str(name) for name in files])


def _all_files_exist(folder: Path, filenames: list[str]) -> bool:
    for filename in filenames:
        path = _safe_child(folder, filename)
        if path is None or not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _safe_child(folder: Path, filename: str) -> Path | None:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        return None
    return folder / path


def _diffusers_repo_is_complete(folder: Path) -> bool:
    try:
        index = json.loads((folder / "model_index.json").read_text())
    except Exception:
        return False
    if not isinstance(index, dict):
        return False
    components = {
        name: spec
        for name, spec in index.items()
        if not name.startswith("_") and isinstance(spec, list) and len(spec) >= 2
    }
    if not components:
        return False

    for name, spec in components.items():
        component_dir = folder / name
        if not component_dir.is_dir():
            return False
        class_name = str(spec[-1]).lower()
        if "scheduler" in class_name:
            if not any(component_dir.glob("*config.json")):
                return False
            continue
        if "tokenizer" in class_name:
            if not any((component_dir / marker).is_file() for marker in _TOKENIZER_MARKERS):
                return False
            continue
        if not (component_dir / "config.json").is_file():
            return False
        if not _component_weights_complete(component_dir):
            return False
    return True


def _component_weights_complete(component_dir: Path) -> bool:
    index_files = list(component_dir.glob("*.safetensors.index.json")) + list(
        component_dir.glob("*.bin.index.json")
    )
    if index_files:
        for index_file in index_files:
            try:
                data = json.loads(index_file.read_text())
            except Exception:
                return False
            weight_map = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            files = sorted({str(name) for name in weight_map.values() if name})
            if not files or not _all_files_exist(component_dir, files):
                return False
        return True

    weights = [
        path
        for path in component_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _WEIGHT_EXT and path.stat().st_size > 0
    ]
    if not weights:
        return False

    shard_groups: dict[tuple[str, str, int], set[int]] = {}
    for path in weights:
        match = _SHARD_RE.match(path.name)
        if not match:
            continue
        key = (
            str(match.group("prefix")),
            str(match.group("suffix")),
            int(match.group("count")),
        )
        shard_groups.setdefault(key, set()).add(int(match.group("index")))

    if shard_groups:
        for (_prefix, _suffix, count), indexes in shard_groups.items():
            if indexes != set(range(1, count + 1)):
                return False
    return True
