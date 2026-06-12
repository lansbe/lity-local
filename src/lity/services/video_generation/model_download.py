"""Real, automatic download of local video-generation models.

Video models live on Hugging Face as multi-file repositories (a diffusers repo,
or an MLX weight folder). This module resolves a catalog entry's ``model_url``
to the concrete files via the HF public API, then streams them — with progress,
preserving subfolders — into ``…/Documents/Lity/Models/Videos/<name>/``.

Mirrors :mod:`lity.services.image_generation.model_download`, minus the
single-file/Civitai branches (no video model is a single A1111 checkpoint). Only
``httpx`` (a base dependency) is used.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lity.services.video_generation.checkpoints import MODEL_MANIFEST

logger = logging.getLogger(__name__)

# (downloaded_bytes, total_bytes_or_0, filename, file_index, file_count)
ProgressCallback = Callable[[int, int, str, int, int], None]
CancelCallback = Callable[[], bool]

_HF_RE = re.compile(r"huggingface\.co/([^/\s?#]+/[^/\s?#]+)")
# Docs/media that bloat a download without being weights or config.
_SKIP_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".gitattributes", ".txt")
_MAX_ATTEMPTS = 3


def _hf_repo(url: str) -> str | None:
    match = _HF_RE.search(url or "")
    return match.group(1) if match else None


def resolve_video_download(model: dict[str, Any]) -> list[dict[str, str]]:
    """Resolve a catalog model to the concrete files to fetch.

    Returns ``[{"url", "filename"}, …]`` where ``filename`` keeps the repo's
    subfolder layout (e.g. ``vae/diffusion_pytorch_model.safetensors``). Empty if
    the host is unsupported or the API call fails — callers treat ``[]`` as
    "can't auto-download, fall back to opening the page". Raises nothing.
    """
    import httpx

    url = str(model.get("model_url", ""))
    repo = _hf_repo(url)
    if not repo:
        return []

    try:
        response = httpx.get(
            f"https://huggingface.co/api/models/{repo}",
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Lity"},
        )
        response.raise_for_status()
        siblings = response.json().get("siblings", []) or []
    except Exception as exc:
        logger.warning("Hugging Face listing failed for %s: %s", repo, exc)
        return []

    files = [str(s.get("rfilename", "")) for s in siblings if s.get("rfilename")]
    # The whole repo is the model. Skip docs/media; keep weights + config in
    # whatever subfolders they live (transformer/, vae/, text_encoder/, …).
    chosen = [f for f in files if f and not f.lower().endswith(_SKIP_SUFFIXES)]
    return [
        {
            "url": f"https://huggingface.co/{repo}/resolve/main/{name}?download=true",
            "filename": name,
        }
        for name in chosen
    ]


def _stream_to_file(
    url: str,
    dest: Path,
    on_chunk: Callable[[int, int], None],
    should_cancel: CancelCallback | None,
) -> None:
    """Stream one URL to ``dest`` (atomically via a .part file)."""
    import httpx

    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Long downloads: generous connect, no overall read deadline.
    timeout = httpx.Timeout(30.0, read=None, write=None, pool=None)
    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "Lity"},
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            with open(part, "wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 256):
                    if should_cancel and should_cancel():
                        handle.close()
                        part.unlink(missing_ok=True)
                        raise InterruptedError("Téléchargement annulé.")
                    handle.write(chunk)
                    done += len(chunk)
                    on_chunk(done, total)
        part.replace(dest)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def _stream_to_file_with_retries(
    url: str,
    dest: Path,
    on_chunk: Callable[[int, int], None],
    should_cancel: CancelCallback | None,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            _stream_to_file(url, dest, on_chunk, should_cancel)
            return
        except InterruptedError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= _MAX_ATTEMPTS:
                break
            logger.info(
                "Retrying video model file %s after attempt %s/%s failed: %s",
                dest.name,
                attempt,
                _MAX_ATTEMPTS,
                exc,
            )
            time.sleep(float(attempt))
    if last_error is not None:
        raise last_error


def _write_manifest(model_dir: Path, model: dict[str, Any], targets: list[dict[str, str]]) -> None:
    files = [str(target["filename"]) for target in targets]
    manifest = {
        "schema": 1,
        "name": str(model.get("name") or model_dir.name),
        "display_name": str(model.get("display_name") or model.get("name") or model_dir.name),
        "backend": str(model.get("backend") or ""),
        "model_url": str(model.get("model_url") or ""),
        "files": files,
        "downloaded_at": int(time.time()),
    }
    if isinstance(model.get("mlx"), dict):
        manifest["mlx"] = dict(model["mlx"])
    path = model_dir / MODEL_MANIFEST
    tmp = model_dir / f"{MODEL_MANIFEST}.part"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def download_video_model(
    model: dict[str, Any],
    dest_root: Path,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    """Download every file of ``model`` into ``dest_root/<name>/``.

    Returns ``{"ok", "path", "files", "message"}``. Never raises.
    """
    name = str(model.get("name") or "modele")
    targets = resolve_video_download(model)
    if not targets:
        return {
            "ok": False,
            "path": "",
            "files": [],
            "message": "Téléchargement automatique indisponible pour cette source — ouvre la page du modèle.",
        }

    model_dir = dest_root / name
    (model_dir / MODEL_MANIFEST).unlink(missing_ok=True)
    written: list[str] = []
    count = len(targets)
    try:
        for index, target in enumerate(targets):
            dest = model_dir / target["filename"]
            if dest.exists() and dest.stat().st_size > 0:
                written.append(target["filename"])
                if on_progress:
                    size = dest.stat().st_size
                    on_progress(size, size, target["filename"], index, count)
                continue

            def chunk_cb(
                done: int, total: int, _fn: str = target["filename"], _i: int = index
            ) -> None:
                if on_progress:
                    on_progress(done, total, _fn, _i, count)

            _stream_to_file_with_retries(target["url"], dest, chunk_cb, should_cancel)
            written.append(target["filename"])
        _write_manifest(model_dir, model, targets)
    except InterruptedError as exc:
        return {"ok": False, "path": str(model_dir), "files": written, "message": str(exc)}
    except Exception as exc:
        logger.warning("Video model download failed (%s): %s", name, exc)
        return {
            "ok": False,
            "path": str(model_dir),
            "files": written,
            "message": f"Échec du téléchargement : {exc}",
        }

    return {
        "ok": bool(written),
        "path": str(model_dir),
        "files": written,
        "message": f"{model.get('display_name', name)} téléchargé ({count} fichiers) dans {model_dir}.",
    }
