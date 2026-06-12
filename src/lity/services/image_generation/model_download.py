"""Real, automatic download of non-Ollama image checkpoints.

Image models don't come from Ollama: they live on Hugging Face (a repo) or
Civitai (a versioned page). This module resolves a catalog entry's
``model_url`` to the actual file URL(s) via each host's public API, then streams
them — with progress — into ``…/Documents/Lity/Models/Images/<name>/``.

No third-party SDK: just ``httpx`` (already a base dependency). Public models
need no token; gated/Civitai-token-only models surface a clear error instead of
silently failing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# (downloaded_bytes, total_bytes_or_0, filename, file_index, file_count)
ProgressCallback = Callable[[int, int, str, int, int], None]
CancelCallback = Callable[[], bool]

_HF_RE = re.compile(r"huggingface\.co/([^/\s?#]+/[^/\s?#]+)")
_CIVITAI_RE = re.compile(r"civitai\.com/models/(\d+)")
_CHECKPOINT_EXT = (".safetensors", ".ckpt", ".gguf")


def _hf_repo(url: str) -> str | None:
    match = _HF_RE.search(url or "")
    return match.group(1) if match else None


def _civitai_id(url: str) -> str | None:
    match = _CIVITAI_RE.search(url or "")
    return match.group(1) if match else None


def resolve_image_download(model: dict[str, Any]) -> list[dict[str, str]]:
    """Resolve a catalog model to the concrete files to fetch.

    Returns ``[{"url", "filename"}, …]`` (empty if the host is unsupported or
    the API call fails). Raises nothing — callers treat ``[]`` as "can't
    auto-download, fall back to opening the page".
    """
    import httpx

    url = str(model.get("model_url", ""))
    backend = str(model.get("backend", ""))

    repo = _hf_repo(url)
    if repo:
        # Catalog entries can name the exact file (multi-variant repos like
        # SDXL-Lightning ship 1/2/4/8-step checkpoints side by side).
        exact = str(model.get("model_file", "")).strip()
        if exact:
            return [
                {
                    "url": f"https://huggingface.co/{repo}/resolve/main/{exact}?download=true",
                    "filename": exact,
                }
            ]
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
        if backend == "automatic1111":
            # The local diffusers engine loads ONE single-file checkpoint. Pick a
            # root-level weight (no subfolder), preferring .safetensors over a
            # pickled .ckpt (safer, and what from_single_file expects), then the
            # fp16 variant (half the download, what the engine runs in anyway),
            # then EMA-only inference weights, falling back to the canonical
            # (shortest) name.
            roots = [f for f in files if "/" not in f and f.lower().endswith(_CHECKPOINT_EXT)]
            safetensors = [f for f in roots if f.lower().endswith(".safetensors")]
            pool = safetensors or roots
            fp16 = [f for f in pool if "fp16" in f.lower()]
            pool = fp16 or pool
            ema = [f for f in pool if "emaonly" in f.lower() or "ema-only" in f.lower()]
            pick_pool = ema or pool
            chosen = sorted(pick_pool, key=len)[:1] if pick_pool else []
        else:
            # Diffusers / ComfyUI / sd.cpp: the whole repo is the model. Skip
            # docs/images that bloat the download without being weights/config.
            chosen = [
                f
                for f in files
                if not f.lower().endswith((".md", ".png", ".jpg", ".gitattributes"))
            ]
        return [
            {
                "url": f"https://huggingface.co/{repo}/resolve/main/{name}?download=true",
                "filename": name,
            }
            for name in chosen
        ]

    civitai_id = _civitai_id(url)
    if civitai_id:
        try:
            response = httpx.get(
                f"https://civitai.com/api/v1/models/{civitai_id}",
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": "Lity"},
            )
            response.raise_for_status()
            versions = response.json().get("modelVersions", []) or []
        except Exception as exc:
            logger.warning("Civitai listing failed for %s: %s", civitai_id, exc)
            return []
        if not versions:
            return []
        files = versions[0].get("files", []) or []
        primary = next((f for f in files if f.get("primary")), files[0] if files else None)
        if not primary or not primary.get("downloadUrl"):
            return []
        return [
            {
                "url": str(primary["downloadUrl"]),
                "filename": str(primary.get("name") or "model.safetensors"),
            }
        ]

    return []


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
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=timeout, headers={"User-Agent": "Lity"}
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


def download_image_model(
    model: dict[str, Any],
    dest_root: Path,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    """Download every file of ``model`` into ``dest_root/<name>/``.

    Returns ``{"ok", "path", "files", "message"}``. Never raises.
    """
    name = str(model.get("name") or "modele")
    targets = resolve_image_download(model)
    if not targets:
        return {
            "ok": False,
            "path": "",
            "files": [],
            "message": "Téléchargement automatique indisponible pour cette source — ouvre la page du modèle.",
        }

    model_dir = dest_root / name
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

            _stream_to_file(target["url"], dest, chunk_cb, should_cancel)
            written.append(target["filename"])
    except InterruptedError as exc:
        return {"ok": False, "path": str(model_dir), "files": written, "message": str(exc)}
    except Exception as exc:
        logger.warning("Image model download failed (%s): %s", name, exc)
        return {
            "ok": False,
            "path": str(model_dir),
            "files": written,
            "message": f"Échec du téléchargement : {exc}",
        }

    if written:
        # Record which engine renders this model so discovery routes it (and so
        # the diffusers engine never tries to load mlx weights).
        from lity.services.image_generation.checkpoints import write_model_marker

        write_model_marker(model_dir, model)

    return {
        "ok": bool(written),
        "path": str(model_dir),
        "files": written,
        "message": f"{model.get('display_name', name)} téléchargé dans {model_dir}.",
    }
