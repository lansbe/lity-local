"""The non-Ollama image-checkpoint downloader: URL resolution + streaming."""

from __future__ import annotations

import lity.services.image_generation.model_download as md


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_resolve_huggingface_a1111_picks_canonical_root_checkpoint(monkeypatch):
    payload = {
        "siblings": [
            {"rfilename": "sd_xl_base_1.0.safetensors"},
            {"rfilename": "sd_xl_base_1.0_0.9vae.safetensors"},
            {"rfilename": "vae/diffusion_pytorch_model.safetensors"},
            {"rfilename": "README.md"},
        ]
    }
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(payload))
    model = {
        "name": "sdxl-base",
        "backend": "automatic1111",
        "model_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
    }
    out = md.resolve_image_download(model)
    # A1111 → exactly one root-level checkpoint, the shortest (canonical) name.
    assert len(out) == 1
    assert out[0]["filename"] == "sd_xl_base_1.0.safetensors"
    assert out[0]["url"].endswith("resolve/main/sd_xl_base_1.0.safetensors?download=true")


def test_resolve_civitai_uses_primary_file(monkeypatch):
    payload = {
        "modelVersions": [
            {
                "files": [
                    {"name": "pruned.safetensors", "primary": False, "downloadUrl": "https://x/2"},
                    {"name": "full.safetensors", "primary": True, "downloadUrl": "https://x/1"},
                ]
            }
        ]
    }
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(payload))
    model = {
        "name": "dreamshaper-xl",
        "backend": "automatic1111",
        "model_url": "https://civitai.com/models/112902/dreamshaper-xl",
    }
    out = md.resolve_image_download(model)
    assert out == [{"url": "https://x/1", "filename": "full.safetensors"}]


def test_unsupported_source_returns_empty():
    assert md.resolve_image_download({"model_url": "https://example.com/x", "backend": "x"}) == []


def test_download_streams_into_named_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        md,
        "resolve_image_download",
        lambda m: [{"url": "http://x/f.safetensors", "filename": "f.safetensors"}],
    )

    class _FakeStream:
        headers = {"content-length": "6"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int = 0):
            yield b"abc"
            yield b"def"

    import httpx

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeStream())
    seen: list[tuple[int, int]] = []
    result = md.download_image_model(
        {
            "name": "sdxl-base",
            "backend": "automatic1111",
            "model_url": "https://huggingface.co/a/b",
        },
        tmp_path,
        on_progress=lambda done, total, fn, i, c: seen.append((done, total)),
    )
    assert result["ok"] is True
    target = tmp_path / "sdxl-base" / "f.safetensors"
    assert target.read_bytes() == b"abcdef"
    assert seen[-1] == (6, 6)
    # No leftover partial file.
    assert not target.with_suffix(".safetensors.part").exists()
