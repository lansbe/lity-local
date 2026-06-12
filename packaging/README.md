# Packaging

Lity is prepared for macOS and Windows packaging, but release builds should be produced on the target OS.

## PyInstaller

Install packaging tools:

```bash
uv sync --extra desktop --extra packaging
```

Build on macOS:

```bash
./scripts/build_macos.sh
```

Build on Windows:

```powershell
.\scripts\build_windows.ps1
```

The scripts call:

```bash
uv run pyinstaller packaging/pyinstaller/lity.spec --noconfirm
```

The PyInstaller spec currently targets a slim desktop bundle. It excludes the
heaviest audio/STT/TTS, reranker, and image-processing Python libraries from
the packaged app even though voice, RAG reranking, and web search are part of
base development installs. Those capabilities still degrade gracefully in the
app when their optional runtime services or bundled Python libraries are not
present; image processing stays optional via `--extra image`.

## Qt Deployment

PySide6 also ships `pyside6-deploy`. It can become the preferred release path once icons, signing, notarization, and Windows installer details are finalized.

## Runtime Data

Packaged runtime assets live under `src/lity/resources`. User data, generated
images/videos, local character profiles, downloaded models, settings, skills,
voices, cache files and logs live under `~/Documents/Lity/` by default, or under
`LITY_HOME` / `--home` when overridden for development or tests.
