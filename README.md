# Lity

Lity is a local-first desktop AI workspace. It gives you a chat interface for
local and CLI-backed models, project-aware context, long-term memory, local
voice tools, image/video generation, and a guarded file-editing workflow.

- Public name: **Lity**
- Repository: `git@github.com:lansbe/lity-local.git`
- Python package: `lity`
- Command: `lity`
- App title: **Lity**

## Highlights

- Chat with Ollama, LM Studio-compatible endpoints, Codex CLI, Claude Code CLI,
  and Grok CLI depending on what is configured locally.
- Keep local JSON memory, conversation history, project workspaces, and file
  context under the Lity data directory.
- Create optional local characters with per-conversation instructions and
  generated emotion portrait packs. Lity itself is the app, not a character.
- Ask the agent to inspect files, search the project, plan, and propose edits
  without writing them until the UI validates the change.
- Index a project with local embeddings for RAG-backed answers.
- Use local STT/TTS with Whisper/Piper integrations.
- Generate images and videos from locally downloaded models when the optional
  runtimes are installed.
- Run as a React/pywebview desktop app, a PySide6 UI, or a console app.
- Extend behavior with local `SKILL.md` folders using the open Agent Skills
  pattern.

## Install

Python 3.10+ is required. `uv` is recommended:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra desktop --extra web --extra dev --extra packaging
```

For the full development/runtime set, including image, document and MCP extras:

```bash
uv sync --extra desktop --extra web --extra image --extra documents --extra mcp --extra dev --extra packaging
```

## Run

Web desktop UI:

```bash
cd frontend
npm install
npm run build
cd ..
uv run lity --ui web
```

Web UI with Vite hot reload:

```bash
cd frontend
npm run dev
```

```bash
uv run lity --ui web --dev
```

One-command macOS development flow:

```bash
./scripts/dev_macos.sh
```

Qt UI:

```bash
uv run lity --ui qt
```

Console:

```bash
uv run lity --console
```

## Project Layout

```text
src/lity/
  app/                  # composition, controller, entrypoints
  core/                 # pure helpers and small domain models
  infrastructure/       # paths, settings, logging
  interfaces/
    cli/                # console adapter
    desktop_qt/         # PySide6 desktop UI
    desktop_web/        # pywebview bridge and bundled web build
  services/
    ai/                 # model providers, prompts, agent loop, tools
    audio/              # STT/TTS integrations
    characters/         # local user-created characters and emotion packs
    commands/           # command policy and execution helpers
    editing/            # guarded file edits
    files/              # workspace and file context
    image_generation/   # local image model management/rendering
    memory/             # JSON memory and multi-conversation store
    rag/                # indexing, retrieval and reranking
    skills/             # Agent Skills discovery, routing and injection
    video_generation/   # local video model management/rendering
    web/                # local web search/fetch helpers
  resources/            # packaged resources and built-in skills
frontend/               # React + Vite + Tailwind UI
packaging/              # PyInstaller release files
scripts/                # development and packaging helpers
tests/                  # deterministic unit/contract tests
```

## User Data

By default, Lity keeps user-visible runtime data under:

```text
~/Documents/Lity/
  config/
  data/
  cache/
  logs/
  Models/
  characters/
  skills/
```

Use `--home` or `LITY_HOME` to point a development run somewhere else:

```bash
uv run lity --home .local-lity --ui web --dev
```

## Frontend

```bash
cd frontend
npm install
npm run dev
npm run build
```

The production build is emitted to
`src/lity/interfaces/desktop_web/web_dist/` so the Python desktop app can bundle
it. Browser-only visual preview is available in Vite dev with:

```text
http://localhost:5173/?lity_mock=1
```

## Quality

```bash
uv lock
uv run ruff check .
uv run ruff format .
uv run pytest
```

Frontend:

```bash
cd frontend
npm run build
```

Tests must stay deterministic and must not require real Ollama, audio, Stable
Diffusion, web search, or video services.

## Packaging

Build on the target OS:

```bash
uv sync --extra desktop --extra web --extra packaging
uv run pyinstaller packaging/pyinstaller/lity.spec --noconfirm
```

Helper scripts:

```bash
./scripts/build_macos.sh
```

```powershell
.\scripts\build_windows.ps1
```

## License And Origin

Lity is released under the MIT License. See [LICENSE](LICENSE).

Lity began from earlier public work in
[AnalogShade/ma-propre-ia-locale](https://github.com/AnalogShade/ma-propre-ia-locale)
and is now maintained as an independent project rather than a GitHub fork. See
[NOTICE.md](NOTICE.md) for the attribution note.
