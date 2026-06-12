# Lity — Frontend (web UI)

React + Vite + TypeScript + Tailwind interface served inside the pywebview
desktop window. It talks to the Python backend through the `window.pywebview.api`
bridge (see `src/bridge.ts`); streamed tokens arrive via `window.__lityBus`.

## Develop

On macOS, the one-command path from the repository root is:

```bash
./scripts/dev_macos.sh
```

It syncs the Python web-dev environment, repairs/reinstalls frontend
dependencies when needed, frees Vite port `5173` if a stale listener is still
running, starts Vite, then starts the pywebview shell.

For real development, use the pywebview window so the UI talks to the Python
backend. Manually, in one terminal start Vite, in another start the desktop
shell pointed at it:

```bash
cd frontend && npm install && npm run dev   # Vite dev server on :5173
uv run lity --ui web --dev          # pywebview window -> real backend + hot reload
```

A plain browser tab at `http://localhost:5173` has no backend. For visual-only
browser preview, use `http://localhost:5173/?lity_mock=1` to enable the dev mock
explicitly.

## Build

```bash
cd frontend
npm install
npm run build
```

The build is emitted to `../src/lity/interfaces/desktop_web/web_dist/`
so PyInstaller bundles it. After building:

```bash
uv run lity --ui web
```

## Structure

- `src/bridge.ts` — typed wrapper over the pywebview API + event bus
- `src/App.tsx` — state orchestration (conversations, streaming, theme)
- `src/components/` — Sidebar, ChatHeader, MessageList/Message, CodeBlock, Composer, EmptyState,
  WorkspacePanel/DiffCard, StepTimeline, SettingsModal, CharactersModal
