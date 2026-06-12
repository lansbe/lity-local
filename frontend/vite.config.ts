import { fileURLToPath, URL } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The build is emitted straight into the Python package so PyInstaller can
// bundle it as data. `base: './'` makes asset paths relative, which is required
// when the app is loaded from a file:// URL inside the pywebview window.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: fileURLToPath(
      new URL('../src/lity/interfaces/desktop_web/web_dist', import.meta.url),
    ),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
  },
})
