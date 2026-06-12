import React from 'react'
import ReactDOM from 'react-dom/client'

import App from './App'
import './index.css'
import 'highlight.js/styles/atom-one-dark.css'
import 'katex/dist/katex.min.css'

// Browser-preview only: stub the pywebview backend so the UI renders without
// the Python process. Keep it opt-in because the pywebview bridge is injected
// after the Vite page starts loading, and an eager mock can win that race.
const shouldInstallDevMock =
  new URLSearchParams(window.location.search).get('lity_mock') === '1' ||
  import.meta.env.VITE_LITY_MOCK === '1'

if (import.meta.env.DEV && shouldInstallDevMock) {
  const { installDevBridge } = await import('./devMock')
  installDevBridge()
}

const root = document.getElementById('root')
if (!root) throw new Error('Élément #root introuvable')

// The bridge (src/bridge.ts) waits for the real pywebview backend.
ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
