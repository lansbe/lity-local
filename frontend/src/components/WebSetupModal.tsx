import { useCallback, useEffect, useRef, useState } from 'react'

import { bridge } from '../bridge'
import type { WebSetupOutcome, WebStatus } from '../types'
import { Button, Modal } from '../ui'
import { GlobeIcon } from './Icons'

/**
 * First-click web setup: SearXNG isn't running, so offer the one-click local
 * install (Docker) — or continue with the DuckDuckGo/Wikipédia fallback.
 * Progress arrives through `searxng_setup` events while the install runs.
 *
 * `onResolve` reports HOW the modal was closed so the caller can decide whether
 * to remember the choice: 'dismissed' (clicked away — not a decision, keep
 * offering), 'fallback' (use DDG/Wikipédia), or 'installed' (SearXNG is up).
 */
export function WebSetupModal({
  status,
  onResolve,
}: {
  status: WebStatus
  onResolve: (outcome: WebSetupOutcome) => void
}) {
  const [installing, setInstalling] = useState(status.setup_running)
  const [progress, setProgress] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  // Resolve EXACTLY once. The success path schedules a delayed close, so a
  // dismiss in that window (or a duplicate event) must not fire onResolve twice.
  const resolvedRef = useRef(false)
  const resolve = useCallback(
    (outcome: WebSetupOutcome) => {
      if (resolvedRef.current) return
      resolvedRef.current = true
      onResolve(outcome)
    },
    [onResolve],
  )

  useEffect(() => {
    let closeTimer: number | undefined
    const off = bridge.on(
      'searxng_setup',
      (payload: { stage: string; message?: string; done?: boolean; ok?: boolean; url?: string }) => {
        if (!payload.done) {
          setProgress(payload.message || '')
          return
        }
        setInstalling(false)
        if (payload.ok) {
          setSuccess(payload.message || `SearXNG opérationnel sur ${payload.url}.`)
          closeTimer = window.setTimeout(() => resolve('installed'), 1200)
        } else {
          setError(payload.message || "L'installation a échoué.")
          setProgress('')
        }
      },
    )
    return () => {
      off()
      if (closeTimer) window.clearTimeout(closeTimer)
    }
  }, [resolve])

  async function install() {
    setError('')
    setProgress('Lancement de l’installation…')
    setInstalling(true)
    try {
      await bridge.setupSearxng()
    } catch (exc) {
      setInstalling(false)
      setError(String(exc))
    }
  }

  return (
    <Modal
      size="sm"
      title="Activer la recherche web"
      icon={<GlobeIcon className="h-5 w-5" />}
      onClose={() => !installing && resolve('dismissed')}
      closeOnOverlay={!installing}
    >
      <p className="text-callout leading-relaxed text-secondary">
        Aucun moteur de recherche local n'est encore installé. L'app peut installer{' '}
        <span className="font-medium text-primary">SearXNG</span> automatiquement (conteneur Docker
        local, configuré pour toi) : recherches privées, rapides, sans clé d'API.
      </p>

      {!status.docker && !success && (
        <div className="mt-3 rounded-lg bg-warn/10 px-3 py-2.5 text-footnote leading-relaxed text-warn">
          Docker n'est pas détecté sur cette machine. Installe Docker Desktop (docker.com) puis
          reviens ici — ou continue avec les moteurs de repli (DuckDuckGo / Wikipédia).
        </div>
      )}

      {progress && !error && !success && (
        <div className="mt-3 flex items-center gap-2 rounded-lg bg-accent/10 px-3 py-2.5 text-footnote text-accent">
          <span className="h-2 w-2 flex-none animate-pulse rounded-full bg-accent" />
          <span className="min-w-0 flex-1">{progress}</span>
        </div>
      )}
      {error && (
        <div className="mt-3 rounded-lg bg-danger/10 px-3 py-2.5 text-footnote text-danger">{error}</div>
      )}
      {success && (
        <div className="mt-3 rounded-lg bg-success/10 px-3 py-2.5 text-footnote text-success">
          {success} Recherche web activée.
        </div>
      )}

      {!success && (
        <div className="mt-4 flex flex-col gap-2">
          {status.docker && (
            <Button variant="primary" size="lg" block onClick={() => void install()} disabled={installing}>
              {installing ? 'Installation en cours…' : 'Installer SearXNG automatiquement'}
            </Button>
          )}
          <Button variant="secondary" size="lg" block onClick={() => resolve('fallback')} disabled={installing}>
            Continuer sans SearXNG ({status.fallback_ddg ? 'DuckDuckGo + ' : ''}Wikipédia)
          </Button>
        </div>
      )}
    </Modal>
  )
}
