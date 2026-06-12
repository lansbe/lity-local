import { bridge } from '../bridge'

// Local image-mode orchestration, lifted out of the App god-component. Pure
// functions that take their dependencies explicitly (no hooks/closures), so
// behaviour is identical and they are trivially testable.
//
// Image generation is now fully local (no external Stable Diffusion server):
// the first activation may install the engine (~2.5 GB) or ask the user to
// download a model — both surfaced as poll/status messages here.

interface LaunchResult {
  status?: string
  message?: string
  progress?: number
}

export interface ImageSessionDeps {
  setImageActive: (active: boolean) => void
  pushSystem: (content: string) => void
}

// Statuses that end the poll loop without activating image mode.
const TERMINAL = new Set(['error', 'missing', 'stopped', 'no_model'])
// Engine install can be long but not unbounded: ~50 min at a 2 s cadence.
const MAX_TRIES = 1500

export function startImagePoll(deps: ImageSessionDeps): void {
  let tries = 0
  let lastBucket = -1
  const id = setInterval(async () => {
    tries += 1
    let result: LaunchResult
    try {
      result = await bridge.pollImageLaunch()
    } catch {
      return // transient — keep polling
    }
    const status = result?.status ?? ''
    if (status === 'ready') {
      clearInterval(id)
      deps.setImageActive(true)
      deps.pushSystem(result.message || 'Mode image activé.')
      return
    }
    if (TERMINAL.has(status)) {
      clearInterval(id)
      deps.pushSystem(result?.message || "Le mode image n'a pas pu démarrer.")
      return
    }
    // installing / waiting: report progress at 25% milestones so a long
    // install gives feedback without flooding the chat every tick.
    const pct = typeof result?.progress === 'number' ? result.progress : 0
    const bucket = Math.floor(pct / 25)
    if (bucket > lastBucket && result?.message) {
      lastBucket = bucket
      deps.pushSystem(result.message)
    }
    if (tries > MAX_TRIES) {
      clearInterval(id)
      deps.pushSystem('Délai dépassé pour la préparation du mode image.')
    }
  }, 2000)
}

export function handleImageStatus(result: LaunchResult, deps: ImageSessionDeps): void {
  const status = result?.status
  if (status === 'ready') {
    deps.setImageActive(true)
    deps.pushSystem(result.message || "Mode image activé. Décris l'image à générer.")
    return
  }
  if (
    status === 'installing' ||
    status === 'launching' ||
    status === 'waiting' ||
    status === 'launched'
  ) {
    deps.pushSystem(result.message || 'Préparation du mode image…')
    startImagePoll(deps)
    return
  }
  // no_model / error / missing / stopped: surface the guidance and stay off.
  deps.pushSystem(result.message || 'Mode image indisponible.')
}

export async function toggleImageSession(
  imageActive: boolean,
  deps: ImageSessionDeps,
): Promise<void> {
  if (imageActive) {
    await bridge.stopImageSession()
    deps.setImageActive(false)
    deps.pushSystem('Mode image désactivé.')
    return
  }
  handleImageStatus(await bridge.startImageSession(), deps)
}
