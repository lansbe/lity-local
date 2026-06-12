import { bridge } from '../bridge'

// Local video-mode orchestration, mirroring imageSession.ts. Pure functions
// that take their dependencies explicitly (no hooks/closures), so behaviour is
// identical and they are trivially testable.
//
// Video generation is fully local (no external server): the first activation
// may install the engine (diffusers + ffmpeg) or ask the user to download a
// model — both surfaced as poll/status messages here.

interface LaunchResult {
  status?: string
  message?: string
  progress?: number
}

export interface VideoSessionDeps {
  setVideoActive: (active: boolean) => void
  pushSystem: (content: string) => void
}

// Statuses that end the poll loop without activating video mode.
const TERMINAL = new Set(['error', 'missing', 'stopped', 'no_model'])
// Engine install can be long but not unbounded: ~50 min at a 2 s cadence.
const MAX_TRIES = 1500

export function startVideoPoll(deps: VideoSessionDeps): void {
  let tries = 0
  let lastBucket = -1
  const id = setInterval(async () => {
    tries += 1
    let result: LaunchResult
    try {
      result = await bridge.pollVideoLaunch()
    } catch {
      return // transient — keep polling
    }
    const status = result?.status ?? ''
    if (status === 'ready') {
      clearInterval(id)
      deps.setVideoActive(true)
      deps.pushSystem(result.message || 'Mode vidéo activé.')
      return
    }
    if (TERMINAL.has(status)) {
      clearInterval(id)
      deps.pushSystem(result?.message || "Le mode vidéo n'a pas pu démarrer.")
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
      deps.pushSystem('Délai dépassé pour la préparation du mode vidéo.')
    }
  }, 2000)
}

export function handleVideoStatus(result: LaunchResult, deps: VideoSessionDeps): void {
  const status = result?.status
  if (status === 'ready') {
    deps.setVideoActive(true)
    deps.pushSystem(result.message || 'Mode vidéo activé. Décris la vidéo à générer.')
    return
  }
  if (
    status === 'installing' ||
    status === 'launching' ||
    status === 'waiting' ||
    status === 'launched'
  ) {
    deps.pushSystem(result.message || 'Préparation du mode vidéo…')
    startVideoPoll(deps)
    return
  }
  // no_model / error / missing / stopped: surface the guidance and stay off.
  deps.pushSystem(result.message || 'Mode vidéo indisponible.')
}

export async function toggleVideoSession(
  videoActive: boolean,
  deps: VideoSessionDeps,
): Promise<void> {
  if (videoActive) {
    await bridge.stopVideoSession()
    deps.setVideoActive(false)
    deps.pushSystem('Mode vidéo désactivé.')
    return
  }
  handleVideoStatus(await bridge.startVideoSession(), deps)
}
