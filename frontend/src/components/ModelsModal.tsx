import { useEffect, useState } from 'react'

import { bridge } from '../bridge'
import { cx } from '../lib/cx'
import type {
  HardwareInfo,
  ImageModelRecommendation,
  ModelDetail,
  ModelRecommendations,
  ModelSuggestion,
  PullStatus,
  VideoModelRecommendation,
  VoicesState,
} from '../types'
import { Badge, Button, Input, Modal, Segmented } from '../ui'
import {
  CheckIcon,
  CpuIcon,
  DownloadIcon,
  ImageIcon,
  StarIcon,
  TrashIcon,
  VideoIcon,
  XIcon,
} from './Icons'

type Tab = 'llm' | 'embeddings' | 'images' | 'videos' | 'voice'
type Tone = 'neutral' | 'accent' | 'success' | 'warn' | 'danger'

const EMPTY_PULL: PullStatus = { active: null, queue: [], progress: {} }

const VERDICT: Record<string, { label: string; tone: Tone }> = {
  excellent: { label: 'Excellent', tone: 'success' },
  bon: { label: 'Bon', tone: 'accent' },
  limite: { label: 'Limite', tone: 'warn' },
  trop_lourd: { label: 'Trop lourd', tone: 'danger' },
}

const GRADE_TONE: Record<string, Tone> = {
  S: 'success',
  A: 'success',
  B: 'accent',
  C: 'warn',
  D: 'warn',
  F: 'danger',
  '?': 'neutral',
}

function deviceSummary(hw: HardwareInfo): string {
  const parts: string[] = []
  if (hw.gpu) parts.push(hw.gpu)
  if (hw.ram_gb) parts.push(`${hw.ram_gb} Go RAM`)
  if (hw.vram_gb) parts.push(`${hw.vram_gb} Go VRAM`)
  if (hw.accelerator) parts.push(hw.accelerator.toUpperCase())
  return parts.join(' · ')
}

export function ModelsModal({ onClose, onChanged }: { onClose: () => void; onChanged?: () => void }) {
  const [tab, setTab] = useState<Tab>('llm')
  const [models, setModels] = useState<ModelDetail[]>([])
  const [reco, setReco] = useState<ModelRecommendations | null>(null)
  const [suggestions, setSuggestions] = useState<ModelSuggestion[]>([])
  const [voices, setVoices] = useState<VoicesState | null>(null)
  const [name, setName] = useState('')
  const [pull, setPull] = useState<PullStatus>(EMPTY_PULL)
  const [notice, setNotice] = useState('')
  const [imagePull, setImagePull] = useState<{ name: string; pct: number } | null>(null)
  const [videoPull, setVideoPull] = useState<{ name: string; pct: number } | null>(null)

  async function refreshModels() {
    try {
      setModels(await bridge.listModelsDetailed())
    } catch {
      setModels([])
    }
    bridge
      .modelRecommendations()
      .then(setReco)
      .catch(() => setReco(null))
  }
  async function refreshVoices() {
    try {
      setVoices(await bridge.listVoices())
    } catch {
      setVoices(null)
    }
  }

  useEffect(() => {
    void refreshModels()
    bridge
      .modelSuggestions()
      .then(setSuggestions)
      .catch(() => setSuggestions([]))
    void refreshVoices()
    bridge
      .pullStatus()
      .then(setPull)
      .catch(() => setPull(EMPTY_PULL))
  }, [])

  useEffect(() => {
    const off1 = bridge.on(
      'pull_progress',
      (payload: PullStatus & { status?: string; completed?: number; total?: number }) => {
        setPull({
          active: payload.active ?? null,
          queue: payload.queue ?? [],
          progress:
            payload.progress ?? { status: payload.status, completed: payload.completed, total: payload.total },
        })
      },
    )
    const off2 = bridge.on(
      'pull_done',
      (payload: PullStatus & { name: string; ok: boolean; message: string; models: ModelDetail[] }) => {
        setModels(payload.models || [])
        bridge.modelRecommendations().then(setReco).catch(() => {})
        setPull({ active: payload.active ?? null, queue: payload.queue ?? [], progress: {} })
        setNotice(payload.ok ? `${payload.name} installé.` : payload.message || `Échec : ${payload.name}`)
        onChanged?.()
      },
    )
    return () => {
      off1()
      off2()
    }
  }, [onChanged])

  // Live progress for image-checkpoint downloads (separate from the Ollama pull).
  useEffect(() => {
    const offProgress = bridge.on('image_pull_progress', (p: { name: string; pct?: number }) => {
      setImagePull({ name: p.name, pct: p.pct ?? 0 })
    })
    const offDone = bridge.on('image_pull_done', (p: { name: string; ok: boolean; message?: string }) => {
      setImagePull(null)
      if (p.message) setNotice(p.message)
      void refreshModels()
      onChanged?.()
    })
    bridge
      .imagePullStatus()
      .then((status) => {
        if (status.active) setImagePull({ name: status.active, pct: 0 })
      })
      .catch(() => {})
    return () => {
      offProgress()
      offDone()
    }
  }, [onChanged])

  // Live progress for video-model downloads (separate from the Ollama pull).
  useEffect(() => {
    const offProgress = bridge.on('video_pull_progress', (p: { name: string; pct?: number }) => {
      setVideoPull({ name: p.name, pct: p.pct ?? 0 })
    })
    const offDone = bridge.on('video_pull_done', async (p: { name: string; ok: boolean; message?: string }) => {
      setVideoPull(null)
      if (p.ok) {
        try {
          await bridge.selectVideoModel(p.name)
        } catch {
          /* keep the download success visible even if selection refresh races */
        }
      }
      if (p.message) setNotice(p.message)
      void refreshModels()
      onChanged?.()
    })
    bridge
      .videoPullStatus()
      .then((status) => {
        if (status.active) setVideoPull({ name: status.active, pct: 0 })
      })
      .catch(() => {})
    return () => {
      offProgress()
      offDone()
    }
  }, [onChanged])

  const installed = (target: string) =>
    models.some((model) => model.name === target || model.name.startsWith(`${target}:`))

  const activePct = pull.progress.total
    ? Math.round(((pull.progress.completed ?? 0) / pull.progress.total) * 100)
    : null

  function pullStateFor(target: string): 'active' | 'queued' | null {
    if (pull.active === target) return 'active'
    if (pull.queue.includes(target)) return 'queued'
    return null
  }

  async function enqueue(target: string) {
    const value = target.trim()
    if (!value) return
    if (value === name) setName('')
    setNotice('')
    try {
      setPull(await bridge.pullModel(value))
    } catch {
      /* ignore — backend stays the source of truth via events */
    }
  }

  // Cancel the active download (no target) or drop a queued model from the line.
  async function cancelPull(target?: string) {
    try {
      setPull(await bridge.cancelPull(target))
    } catch {
      /* ignore — backend stays the source of truth via events */
    }
  }

  async function remove(modelName: string) {
    if (!window.confirm(`Supprimer le modèle ${modelName} ?`)) return
    const result = await bridge.deleteModel(modelName)
    setModels(result.models || [])
    bridge.modelRecommendations().then(setReco).catch(() => {})
    onChanged?.()
  }

  async function downloadVoice(voiceId: string) {
    setNotice('Téléchargement de la voix…')
    await bridge.downloadVoice(voiceId)
    for (const delay of [3000, 6000, 12000]) {
      setTimeout(() => void refreshVoices(), delay)
    }
  }

  async function openModelUrl(model: ImageModelRecommendation) {
    const result = await bridge.openExternal(model.model_url)
    setNotice(
      result.ok
        ? `Page de téléchargement ouverte pour ${model.display_name}.`
        : `Impossible d'ouvrir la page de ${model.display_name}.`,
    )
  }

  // Auto-download a non-Ollama image checkpoint into ~/Documents/Lity/Models/Images.
  async function downloadImage(model: ImageModelRecommendation) {
    setNotice('')
    try {
      const result = await bridge.downloadImageModel(model.name)
      if (result.running) setImagePull({ name: model.name, pct: 0 })
      else if (result.message) setNotice(result.message)
    } catch {
      // Auto-download path unavailable → fall back to opening the model page.
      await openModelUrl(model)
    }
  }

  // Choose which downloaded model the in-process engine generates with.
  async function useImageModel(model: ImageModelRecommendation) {
    try {
      await bridge.selectImageModel(model.name)
      setNotice(`${model.display_name} est maintenant le modèle actif pour la génération.`)
      void refreshModels()
    } catch {
      setNotice(`Impossible de sélectionner ${model.display_name}.`)
    }
  }

  async function openVideoUrl(model: VideoModelRecommendation) {
    const result = await bridge.openExternal(model.model_url)
    setNotice(
      result.ok
        ? `Page de téléchargement ouverte pour ${model.display_name}.`
        : `Impossible d'ouvrir la page de ${model.display_name}.`,
    )
  }

  // Auto-download a video model into ~/Documents/Lity/Models/Videos.
  async function downloadVideo(model: VideoModelRecommendation) {
    setNotice('')
    try {
      const result = await bridge.downloadVideoModel(model.name)
      if (result.running) setVideoPull({ name: model.name, pct: 0 })
      else {
        if (result.ok) await bridge.selectVideoModel(model.name)
        if (result.message) setNotice(result.message)
        void refreshModels()
        onChanged?.()
      }
    } catch {
      // Auto-download path unavailable → fall back to opening the model page.
      await openVideoUrl(model)
    }
  }

  // Choose which downloaded video model the in-process engine generates with.
  async function useVideoModel(model: VideoModelRecommendation) {
    try {
      await bridge.selectVideoModel(model.name)
      setNotice(`${model.display_name} est maintenant le modèle actif pour la génération.`)
      void refreshModels()
    } catch {
      setNotice(`Impossible de sélectionner ${model.display_name}.`)
    }
  }

  function PullControl({ target }: { target: string }) {
    const state = pullStateFor(target)
    if (state === 'active') {
      return (
        <span className="flex flex-none items-center gap-1.5 whitespace-nowrap text-footnote font-medium text-accent">
          <DownloadIcon className="h-3.5 w-3.5 animate-pulse" />
          {activePct ?? 0}%
          <button
            type="button"
            onClick={() => cancelPull(target)}
            title="Annuler le téléchargement"
            aria-label={`Annuler le téléchargement de ${target}`}
            className="rounded p-0.5 text-tertiary transition-colors hover:text-danger"
          >
            <XIcon className="h-3.5 w-3.5" />
          </button>
        </span>
      )
    }
    if (state === 'queued') {
      return (
        <span className="flex flex-none items-center gap-1.5 whitespace-nowrap text-footnote text-tertiary">
          En file
          <button
            type="button"
            onClick={() => cancelPull(target)}
            title="Retirer de la file"
            aria-label={`Retirer ${target} de la file`}
            className="rounded p-0.5 transition-colors hover:text-danger"
          >
            <XIcon className="h-3.5 w-3.5" />
          </button>
        </span>
      )
    }
    return (
      <Button size="sm" variant="primary" icon={<DownloadIcon className="h-3.5 w-3.5" />} onClick={() => enqueue(target)}>
        Tirer
      </Button>
    )
  }

  const embeddingSuggestions = suggestions.filter((s) => s.category === 'embedding')
  const imageRecommendations = reco?.image_models ?? []
  // Backends rendered in-app: single-file SD/SDXL via the in-process
  // diffusers engine ("automatic1111"), and MLX (mflux) models via the
  // out-of-process MLX engine. The rest (ComfyUI / sd.cpp) still need a
  // dedicated runtime not embedded in the app.
  const LOCAL_IMAGE_BACKENDS = new Set(['automatic1111', 'mlx'])
  const localImages = imageRecommendations.filter((model) =>
    LOCAL_IMAGE_BACKENDS.has(model.backend),
  )
  const advancedImages = imageRecommendations.filter(
    (model) => !LOCAL_IMAGE_BACKENDS.has(model.backend),
  )

  const videoRecommendations = reco?.video_models ?? []
  // Diffusers renders in-app; MLX models run through the managed ltx-2-mlx runtime.
  const LOCAL_VIDEO_BACKENDS = new Set(['diffusers', 'mlx'])
  const localVideos = videoRecommendations.filter((model) =>
    LOCAL_VIDEO_BACKENDS.has(model.backend),
  )
  const advancedVideos = videoRecommendations.filter(
    (model) => !LOCAL_VIDEO_BACKENDS.has(model.backend),
  )

  function Suggestions({ items }: { items: ModelSuggestion[] }) {
    return (
      <div className="space-y-1">
        {items.map((item) => (
          <div key={item.name} className="flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2">
            <div className="min-w-0 flex-1">
              <div className="truncate font-mono text-callout text-primary">{item.name}</div>
              <div className="truncate text-caption text-tertiary">{item.note}</div>
            </div>
            {installed(item.name) ? (
              <span className="flex items-center gap-1 text-footnote text-success">
                <CheckIcon className="h-3.5 w-3.5" /> Installé
              </span>
            ) : (
              <PullControl target={item.name} />
            )}
          </div>
        ))}
      </div>
    )
  }

  function ImageRows({ title, items }: { title: string; items: ImageModelRecommendation[] }) {
    if (items.length === 0) return null
    return (
      <div className="space-y-1.5">
        <p className="text-caption font-medium uppercase tracking-wide text-tertiary">{title}</p>
        {items.map((model) => (
          <div
            key={model.name}
            title={model.install_hint}
            className="group flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="truncate font-mono text-callout text-primary">
                  {model.display_name}
                </span>
                {model.recommended && (
                  <span className="flex flex-none items-center gap-0.5 text-caption font-semibold text-accent">
                    <StarIcon className="h-3 w-3" />
                    Recommandé
                  </span>
                )}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-1.5 text-caption text-tertiary">
                {model.grade && (
                  <Badge tone={GRADE_TONE[model.grade] ?? 'neutral'} title={model.grade_label}>
                    {model.grade}
                  </Badge>
                )}
                <Badge tone={VERDICT[model.verdict]?.tone ?? 'neutral'}>
                  {VERDICT[model.verdict]?.label ?? model.verdict}
                </Badge>
                <span>{model.vram_gb} Go VRAM</span>
                <span>· {model.speed}</span>
                <span>· {model.backend}</span>
                <span className="hidden sm:inline">· {model.license}</span>
              </div>
            </div>
            {model.installed ? (
              model.selected ? (
                <span
                  title="Modèle utilisé pour la génération"
                  className="flex flex-none items-center gap-1 whitespace-nowrap text-footnote font-medium text-accent"
                >
                  <CheckIcon className="h-3.5 w-3.5" /> Actif
                </span>
              ) : (
                <Button
                  size="sm"
                  variant="secondary"
                  title="Utiliser ce modèle pour générer"
                  onClick={() => useImageModel(model)}
                >
                  Utiliser
                </Button>
              )
            ) : imagePull?.name === model.name ? (
              <span className="flex flex-none items-center gap-1.5 whitespace-nowrap text-footnote font-medium text-accent">
                <DownloadIcon className="h-3.5 w-3.5 animate-pulse" />
                {imagePull.pct}%
              </span>
            ) : (
              <Button
                size="sm"
                variant="secondary"
                aria-label={`Télécharger ${model.display_name}`}
                icon={<DownloadIcon className="h-3.5 w-3.5" />}
                onClick={() => downloadImage(model)}
                disabled={Boolean(imagePull)}
              >
                <span className="hidden sm:inline">Télécharger</span>
              </Button>
            )}
          </div>
        ))}
      </div>
    )
  }

  function VideoRows({ title, items }: { title: string; items: VideoModelRecommendation[] }) {
    if (items.length === 0) return null
    return (
      <div className="space-y-1.5">
        <p className="text-caption font-medium uppercase tracking-wide text-tertiary">{title}</p>
        {items.map((model) => (
          <div
            key={model.name}
            title={model.install_hint}
            className="group flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="truncate font-mono text-callout text-primary">
                  {model.display_name}
                </span>
                {model.recommended && (
                  <span className="flex flex-none items-center gap-0.5 text-caption font-semibold text-accent">
                    <StarIcon className="h-3 w-3" />
                    Recommandé
                  </span>
                )}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-1.5 text-caption text-tertiary">
                {model.grade && (
                  <Badge tone={GRADE_TONE[model.grade] ?? 'neutral'} title={model.grade_label}>
                    {model.grade}
                  </Badge>
                )}
                <Badge tone={VERDICT[model.verdict]?.tone ?? 'neutral'}>
                  {VERDICT[model.verdict]?.label ?? model.verdict}
                </Badge>
                <span>{model.vram_gb} Go VRAM</span>
                <span>· {model.speed}</span>
                <span>· {model.input_type === 'image' ? 'image→vidéo' : 'texte→vidéo'}</span>
                <span className="hidden sm:inline">· {model.license}</span>
              </div>
            </div>
            {model.installed ? (
              model.selected ? (
                <span
                  title="Modèle utilisé pour la génération"
                  className="flex flex-none items-center gap-1 whitespace-nowrap text-footnote font-medium text-accent"
                >
                  <CheckIcon className="h-3.5 w-3.5" /> Actif
                </span>
              ) : (
                <Button
                  size="sm"
                  variant="secondary"
                  title="Utiliser ce modèle pour générer"
                  onClick={() => useVideoModel(model)}
                >
                  Utiliser
                </Button>
              )
            ) : videoPull?.name === model.name ? (
              <span className="flex flex-none items-center gap-1.5 whitespace-nowrap text-footnote font-medium text-accent">
                <DownloadIcon className="h-3.5 w-3.5 animate-pulse" />
                {videoPull.pct}%
              </span>
            ) : (
              <Button
                size="sm"
                variant="secondary"
                aria-label={`Télécharger ${model.display_name}`}
                icon={<DownloadIcon className="h-3.5 w-3.5" />}
                onClick={() => downloadVideo(model)}
                disabled={Boolean(videoPull)}
              >
                <span className="hidden sm:inline">Télécharger</span>
              </Button>
            )}
          </div>
        ))}
      </div>
    )
  }

  return (
    <Modal
      title="Modèles & dépendances"
      size="lg"
      icon={<CpuIcon className="h-5 w-5" />}
      onClose={onClose}
      bodyClassName="space-y-4"
      footer={
        tab === 'llm' ? (
          <div className="flex w-full items-center gap-2">
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && enqueue(name)}
              placeholder="Autre modèle (ex. : phi3, gemma2)…"
              className="min-w-0 flex-1"
            />
            <Button variant="primary" onClick={() => enqueue(name)} disabled={!name.trim()}>
              Tirer
            </Button>
          </div>
        ) : (
          <Button variant="primary" onClick={onClose}>
            Fermer
          </Button>
        )
      }
    >
      <Segmented<Tab>
        value={tab}
        onChange={setTab}
        options={[
          { value: 'llm', label: 'LLM' },
          { value: 'embeddings', label: 'Embeddings' },
          { value: 'images', label: 'Images' },
          { value: 'videos', label: 'Vidéos' },
          { value: 'voice', label: 'Voix' },
        ]}
      />

      {pull.active && (
        <div className="space-y-1.5 rounded-lg bg-accent/10 px-3 py-2 text-footnote text-accent">
          <div className="flex items-center gap-2">
            <DownloadIcon className="h-3.5 w-3.5 flex-none animate-pulse" />
            <span className="min-w-0 flex-1 truncate">
              Téléchargement : <span className="font-mono">{pull.active}</span>
              {activePct !== null ? ` — ${activePct}%` : ''}
            </span>
            <button
              type="button"
              onClick={() => cancelPull(pull.active ?? undefined)}
              className="flex-none rounded px-1.5 py-0.5 text-tertiary transition-colors hover:bg-danger/10 hover:text-danger"
            >
              Annuler
            </button>
          </div>
          {pull.queue.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 border-t border-accent/15 pt-1.5 text-caption text-tertiary">
              <span className="font-medium">En file ({pull.queue.length}) :</span>
              {pull.queue.map((queued) => (
                <span
                  key={queued}
                  className="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 font-mono"
                >
                  {queued}
                  <button
                    type="button"
                    onClick={() => cancelPull(queued)}
                    title="Retirer de la file"
                    aria-label={`Retirer ${queued} de la file`}
                    className="rounded transition-colors hover:text-danger"
                  >
                    <XIcon className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {tab === 'llm' && (
        <div className="space-y-3">
          {reco && (
            <div className="flex items-center gap-2.5 rounded-xl border border-hairline bg-surface-2/50 px-3 py-2.5">
              <CpuIcon className="h-4 w-4 flex-none text-tertiary" />
              <div className="min-w-0">
                <div className="text-footnote font-medium text-secondary">Votre appareil</div>
                <div className="truncate text-caption text-tertiary">{deviceSummary(reco.hardware)}</div>
              </div>
            </div>
          )}
          <p className="text-caption font-medium uppercase tracking-wide text-tertiary">
            Recommandés pour votre appareil
          </p>
          {!reco && <p className="text-callout text-tertiary">Analyse du matériel…</p>}
          <div className="space-y-1.5">
            {reco?.models
              .filter((model) => model.kind !== 'embed')
              .map((model) => (
                <div
                  key={model.name}
                  className={cx(
                    'group flex items-center gap-2 rounded-lg px-3 py-2',
                    model.recommended ? 'bg-accent/8 ring-1 ring-accent/25' : 'bg-surface-2',
                  )}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate font-mono text-callout text-primary">{model.name}</span>
                      {model.recommended && (
                        <span className="flex flex-none items-center gap-0.5 text-caption font-semibold text-accent">
                          <StarIcon className="h-3 w-3" />
                          Recommandé
                        </span>
                      )}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 text-caption text-tertiary">
                      {model.grade && (
                        <Badge tone={GRADE_TONE[model.grade] ?? 'neutral'} title={model.grade_label}>
                          {model.grade}
                        </Badge>
                      )}
                      <Badge tone={VERDICT[model.verdict]?.tone ?? 'neutral'}>
                        {VERDICT[model.verdict]?.label ?? model.verdict}
                      </Badge>
                      <span>{model.size_gb} Go</span>
                      <span>· {model.speed}</span>
                      <span className="text-tertiary/70">· {model.kind}</span>
                    </div>
                  </div>
                  {model.installed ? (
                    <button
                      type="button"
                      onClick={() => remove(model.name)}
                      title="Supprimer"
                      aria-label="Supprimer"
                      className="flex-none rounded p-1.5 text-tertiary opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                    >
                      <TrashIcon className="h-3.5 w-3.5" />
                    </button>
                  ) : (
                    <PullControl target={model.name} />
                  )}
                </div>
              ))}
          </div>
        </div>
      )}

      {tab === 'embeddings' && (
        <div className="space-y-3">
          <p className="text-callout text-secondary">
            Pour la recherche RAG. Tire un modèle d'embedding, puis sélectionne-le dans Réglages.
          </p>
          <Suggestions items={embeddingSuggestions} />
        </div>
      )}

      {tab === 'images' && (
        <div className="space-y-3">
          {reco && (
            <div className="flex items-center gap-2.5 rounded-xl border border-hairline bg-surface-2/50 px-3 py-2.5">
              <ImageIcon className="h-4 w-4 flex-none text-tertiary" />
              <div className="min-w-0">
                <div className="text-footnote font-medium text-secondary">Votre appareil</div>
                <div className="truncate text-caption text-tertiary">{deviceSummary(reco.hardware)}</div>
              </div>
            </div>
          )}
          {!reco && <p className="text-callout text-tertiary">Analyse du matériel…</p>}
          <p className="rounded-lg bg-surface-2/60 px-3 py-2 text-caption leading-relaxed text-tertiary">
            100 % local, sans serveur externe : télécharge un modèle, clique{' '}
            <span className="font-medium text-secondary">Utiliser</span>, puis active le mode image
            sous le chat et décris ton image.
          </p>
          <ImageRows title="Génération locale intégrée (SD/SDXL · MLX)" items={localImages} />
          <ImageRows title="Avancés · moteur dédié externe requis" items={advancedImages} />
        </div>
      )}

      {tab === 'videos' && (
        <div className="space-y-3">
          {reco && (
            <div className="flex items-center gap-2.5 rounded-xl border border-hairline bg-surface-2/50 px-3 py-2.5">
              <VideoIcon className="h-4 w-4 flex-none text-tertiary" />
              <div className="min-w-0">
                <div className="text-footnote font-medium text-secondary">Votre appareil</div>
                <div className="truncate text-caption text-tertiary">{deviceSummary(reco.hardware)}</div>
              </div>
            </div>
          )}
          {!reco && <p className="text-callout text-tertiary">Analyse du matériel…</p>}
          <p className="rounded-lg bg-surface-2/60 px-3 py-2 text-caption leading-relaxed text-tertiary">
            100 % local, sans serveur externe : télécharge un modèle, clique{' '}
            <span className="font-medium text-secondary">Utiliser</span>, puis active le mode vidéo
            sous le chat et décris ta vidéo. Sur 16 Go, vise des clips courts (~3 s, 480p).
          </p>
          <VideoRows title="Génération locale intégrée (diffusers · MLX)" items={localVideos} />
          <VideoRows title="Avancés · moteur dédié externe requis" items={advancedVideos} />
        </div>
      )}

      {tab === 'voice' && (
        <div className="space-y-3">
          {voices && !voices.available && (
            <p className="text-callout text-tertiary">
              Dépendances audio absentes — installe l'extra audio (faster-whisper, piper-tts,
              sounddevice).
            </p>
          )}
          {voices && voices.available && (
            <>
              {voices.installed.length > 0 && (
                <div>
                  <p className="mb-1.5 text-caption font-medium uppercase tracking-wide text-tertiary">
                    Voix installées
                  </p>
                  {voices.installed.map((voice) => (
                    <label
                      key={voice}
                      className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-callout text-primary"
                    >
                      <input
                        type="radio"
                        className="accent-accent"
                        checked={voices.current === voice}
                        onChange={() => bridge.setVoice(voice).then(() => void refreshVoices())}
                      />
                      <span className="font-mono">{voice}</span>
                    </label>
                  ))}
                </div>
              )}
              <p className="pt-1 text-caption font-medium uppercase tracking-wide text-tertiary">
                Catalogue · Piper
              </p>
              {voices.catalog.map((voice) => (
                <div key={voice.id} className="flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2">
                  <span className="min-w-0 flex-1 truncate text-callout text-primary">{voice.label}</span>
                  {voices.installed.includes(voice.id) ? (
                    <span className="flex items-center gap-1 text-footnote text-success">
                      <CheckIcon className="h-3.5 w-3.5" /> Installée
                    </span>
                  ) : (
                    <Button
                      size="sm"
                      variant="primary"
                      icon={<DownloadIcon className="h-3.5 w-3.5" />}
                      onClick={() => downloadVoice(voice.id)}
                    >
                      Télécharger
                    </Button>
                  )}
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {notice && <p className="text-caption text-tertiary">{notice}</p>}
    </Modal>
  )
}
