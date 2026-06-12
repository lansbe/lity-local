import type { ReactNode } from 'react'

import { cx } from '../lib/cx'
import { CheckIcon, GlobeIcon, ImageIcon, MicIcon, TerminalIcon, VideoIcon, ZapIcon } from './Icons'

type Tone = 'accent' | 'warn' | 'danger'

interface ModeBarProps {
  agentMode: boolean
  webSearch: boolean
  yolo: boolean
  imageActive: boolean
  videoActive: boolean
  voiceMode: boolean
  allowCommands: boolean
  commandApproval: boolean
  onToggleAgent: () => void
  onToggleWeb: () => void
  onToggleYolo: () => void
  onToggleImage: () => void
  onToggleVideo: () => void
  onToggleVoice: () => void
  onToggleCommands: () => void
  onToggleApproval: () => void
}

const TONE_ACTIVE: Record<Tone, string> = {
  accent: 'bg-accent/12 text-accent',
  warn: 'bg-warn/14 text-warn',
  danger: 'bg-danger/12 text-danger',
}

function Chip({
  active,
  tone = 'accent',
  icon,
  label,
  title,
  onClick,
}: {
  active: boolean
  tone?: Tone
  icon: ReactNode
  label: string
  title: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-pressed={active}
      className={cx(
        'inline-flex h-7 flex-none items-center gap-1.5 rounded-full px-2.5 text-footnote font-medium transition-colors duration-fast outline-none focus-visible:ring-2 focus-visible:ring-accent/40',
        active ? TONE_ACTIVE[tone] : 'text-tertiary hover:bg-surface-2 hover:text-secondary',
      )}
    >
      <span className="flex-none">{icon}</span>
      {label}
    </button>
  )
}

/** Compact, icon-first mode toggles anchored above the composer. Active state is
 *  conveyed by colour (accent = on, amber = caution, red = autonomous), not text. */
export function ModeBar({
  agentMode,
  webSearch,
  yolo,
  imageActive,
  videoActive,
  voiceMode,
  allowCommands,
  commandApproval,
  onToggleAgent,
  onToggleWeb,
  onToggleYolo,
  onToggleImage,
  onToggleVideo,
  onToggleVoice,
  onToggleCommands,
  onToggleApproval,
}: ModeBarProps) {
  return (
    <div className="no-scrollbar flex items-center gap-1 overflow-x-auto">
      <Chip
        active={agentMode}
        icon={<ZapIcon className="h-3.5 w-3.5" />}
        label="Agent"
        title="Mode agent : l'IA inspecte le projet avec des outils avant de répondre"
        onClick={onToggleAgent}
      />
      <Chip
        active={webSearch}
        icon={<GlobeIcon className="h-3.5 w-3.5" />}
        label="Web"
        title="Recherche web : l'agent cherche et lit des pages (SearXNG / DuckDuckGo / Wikipédia)"
        onClick={onToggleWeb}
      />
      <Chip
        active={yolo}
        tone="danger"
        icon={<ZapIcon className="h-3.5 w-3.5" />}
        label="Autonome"
        title="Mode autonome : l'agent crée/modifie les fichiers et exécute des commandes sans valider chaque changement"
        onClick={onToggleYolo}
      />
      <Chip
        active={imageActive}
        icon={<ImageIcon className="h-3.5 w-3.5" />}
        label="Image"
        title="Mode image : génère des images avec Stable Diffusion"
        onClick={onToggleImage}
      />
      <Chip
        active={videoActive}
        icon={<VideoIcon className="h-3.5 w-3.5" />}
        label="Vidéo"
        title="Mode vidéo : génère de courtes vidéos en local (Wan)"
        onClick={onToggleVideo}
      />
      <Chip
        active={voiceMode}
        icon={<MicIcon className="h-3.5 w-3.5" />}
        label="Vocal"
        title="Conversation vocale mains-libres : parle, la réponse est lue à voix haute, puis le micro réécoute"
        onClick={onToggleVoice}
      />

      {agentMode && !yolo && (
        <>
          <span className="mx-0.5 h-4 w-px bg-hairline" />
          <Chip
            active={allowCommands}
            tone="warn"
            icon={<TerminalIcon className="h-3.5 w-3.5" />}
            label="Commandes"
            title="Autoriser l'agent à exécuter des commandes shell dans le dossier de travail"
            onClick={onToggleCommands}
          />
        </>
      )}
      {agentMode && (allowCommands || yolo) && (
        <Chip
          active={commandApproval}
          icon={<CheckIcon className="h-3.5 w-3.5" />}
          label="Validation"
          title="Demander une confirmation avant chaque commande"
          onClick={onToggleApproval}
        />
      )}
    </div>
  )
}
