import { useState } from 'react'

import { bridge } from '../bridge'
import type { ChatMessage } from '../types'
import { splitReasoning } from '../lib/reasoning'
import { extractSources } from '../lib/sources'
import { stripFileBlocks } from '../lib/strip'
import { CheckIcon, CopyIcon, PencilIcon, RefreshIcon, VolumeIcon } from './Icons'
import { Markdown } from './Markdown'
import { ReasoningBlock } from './ReasoningBlock'
import { SourceCards } from './SourceCards'
import { StepTimeline } from './StepTimeline'

interface MessageProps {
  message: ChatMessage
  characterAvatar?: string
  onRegenerate?: () => void
  onEdit?: (text: string) => void
}

// Human-readable turn duration: one decimal under 10 s, whole seconds up to
// a minute, then "m min s".
function formatElapsed(ms: number): string {
  const seconds = ms / 1000
  if (seconds < 60) {
    const value = seconds < 10 ? seconds.toFixed(1) : String(Math.round(seconds))
    return `${value.replace('.', ',')} s`
  }
  const minutes = Math.floor(seconds / 60)
  return `${minutes} min ${Math.round(seconds % 60)} s`
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-2">
      {[0, 1, 2].map((index) => (
        <span
          key={index}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-tertiary"
          style={{ animationDelay: `${index * 0.15}s` }}
        />
      ))}
    </div>
  )
}

function ActionButton({
  onClick,
  title,
  children,
}: {
  onClick: () => void
  title: string
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      className="rounded-md p-1.5 text-tertiary transition-colors hover:bg-surface-2 hover:text-primary"
    >
      {children}
    </button>
  )
}

export function Message({ message, characterAvatar = '', onRegenerate, onEdit }: MessageProps) {
  const [copied, setCopied] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(message.content)
  const [speaking, setSpeaking] = useState(false)

  // Spoken/copied text excludes the <think> reasoning (answer only).
  const spokenText = splitReasoning(message.content).answer || message.content

  async function toggleSpeak() {
    if (speaking) {
      await bridge.stopSpeaking()
      setSpeaking(false)
      return
    }
    const result = await bridge.speak(spokenText)
    if (result.ok) setSpeaking(true)
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(spokenText)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }

  const copyButton = (
    <ActionButton onClick={copy} title="Copier">
      {copied ? <CheckIcon className="h-3.5 w-3.5" /> : <CopyIcon className="h-3.5 w-3.5" />}
    </ActionButton>
  )

  if (message.role === 'system') {
    return (
      <div className="my-1.5 flex justify-center px-2">
        <div className="max-w-[42rem] select-text whitespace-pre-wrap rounded-lg border border-hairline bg-surface-2/70 px-3.5 py-2 text-footnote leading-relaxed text-secondary">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.role === 'user') {
    if (editing) {
      return (
        <div className="flex justify-end">
          <div className="w-full max-w-[80%]">
            <textarea
              value={draft}
              autoFocus
              rows={3}
              onChange={(event) => setDraft(event.target.value)}
              className="w-full resize-none rounded-2xl border border-hairline-strong bg-surface p-3 text-body text-primary outline-none transition-colors focus:border-accent/70"
            />
            <div className="mt-2 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setEditing(false)
                  setDraft(message.content)
                }}
                className="rounded-md px-3 py-1.5 text-callout text-secondary transition-colors hover:bg-surface-2"
              >
                Annuler
              </button>
              <button
                type="button"
                onClick={() => {
                  const text = draft.trim()
                  setEditing(false)
                  if (text) onEdit?.(text)
                }}
                className="rounded-md bg-solid px-3 py-1.5 text-callout font-medium text-solid-contrast transition-colors hover:bg-solid-hover"
              >
                Envoyer
              </button>
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="group flex flex-col items-end">
        {message.images && message.images.length > 0 && (
          <div className="mb-1.5 flex max-w-[80%] flex-wrap justify-end gap-1.5">
            {message.images.map((src, index) => (
              <img
                key={index}
                src={src}
                alt="Image jointe"
                className="h-28 w-28 rounded-xl border border-hairline object-cover"
              />
            ))}
          </div>
        )}
        {message.content && (
          <div className="max-w-[80%] select-text whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-surface-2 px-4 py-2.5 text-body-lg text-primary">
            {message.content}
          </div>
        )}
        <div className="mt-1 flex gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
          {copyButton}
          {onEdit && (
            <ActionButton
              onClick={() => {
                setDraft(message.content)
                setEditing(true)
              }}
              title="Éditer et renvoyer"
            >
              <PencilIcon className="h-3.5 w-3.5" />
            </ActionButton>
          )}
        </div>
      </div>
    )
  }

  const displayContent = stripFileBlocks(message.content)
  const { reasoning, answer, open: reasoningOpen } = splitReasoning(displayContent)
  const sources = message.pending ? [] : extractSources(message.steps)

  return (
    <div className="group flex gap-3">
      {characterAvatar && (
        <div className="mt-0.5 h-8 w-8 flex-none overflow-hidden rounded-md border border-hairline bg-surface-2">
          <img src={characterAvatar} alt="" className="h-full w-full object-cover" />
        </div>
      )}
      <div className="min-w-0 flex-1 select-text pt-0.5 text-primary">
        {message.steps && message.steps.length > 0 && <StepTimeline steps={message.steps} />}
        {message.image && (
          <img
            src={message.image}
            alt="Image générée"
            className="mb-2 max-w-sm rounded-xl border border-hairline"
          />
        )}
        {message.video && (
          <video
            src={message.video}
            controls
            loop
            className="mb-2 max-w-sm rounded-xl border border-hairline"
          />
        )}
        {reasoning !== null && <ReasoningBlock text={reasoning} open={reasoningOpen} />}
        {answer ? (
          <>
            <Markdown content={answer} />
            {message.pending && (
              <span className="ml-0.5 inline-block h-4 w-[3px] animate-blink bg-primary align-text-bottom" />
            )}
          </>
        ) : message.pending ? (
          // Reasoning may be streaming, or the turn just started — answer pending.
          <TypingDots />
        ) : null}
        {sources.length > 0 && <SourceCards sources={sources} />}
        {!message.pending && (message.content || message.image || message.video) && (
          <div className="mt-1 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
            {message.content && copyButton}
            {message.content && (
              <ActionButton onClick={toggleSpeak} title={speaking ? 'Arrêter la lecture' : 'Lire à voix haute'}>
                <VolumeIcon className={`h-3.5 w-3.5 ${speaking ? 'text-accent' : ''}`} />
              </ActionButton>
            )}
            {onRegenerate && (
              <ActionButton onClick={onRegenerate} title="Régénérer">
                <RefreshIcon className="h-3.5 w-3.5" />
              </ActionButton>
            )}
            {typeof message.elapsedMs === 'number' && (
              <span
                className="ml-1 select-none text-footnote text-tertiary tabular-nums"
                title="Temps de réponse"
              >
                {formatElapsed(message.elapsedMs)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
