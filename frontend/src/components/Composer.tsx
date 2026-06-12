import { useEffect, useRef, useState, type ReactNode } from 'react'

import { bridge } from '../bridge'
import { cx } from '../lib/cx'
import { ArrowUpIcon, FileIcon, GlobeIcon, MicIcon, PaperclipIcon, StopIcon, XIcon } from './Icons'

interface ComposerProps {
  busy: boolean
  onSend: (text: string, images?: string[]) => void
  onStop: () => void
  prefill?: string
  prefillNonce?: number
  voiceMode?: boolean
  relistenNonce?: number
  /** Per-conversation model selector, rendered left of the send button. */
  modelPicker?: ReactNode
}

interface Attachment {
  name: string
  dataUrl: string
}

interface TextAttachment {
  name: string
  content: string
}

const MAX_HEIGHT = 220
const TEXT_EXT =
  /\.(txt|md|markdown|py|js|ts|tsx|jsx|json|csv|ya?ml|toml|html?|css|sh|java|c|cpp|h|go|rs|rb|php|xml|ini|cfg|log|sql)$/i
// Match images by EXTENSION too: some formats (notably .avif/.heic) arrive with
// an empty MIME type, so file.type alone misses them.
const IMAGE_EXT = /\.(png|jpe?g|gif|webp|avif|heic|heif|bmp|tiff?|svg)$/i
// Longest side a sent image is downscaled to — keeps the base64 payload (and the
// model's visual-token cost) sane without losing legibility.
const MAX_IMAGE_DIM = 2048

function isTextFile(file: File): boolean {
  return file.type.startsWith('text/') || TEXT_EXT.test(file.name)
}

function isImageFile(file: File): boolean {
  return file.type.startsWith('image/') || IMAGE_EXT.test(file.name)
}

function isDocFile(file: File): boolean {
  return /\.(pdf|docx)$/i.test(file.name) || file.type === 'application/pdf'
}

/**
 * Read an image file and re-encode it to PNG via a canvas. This (a) normalises
 * exotic formats (AVIF/HEIC/WebP) to something every local vision model can
 * decode, and (b) downscales oversized photos. If the browser can't decode the
 * format, fall back to the raw data URL so the backend can still try.
 */
function fileToImageDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error)
    reader.onload = () => {
      const raw = String(reader.result || '')
      if (!raw) return resolve(raw)
      const image = new Image()
      image.onload = () => {
        try {
          let { naturalWidth: width, naturalHeight: height } = image
          const longest = Math.max(width, height)
          if (longest > MAX_IMAGE_DIM) {
            const scale = MAX_IMAGE_DIM / longest
            width = Math.round(width * scale)
            height = Math.round(height * scale)
          }
          const canvas = document.createElement('canvas')
          canvas.width = width
          canvas.height = height
          const ctx = canvas.getContext('2d')
          if (!ctx || !width || !height) return resolve(raw)
          ctx.drawImage(image, 0, 0, width, height)
          resolve(canvas.toDataURL('image/png'))
        } catch {
          resolve(raw) // conversion failed → keep the original
        }
      }
      image.onerror = () => resolve(raw) // undecodable here → let the backend try
      image.src = raw
    }
    reader.readAsDataURL(file)
  })
}

export function Composer({
  busy,
  onSend,
  onStop,
  prefill,
  prefillNonce,
  voiceMode = false,
  relistenNonce = 0,
  modelPicker,
}: ComposerProps) {
  const [text, setText] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [textFiles, setTextFiles] = useState<TextAttachment[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [fetchedUrls, setFetchedUrls] = useState<string[]>([])
  const [loadingUrl, setLoadingUrl] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (prefill) {
      setText(prefill)
      textareaRef.current?.focus()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillNonce])

  useEffect(() => {
    const element = textareaRef.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT)}px`
  }, [text])

  function readFiles(files: FileList | File[]) {
    for (const file of Array.from(files)) {
      if (isImageFile(file)) {
        fileToImageDataUrl(file)
          .then((dataUrl) => {
            if (dataUrl) setAttachments((previous) => [...previous, { name: file.name, dataUrl }])
          })
          .catch(() => {})
      } else if (isTextFile(file)) {
        const reader = new FileReader()
        reader.onload = () => {
          const content = String(reader.result || '')
          setTextFiles((previous) => [...previous, { name: file.name, content }])
        }
        reader.readAsText(file)
      } else if (isDocFile(file)) {
        const reader = new FileReader()
        reader.onload = async () => {
          const dataUrl = String(reader.result || '')
          try {
            const result = await bridge.extractDocument(file.name, dataUrl)
            setTextFiles((previous) => [
              ...previous,
              {
                name: file.name,
                content: result.ok
                  ? result.text
                  : `[${file.name} : ${result.error || 'extraction impossible'}]`,
              },
            ])
          } catch {
            setTextFiles((previous) => [
              ...previous,
              { name: file.name, content: `[${file.name} : extraction indisponible]` },
            ])
          }
        }
        reader.readAsDataURL(file)
      } else {
        setTextFiles((previous) => [
          ...previous,
          { name: file.name, content: '[format binaire non supporté — texte uniquement]' },
        ])
      }
    }
  }

  function submit() {
    const trimmed = text.trim()
    if ((!trimmed && attachments.length === 0 && textFiles.length === 0) || busy) return
    const prefix = textFiles
      .map((file) => `Fichier ${file.name} :\n\`\`\`\n${file.content}\n\`\`\``)
      .join('\n\n')
    const fullText = prefix ? (trimmed ? `${prefix}\n\n${trimmed}` : prefix) : trimmed
    onSend(fullText, attachments.length ? attachments.map((a) => a.dataUrl) : undefined)
    setText('')
    setAttachments([])
    setTextFiles([])
  }

  async function toggleMic() {
    if (recording) {
      setRecording(false)
      setTranscribing(true)
      try {
        const result = await bridge.stopRecording()
        const said = (result.text || '').trim()
        if (voiceMode && said) {
          onSend(said)
        } else if (said) {
          setText((current) => (current ? `${current} ${said}` : said))
        }
      } finally {
        setTranscribing(false)
      }
      return
    }
    const result = await bridge.startRecording()
    if (result.ok) setRecording(true)
  }

  useEffect(() => {
    if (voiceMode) {
      bridge
        .startRecording()
        .then((result) => {
          if (result.ok) setRecording(true)
        })
        .catch(() => {})
    } else {
      setRecording((wasRecording) => {
        if (wasRecording) void bridge.stopRecording().catch(() => {})
        return false
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voiceMode])

  useEffect(() => {
    if (!voiceMode || relistenNonce === 0) return
    bridge
      .startRecording()
      .then((result) => {
        if (result.ok) setRecording(true)
      })
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [relistenNonce])

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  async function addPage(url: string) {
    setLoadingUrl(true)
    try {
      const result = await bridge.fetchPage(url)
      const title = result.title || url
      setTextFiles((previous) => [
        ...previous,
        {
          name: title,
          content: result.ok
            ? `Page web « ${title} » (${url}) :\n${result.text}`
            : `[${url} : ${result.error || 'lecture impossible'}]`,
        },
      ])
      setFetchedUrls((previous) => [...previous, url])
    } finally {
      setLoadingUrl(false)
    }
  }

  const urlMatch = text.match(/https?:\/\/[^\s]+/)
  const urlCandidate = urlMatch ? urlMatch[0].replace(/[.,;:!?)]+$/, '') : null
  const showUrlChip = Boolean(urlCandidate) && !fetchedUrls.includes(urlCandidate as string)
  const urlLabel = (() => {
    try {
      return new URL(urlCandidate as string).hostname
    } catch {
      return urlCandidate
    }
  })()

  const canSend = Boolean(text.trim()) || attachments.length > 0 || textFiles.length > 0

  return (
    <div className="px-4 pb-5 pt-2">
      <div className="mx-auto max-w-[45rem]">
        <div
          onDragOver={(event) => {
            event.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragOver(false)
            readFiles(event.dataTransfer.files)
          }}
          className={cx(
            'rounded-2xl border bg-surface px-3 py-2.5 shadow-sm transition-colors focus-within:border-accent/70',
            dragOver ? 'border-accent' : 'border-hairline-strong',
          )}
        >
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((attachment, index) => (
                <div key={index} className="relative">
                  <img
                    src={attachment.dataUrl}
                    alt={attachment.name}
                    className="h-16 w-16 rounded-lg border border-hairline object-cover"
                  />
                  <button
                    type="button"
                    onClick={() => setAttachments((prev) => prev.filter((_, i) => i !== index))}
                    className="absolute -right-1.5 -top-1.5 rounded-full bg-primary/80 p-0.5 text-surface backdrop-blur transition-colors hover:bg-primary"
                    aria-label="Retirer l'image"
                  >
                    <XIcon className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
          {textFiles.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {textFiles.map((file, index) => (
                <span
                  key={index}
                  className="flex items-center gap-1.5 rounded-lg bg-surface-2 py-1 pl-2 pr-1.5 text-footnote text-secondary"
                  title={file.name}
                >
                  <FileIcon className="h-3.5 w-3.5 flex-none text-tertiary" />
                  <span className="max-w-[180px] truncate">{file.name}</span>
                  <button
                    type="button"
                    onClick={() => setTextFiles((prev) => prev.filter((_, i) => i !== index))}
                    className="rounded p-0.5 text-tertiary hover:text-danger"
                    aria-label="Retirer le fichier"
                  >
                    <XIcon className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
          {showUrlChip && (
            <div className="mb-2">
              <button
                type="button"
                onClick={() => addPage(urlCandidate as string)}
                disabled={loadingUrl}
                className="inline-flex items-center gap-1.5 rounded-lg border border-accent/30 bg-accent/10 px-2.5 py-1 text-footnote font-medium text-accent transition-colors hover:bg-accent/15 disabled:opacity-50"
                title="Lire la page et l'ajouter au contexte"
              >
                <GlobeIcon className="h-3.5 w-3.5" />
                {loadingUrl ? 'Lecture de la page…' : `Lire : ${urlLabel}`}
              </button>
            </div>
          )}

          <textarea
            ref={textareaRef}
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder="Écris un message…"
            className="block max-h-[220px] w-full resize-none bg-transparent px-1 py-1 text-body-lg text-primary outline-none placeholder:text-tertiary"
          />

          <div className="flex items-center gap-2 pt-1.5">
            <div className="flex flex-none items-center gap-0.5">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                title="Joindre un fichier (image, texte, PDF, DOCX)"
                aria-label="Joindre un fichier"
                className="flex h-8 w-8 items-center justify-center rounded-md text-secondary transition-colors hover:bg-surface-2 hover:text-primary"
              >
                <PaperclipIcon className="h-[18px] w-[18px]" />
              </button>
              <button
                type="button"
                onClick={toggleMic}
                title={recording ? 'Arrêter et transcrire' : 'Dictée vocale'}
                aria-label="Dictée vocale"
                className={cx(
                  'flex h-8 w-8 items-center justify-center rounded-md transition-colors',
                  recording
                    ? 'animate-pulse bg-danger/15 text-danger'
                    : 'text-secondary hover:bg-surface-2 hover:text-primary',
                  transcribing && 'opacity-50',
                )}
              >
                <MicIcon className="h-[18px] w-[18px]" />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                onChange={(event) => {
                  if (event.target.files) readFiles(event.target.files)
                  event.target.value = ''
                }}
              />
            </div>

            <div className="ml-auto flex min-w-0 items-center gap-1.5">
              {modelPicker}
              {busy ? (
                <button
                  type="button"
                  onClick={onStop}
                  title="Arrêter la génération"
                  aria-label="Arrêter"
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-surface-2 text-primary transition-colors hover:bg-surface-3"
                >
                  <StopIcon className="h-4 w-4" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={submit}
                  disabled={!canSend}
                  title="Envoyer"
                  aria-label="Envoyer"
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-solid text-solid-contrast transition-[background-color,transform] hover:bg-solid-hover active:scale-95 disabled:cursor-not-allowed disabled:bg-surface-3 disabled:text-tertiary"
                >
                  <ArrowUpIcon className="h-[18px] w-[18px]" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
