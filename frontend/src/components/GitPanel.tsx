import { useEffect, useState } from 'react'

import { bridge } from '../bridge'
import type { GitStatus } from '../types'
import { Button, IconButton, Textarea } from '../ui'
import { RefreshIcon } from './Icons'

export function GitPanel() {
  const [status, setStatus] = useState<GitStatus | null>(null)
  const [message, setMessage] = useState('')
  const [committing, setCommitting] = useState(false)
  const [note, setNote] = useState('')

  async function refresh() {
    try {
      setStatus(await bridge.gitStatus())
    } catch {
      setStatus(null)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  async function commit() {
    if (!message.trim() || committing) return
    setCommitting(true)
    try {
      const result = await bridge.gitCommit(message)
      setNote(result.message)
      if (result.ok) {
        setMessage('')
        if (result.status) setStatus(result.status)
      }
    } finally {
      setCommitting(false)
    }
  }

  if (status === null) {
    return <p className="px-1 py-6 text-center text-footnote text-tertiary">Chargement…</p>
  }
  if (!status.is_repo) {
    return (
      <p className="px-1 py-6 text-center text-footnote text-tertiary">
        Le dossier de travail n'est pas un dépôt Git.
      </p>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-callout text-secondary">
          Branche <span className="font-mono text-primary">{status.branch || '—'}</span>
        </span>
        <IconButton size="sm" label="Rafraîchir" onClick={refresh}>
          <RefreshIcon className="h-4 w-4" />
        </IconButton>
      </div>

      <div className="space-y-0.5">
        {status.files.length === 0 ? (
          <p className="px-1 py-2 text-footnote text-tertiary">Aucun changement.</p>
        ) : (
          status.files.map((file) => (
            <div key={file.path} className="flex items-center gap-2 px-1 text-footnote">
              <span className="w-6 flex-none font-mono text-accent">{file.status || '?'}</span>
              <span className="truncate font-mono text-secondary">{file.path}</span>
            </div>
          ))
        )}
      </div>

      <div className="border-t border-hairline pt-3">
        <Textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          rows={2}
          placeholder="Message de commit (git add -A puis commit)"
        />
        <Button
          variant="primary"
          block
          className="mt-2"
          onClick={commit}
          disabled={committing || !message.trim() || status.files.length === 0}
        >
          {committing ? 'Commit…' : 'Committer'}
        </Button>
        {note && <p className="mt-1.5 text-footnote text-tertiary">{note}</p>}
      </div>
    </div>
  )
}
