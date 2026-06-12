import { useMemo, useState } from 'react'

import type { Change, LoadedFile } from '../types'
import { cx } from '../lib/cx'
import { Badge, Button, IconButton, Segmented, Toggle } from '../ui'
import { DiffCard } from './DiffCard'
import { GitPanel } from './GitPanel'
import { FileIcon, FolderIcon, RefreshIcon, SearchIcon, XIcon } from './Icons'

interface WorkspacePanelProps {
  changes: Change[]
  onApply: (change: Change) => void
  onReject: (change: Change) => void
  files: string[]
  workdir: string
  loaded: LoadedFile[]
  onLoadFile: (path: string) => void
  onCloseFile: (path: string) => void
  onChooseWorkdir: () => void
  onRefreshFiles: () => void
  onClose: () => void
  ragEnabled: boolean
  indexedChunks: number
  indexing: boolean
  onIndexProject: () => void
  onToggleRag: () => void
  changeCount: number
  onUndo: () => void
}

type Tab = 'changes' | 'files' | 'git'

export function WorkspacePanel(props: WorkspacePanelProps) {
  const pendingCount = props.changes.filter((change) => change.status === 'pending').length
  const [tab, setTab] = useState<Tab>(pendingCount > 0 ? 'changes' : 'files')
  const [query, setQuery] = useState('')

  const filteredFiles = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return needle ? props.files.filter((file) => file.toLowerCase().includes(needle)) : props.files
  }, [props.files, query])

  const loadedPaths = new Set(props.loaded.map((file) => file.rel))

  return (
    <aside className="flex h-full w-full flex-col border-l border-hairline bg-panel">
      <div className="flex items-center gap-2 px-3 py-2.5">
        <span className="flex-1 text-callout font-semibold text-primary">Atelier de code</span>
        <IconButton size="sm" label="Fermer le panneau" onClick={props.onClose}>
          <XIcon className="h-4 w-4" />
        </IconButton>
      </div>

      <div className="px-3 pb-3">
        <Segmented<Tab>
          className="w-full"
          value={tab}
          onChange={setTab}
          options={[
            {
              value: 'changes',
              label: (
                <span className="flex items-center gap-1.5">
                  Changements
                  {pendingCount > 0 && <Badge tone="accent">{pendingCount}</Badge>}
                </span>
              ),
            },
            { value: 'files', label: 'Fichiers' },
            { value: 'git', label: 'Git' },
          ]}
        />
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-3">
        {tab === 'changes' && props.changeCount > 0 && (
          <Button
            variant="secondary"
            size="sm"
            block
            className="mb-3"
            icon={<RefreshIcon className="h-3.5 w-3.5" />}
            onClick={props.onUndo}
          >
            Annuler le dernier changement ({props.changeCount})
          </Button>
        )}
        {tab === 'git' ? (
          <GitPanel />
        ) : tab === 'changes' ? (
          props.changes.length === 0 ? (
            <p className="px-1 py-10 text-center text-footnote leading-relaxed text-tertiary">
              Les fichiers créés ou modifiés proposés par l'assistant apparaîtront ici, avec un diff
              et un bouton Appliquer.
            </p>
          ) : (
            <div className="space-y-3">
              {props.changes.map((change) => (
                <DiffCard key={change.id} change={change} onApply={props.onApply} onReject={props.onReject} />
              ))}
            </div>
          )
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={props.onChooseWorkdir}
                title={props.workdir || 'Choisir un dossier de travail'}
                className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-md border border-hairline bg-surface px-2.5 text-callout text-secondary shadow-xs transition-colors hover:bg-surface-2 hover:text-primary"
              >
                <FolderIcon className="h-4 w-4 flex-none text-tertiary" />
                <span className="truncate">
                  {props.workdir ? props.workdir.split(/[\\/]/).filter(Boolean).pop() : 'Choisir un dossier'}
                </span>
              </button>
              <IconButton label="Rafraîchir" onClick={props.onRefreshFiles}>
                <RefreshIcon className="h-4 w-4" />
              </IconButton>
            </div>

            {props.loaded.length > 0 && (
              <div>
                <p className="mb-1.5 text-caption font-medium uppercase tracking-wide text-tertiary">
                  Contexte chargé
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {props.loaded.map((file) => (
                    <span
                      key={file.path}
                      className="flex items-center gap-1 rounded-full bg-accent/12 py-0.5 pl-2 pr-1 text-footnote text-accent"
                      title={file.rel}
                    >
                      {file.name}
                      <button
                        type="button"
                        onClick={() => props.onCloseFile(file.rel)}
                        className="rounded-full p-0.5 hover:bg-accent/20"
                        aria-label="Retirer du contexte"
                      >
                        <XIcon className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-xl border border-hairline bg-surface p-3 shadow-xs">
              <div className="flex items-center justify-between">
                <span className="text-footnote font-medium text-secondary">Connaissance projet · RAG</span>
                {props.indexedChunks > 0 && (
                  <Toggle checked={props.ragEnabled} onChange={props.onToggleRag} label="Activer le RAG" />
                )}
              </div>
              <Button
                variant="primary"
                size="sm"
                block
                className="mt-2.5"
                onClick={props.onIndexProject}
                disabled={props.indexing || !props.workdir}
              >
                {props.indexing
                  ? 'Indexation…'
                  : props.indexedChunks > 0
                    ? `Réindexer · ${props.indexedChunks} extraits`
                    : 'Indexer le projet'}
              </Button>
              <p className="mt-2 text-caption leading-relaxed text-tertiary">
                Indexe les fichiers pour que l'IA réponde sur tout le dépôt (nécessite un modèle
                d'embedding, ex. : ollama pull nomic-embed-text).
              </p>
            </div>

            {props.workdir ? (
              <>
                <div className="flex h-9 items-center gap-2 rounded-md border border-hairline bg-surface px-2.5 transition-colors focus-within:border-accent/70">
                  <SearchIcon className="h-4 w-4 flex-none text-tertiary" />
                  <input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Filtrer les fichiers…"
                    className="w-full bg-transparent text-callout text-primary outline-none placeholder:text-tertiary"
                  />
                </div>
                <div className="space-y-px">
                  {filteredFiles.length === 0 && (
                    <p className="px-1 py-4 text-center text-footnote text-tertiary">Aucun fichier</p>
                  )}
                  {filteredFiles.map((file) => (
                    <button
                      key={file}
                      type="button"
                      onClick={() => props.onLoadFile(file)}
                      title="Ajouter au contexte"
                      className={cx(
                        'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors',
                        loadedPaths.has(file) ? 'text-accent' : 'text-secondary hover:bg-surface-2 hover:text-primary',
                      )}
                    >
                      <FileIcon className="h-3.5 w-3.5 flex-none opacity-60" />
                      <span className="truncate font-mono text-caption">{file}</span>
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <p className="px-1 py-6 text-center text-footnote leading-relaxed text-tertiary">
                Choisis un dossier de travail pour parcourir les fichiers et les ajouter au contexte.
              </p>
            )}
          </div>
        )}
      </div>
    </aside>
  )
}
