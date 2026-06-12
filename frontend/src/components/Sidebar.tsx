import { useMemo, useState } from 'react'

import type { ConversationMeta } from '../types'
import { cx } from '../lib/cx'
import { IconButton } from '../ui'
import { HealthMenu } from './HealthMenu'
import {
  DownloadIcon,
  FolderIcon,
  GearIcon,
  MoonIcon,
  PencilIcon,
  PinIcon,
  SearchIcon,
  SidebarIcon,
  SquarePenIcon,
  SunIcon,
  TrashIcon,
} from './Icons'

interface SidebarProps {
  conversations: ConversationMeta[]
  activeId: string
  query: string
  theme: 'light' | 'dark'
  onQueryChange: (query: string) => void
  onNew: () => void
  onSelect: (id: string) => void
  onRename: (id: string, title: string) => void
  onDelete: (id: string) => void
  onPin: (id: string, pinned: boolean) => void
  onExport: (id: string) => void
  onCollapse: () => void
  onToggleTheme: () => void
  onOpenSettings: () => void
}

function projectLabel(workdir: string): string {
  if (!workdir) return 'Sans projet'
  return workdir.split(/[\\/]/).filter(Boolean).pop() || workdir
}

export function Sidebar({
  conversations,
  activeId,
  query,
  theme,
  onQueryChange,
  onNew,
  onSelect,
  onRename,
  onDelete,
  onPin,
  onExport,
  onCollapse,
  onToggleTheme,
  onOpenSettings,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftTitle, setDraftTitle] = useState('')

  const groups = useMemo(() => {
    const map = new Map<string, ConversationMeta[]>()
    for (const conversation of conversations) {
      const key = conversation.workdir || ''
      if (!map.has(key)) map.set(key, [])
      map.get(key)!.push(conversation)
    }
    return Array.from(map.entries())
  }, [conversations])

  const showProjectHeaders = groups.some(([workdir]) => workdir !== '')

  function startRename(conversation: ConversationMeta) {
    setEditingId(conversation.id)
    setDraftTitle(conversation.title)
  }

  function commitRename() {
    if (editingId) {
      const title = draftTitle.trim()
      if (title) onRename(editingId, title)
    }
    setEditingId(null)
  }

  function renderItem(conversation: ConversationMeta) {
    const isActive = conversation.id === activeId
    const isEditing = conversation.id === editingId
    return (
      <div
        key={conversation.id}
        className={cx(
          'group flex items-center gap-1 rounded-md pl-2.5 pr-1 transition-colors',
          isActive ? 'bg-accent/10 text-primary' : 'text-secondary hover:bg-surface-2',
        )}
      >
        {isEditing ? (
          <input
            autoFocus
            value={draftTitle}
            onChange={(event) => setDraftTitle(event.target.value)}
            onBlur={commitRename}
            onKeyDown={(event) => {
              if (event.key === 'Enter') commitRename()
              if (event.key === 'Escape') setEditingId(null)
            }}
            className="my-1 w-full rounded border border-hairline-strong bg-surface px-1.5 py-1 text-callout text-primary outline-none focus:border-accent/70"
          />
        ) : (
          <button
            type="button"
            onClick={() => onSelect(conversation.id)}
            onDoubleClick={() => startRename(conversation)}
            className="flex h-9 flex-1 items-center gap-1.5 truncate text-left text-callout"
            title={conversation.title}
          >
            {conversation.pinned && <PinIcon className="h-3 w-3 flex-none text-accent" />}
            <span className="truncate">{conversation.title}</span>
          </button>
        )}

        {!isEditing && (
          <div className="flex flex-none items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
            <RowAction
              label={conversation.pinned ? 'Désépingler' : 'Épingler'}
              onClick={() => onPin(conversation.id, !conversation.pinned)}
              active={conversation.pinned}
            >
              <PinIcon className="h-3.5 w-3.5" />
            </RowAction>
            <RowAction label="Exporter (Markdown)" onClick={() => onExport(conversation.id)}>
              <DownloadIcon className="h-3.5 w-3.5" />
            </RowAction>
            <RowAction label="Renommer" onClick={() => startRename(conversation)}>
              <PencilIcon className="h-3.5 w-3.5" />
            </RowAction>
            <RowAction
              label="Supprimer"
              danger
              onClick={() => {
                if (window.confirm(`Supprimer « ${conversation.title} » ?`)) onDelete(conversation.id)
              }}
            >
              <TrashIcon className="h-3.5 w-3.5" />
            </RowAction>
          </div>
        )}
      </div>
    )
  }

  return (
    <aside className="flex w-[272px] flex-none flex-col border-r border-hairline bg-panel">
      <div className="flex items-center gap-2.5 px-3 pb-1 pt-3">
        <span className="flex-1 truncate text-callout font-semibold text-primary">
          Lity
        </span>
        <IconButton size="sm" label="Replier la barre latérale" onClick={onCollapse}>
          <SidebarIcon className="h-4 w-4" />
        </IconButton>
      </div>

      <div className="px-3 pt-2">
        <button
          type="button"
          onClick={onNew}
          title="Nouvelle conversation (⌘N)"
          className="group flex h-9 w-full items-center gap-2.5 rounded-lg border border-hairline bg-surface px-3 text-callout font-medium text-primary shadow-xs transition-colors hover:bg-surface-2 active:scale-[0.99]"
        >
          <SquarePenIcon className="h-4 w-4 flex-none text-accent" />
          <span className="flex-1 truncate whitespace-nowrap text-left">Nouvelle conversation</span>
        </button>
      </div>

      <div className="px-3 py-3">
        <div className="flex h-9 items-center gap-2 rounded-md border border-hairline bg-surface px-2.5 transition-colors focus-within:border-accent/70">
          <SearchIcon className="h-4 w-4 flex-none text-tertiary" />
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Rechercher…"
            className="w-full bg-transparent text-callout text-primary outline-none placeholder:text-tertiary"
          />
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 pb-2">
        {conversations.length === 0 && (
          <p className="px-3 py-8 text-center text-footnote text-tertiary">Aucune conversation</p>
        )}
        {showProjectHeaders ? (
          groups.map(([workdir, items]) => (
            <div key={workdir || '__none__'} className="mb-3">
              <div className="flex items-center gap-1.5 px-2.5 pb-1 pt-1 text-caption font-medium uppercase tracking-wide text-tertiary">
                <FolderIcon className="h-3 w-3" />
                <span className="truncate" title={workdir}>
                  {projectLabel(workdir)}
                </span>
              </div>
              <div className="space-y-px">{items.map(renderItem)}</div>
            </div>
          ))
        ) : (
          <div className="space-y-px">{conversations.map(renderItem)}</div>
        )}
      </nav>

      <div className="flex items-center gap-1 border-t border-hairline px-2 py-2">
        <HealthMenu />
        <IconButton size="sm" label={theme === 'dark' ? 'Passer en clair' : 'Passer en sombre'} onClick={onToggleTheme}>
          {theme === 'dark' ? <SunIcon className="h-4 w-4" /> : <MoonIcon className="h-4 w-4" />}
        </IconButton>
        <IconButton size="sm" label="Réglages" onClick={onOpenSettings}>
          <GearIcon className="h-4 w-4" />
        </IconButton>
      </div>
    </aside>
  )
}

function RowAction({
  label,
  onClick,
  danger,
  active,
  children,
}: {
  label: string
  onClick: () => void
  danger?: boolean
  active?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className={cx(
        'rounded p-1 transition-colors',
        danger
          ? 'text-tertiary hover:text-danger'
          : active
            ? 'text-accent'
            : 'text-tertiary hover:text-primary',
      )}
    >
      {children}
    </button>
  )
}
